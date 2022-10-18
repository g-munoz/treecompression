import json
import logging
import re
import sys
from dataclasses import dataclass
from distutils.dir_util import mkpath
from glob import glob
from math import floor
from os.path import exists, dirname
from queue import Queue
from random import shuffle
from time import time
from typing import Union, Callable, Tuple, List, Dict, Optional

import gurobipy as gp
import h5py
import numpy as np
from gurobipy import GRB
from p_tqdm import p_umap

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
logger.addHandler(stdout_handler)

COMPRESSED = 0
UNCOMPRESSABLE = 1

Expr = Union[gp.Var, gp.LinExpr]


@dataclass
class Direction:
    pi: Expr
    pi_0: float


# -----------------------------------------------------------------------------


class DropCompression:
    """
    Compression method that deletes all children of a node if the value of its LP
    relaxation is already as good the dual bound provided by the entire tree.
    """

    def compress(self, model_filename: str, tree_filename: str) -> Tuple[dict, dict]:
        tree = _read_json(tree_filename)
        stats = _stats_init(model_filename, tree_filename, tree)
        self._compress_inplace(tree, stats)
        return tree, stats

    @staticmethod
    def _compress_inplace(tree: dict, stats: dict) -> None:
        globalbnd = tree["nodes"]["0"]["subtree_bound"]
        sense = 1 if tree["sense"] == "min" else -1

        def callback(node_id: str) -> int:
            if sense * tree["nodes"][node_id]["obj"] >= sense * globalbnd:
                _drop(tree, node_id)
                return COMPRESSED
            else:
                return UNCOMPRESSABLE

        _down_search(tree, stats, callback)
        _stats_finalize(stats, tree)


# -----------------------------------------------------------------------------


class StrongBranchCompression:
    """
    Compression method that evaluates all single-variable directions.
    """

    def __init__(self, time_limit: float = 30):
        self.time_limit = time_limit

    def compress(self, model_filename: str, tree_filename: str) -> Tuple[dict, dict]:
        initial_time = time()
        tree = _read_json(tree_filename)
        stats = _stats_init(model_filename, tree_filename, tree)

        # Run DropCompression first
        drop_stats = _stats_init(model_filename, tree_filename, tree)
        DropCompression()._compress_inplace(tree, drop_stats)

        model = _read_model(model_filename)
        vtype = model.getAttr("vtype", model.getVars())
        sense = model.modelSense
        model = model.relax()
        globalbnd = tree["nodes"]["0"]["subtree_bound"]

        def callback(node_id: str) -> int:
            # Check available time
            elapsed_time = time() - initial_time
            if elapsed_time > self.time_limit:
                raise TimeoutError()

            # Check if node can be dropped
            if sense * tree["nodes"][node_id]["obj"] >= sense * globalbnd:
                _drop(tree, node_id)
                return COMPRESSED

            # Solve LP relaxation of the node
            original_bounds = _set_bounds(model, tree, node_id)
            _optimize(model, initial_time, self.time_limit)
            x = model.getVars()
            xv = model.getAttr("x", x)

            # Find fractional vars
            frac_vars = _find_frac_vars(xv, vtype)
            if len(frac_vars) == 0:
                _restore_bounds(model, original_bounds)
                return UNCOMPRESSABLE

            # Build candidate directions
            directions = []
            for (_, i) in frac_vars:
                directions.append(Direction(pi=x[i], pi_0=floor(xv[i])))

            # Evaluate directions
            elapsed_time = time() - initial_time
            best_dir, best_val, best_vals = _evaluate(
                model, directions, stats, self.time_limit - elapsed_time
            )

            # Restore bounds
            _restore_bounds(model, original_bounds)

            if sense * best_val >= sense * globalbnd:
                assert best_dir is not None
                assert best_vals is not None
                _replace(tree, node_id, best_dir.pi, best_dir.pi_0, best_vals)
                return COMPRESSED
            else:
                return UNCOMPRESSABLE

        _down_search(tree, stats, callback)
        _stats_finalize(stats, tree)
        return tree, stats


# -----------------------------------------------------------------------------


class PairsCompression:
    """
    Compression method that evaluates directions based on pairs of fractional
    decision variables (such as x1 + x2, or x1 - x2).
    """

    def __init__(self, time_limit: float = 30):
        self.time_limit = time_limit

    def compress(self, model_filename: str, tree_filename: str) -> Tuple[dict, dict]:
        initial_time = time()
        tree = _read_json(tree_filename)
        stats = _stats_init(model_filename, tree_filename, tree)

        # Run DropCompression first
        drop_stats = _stats_init(model_filename, tree_filename, tree)
        DropCompression()._compress_inplace(tree, drop_stats)

        model = _read_model(model_filename)
        vtype = model.getAttr("vtype", model.getVars())
        sense = model.modelSense
        model = model.relax()
        globalbnd = tree["nodes"]["0"]["subtree_bound"]

        def callback(node_id: str) -> int:
            # Check available time
            elapsed_time = time() - initial_time
            if elapsed_time > self.time_limit:
                raise TimeoutError()

            # Check if node can be dropped
            if sense * tree["nodes"][node_id]["obj"] >= sense * globalbnd:
                _drop(tree, node_id)
                return COMPRESSED

            # Solve LP relaxation of the node
            original_bounds = _set_bounds(model, tree, node_id)
            _optimize(model, initial_time, self.time_limit)
            x = model.getVars()
            xv = model.getAttr("x", x)

            # Find fractional vars
            frac_vars = _find_frac_vars(xv, vtype)
            logger.debug(f"Found {len(frac_vars)} frac vars")

            # # Filter fractional variables
            # frac_vars = [
            #     (score, var)
            #     for (score, var) in frac_vars
            #     if x[var].varName in tree["nodes"]["0"]["subtree_support"]
            # ]
            # logger.debug(f"Filtered down to {len(frac_vars)} frac vars")

            if len(frac_vars) == 0:
                _restore_bounds(model, original_bounds)
                return UNCOMPRESSABLE

            # Build candidate directions
            directions = []
            for (_, i1) in frac_vars:
                directions.append(Direction(pi=x[i1], pi_0=floor(xv[i1])))
                for (_, i2) in frac_vars:
                    if i2 <= i1:
                        continue
                    for pi in [x[i1] + x[i2], x[i1] - x[i2]]:
                        pi_0 = floor(pi.getValue())
                        directions.append(Direction(pi=pi, pi_0=pi_0))

            # Evaluate directions
            elapsed_time = time() - initial_time
            best_dir, best_val, best_vals = _evaluate(
                model, directions, stats, self.time_limit - elapsed_time
            )

            # Restore bounds
            _restore_bounds(model, original_bounds)

            if sense * best_val >= sense * globalbnd:
                assert best_dir is not None
                assert best_vals is not None
                _replace(tree, node_id, best_dir.pi, best_dir.pi_0, best_vals)
                return COMPRESSED
            else:
                return UNCOMPRESSABLE

        _down_search(tree, stats, callback)
        _stats_finalize(stats, tree)
        return tree, stats


# -----------------------------------------------------------------------------


class OweMeh2001Compression:
    """
    Compression method based on the heuristic procedure by:

        Owen, J. H., & Mehrotra, S. (2001). Experimental results on using
        general disjunctions in branch-and-bound for general-integer linear
        programs. Computational optimization and applications, 20(2), 159-170.

    """

    def __init__(
        self,
        tree_search,
        time_limit: float = 30,
        max_vars: int = 1_000_000,
        max_iterations=1_000,
    ):
        self.time_limit = time_limit
        self.max_vars = max_vars
        self.max_iterations = max_iterations
        self.tree_search = tree_search

    def compress(self, model_filename: str, tree_filename: str) -> Tuple[dict, dict]:
        initial_time = time()
        tree = _read_json(tree_filename)
        stats = _stats_init(model_filename, tree_filename, tree)

        # Run DropCompression first
        drop_stats = _stats_init(model_filename, tree_filename, tree)
        DropCompression()._compress_inplace(tree, drop_stats)

        # Load pseudocosts
        def _fix_nan(a):
            a[np.isnan(a)] = np.nanmean(a)

        h5_filename = model_filename.replace(".mps.gz", ".h5")
        h5 = h5py.File(h5_filename, "r")
        var_pseudocost_up = np.array(h5["bb_var_pseudocost_up"])
        var_pseudocost_down = np.array(h5["bb_var_pseudocost_down"])
        h5.close()
        _fix_nan(var_pseudocost_up)
        _fix_nan(var_pseudocost_down)

        def _filter_vars(frac_vars):
            score = [
                (
                    var_pseudocost_up[var_idx]
                    * var_frac
                    * var_pseudocost_down[var_idx]
                    * (1 - var_frac),
                    var_frac,
                    var_idx,
                )
                for (var_frac, var_idx) in frac_vars
            ]
            score.sort(reverse=True)
            score = score[: min(len(score), self.max_vars)]
            return [(frac, idx) for (_, frac, idx) in score]

        model = _read_model(model_filename)
        vtype = model.getAttr("vtype", model.getVars())
        sense = model.modelSense
        model = model.relax()
        globalbnd = tree["nodes"]["0"]["subtree_bound"]

        def callback(node_id: str) -> int:
            # Check available time
            elapsed_time = time() - initial_time
            if elapsed_time > self.time_limit:
                raise TimeoutError()

            # Check if node can be dropped
            if sense * tree["nodes"][node_id]["obj"] >= sense * globalbnd:
                _drop(tree, node_id)
                return COMPRESSED

            # Solve LP relaxation of the node
            original_bounds = _set_bounds(model, tree, node_id)
            _optimize(model, initial_time, self.time_limit)
            x = model.getVars()
            xv = model.getAttr("x", x)

            # Create directions
            frac_vars = _find_frac_vars(xv, vtype)
            frac_vars = _filter_vars(frac_vars)
            directions = []
            for (_, i) in frac_vars:
                directions.append(Direction(pi=x[i], pi_0=floor(xv[i])))

            best_val, best_dir, best_vals = float("-inf"), None, None

            for iteration in range(self.max_iterations):
                if len(directions) == 0:
                    break

                elapsed_time = time() - initial_time
                curr_dir, curr_val, (curr_left_val, curr_right_val) = _evaluate(
                    model,
                    directions,
                    stats,
                    time_limit=self.time_limit - elapsed_time,
                    target=globalbnd,
                )
                if sense * curr_val > sense * best_val:
                    best_val = curr_val
                    best_dir = curr_dir
                    best_vals = (curr_left_val, curr_right_val)
                else:
                    break

                if sense * best_val >= sense * globalbnd:
                    break

                # Re-solve LP relaxation of the worst side of the best direction
                if curr_val == curr_left_val:
                    c = model.addConstr(curr_dir.pi <= curr_dir.pi_0)
                else:
                    c = model.addConstr(curr_dir.pi >= curr_dir.pi_0 + 1)

                _optimize(model, initial_time, self.time_limit)
                xv = model.getAttr("x", x)

                # Construct new set of candidate directions
                frac_vars = _find_frac_vars(xv, vtype)
                frac_vars = _filter_vars(frac_vars)
                directions = []
                for (_, i) in frac_vars:
                    for pi in [curr_dir.pi + x[i], curr_dir.pi - x[i]]:
                        pi_0 = floor(pi.getValue())
                        directions.append(Direction(pi=pi, pi_0=pi_0))

                model.remove(c)
                model.update()

            _restore_bounds(model, original_bounds)

            if sense * best_val >= sense * globalbnd:
                assert best_dir is not None
                assert best_vals is not None
                _replace(tree, node_id, best_dir.pi, best_dir.pi_0, best_vals)
                return COMPRESSED
            else:
                return UNCOMPRESSABLE

        self.tree_search(tree, stats, callback)
        _stats_finalize(stats, tree)
        return tree, stats


# -----------------------------------------------------------------------------


class MahChi2013Compression:
    """
    Compression method that: (1) finds tight constraints in the node LP to be
    used as "foundational constraints"; (2) builds a direction that is
    approximately parallel to these foundational constraints by taking the sign
    of each coefficient; (3) builds a perpendicular direction to the parallel
    ones. Based on the method proposed by:

        Mahmoud, H., & Chinneck, J. W. (2013). Achieving MILP feasibility
        quickly using general disjunctions. Computers & operations research,
        40(8), 2094-2102.
    
    """

    def __init__(self, time_limit: float = 30, max_directions: int = 100):
        self.time_limit = time_limit
        self.max_directions = max_directions

    def compress(self, model_filename: str, tree_filename: str) -> Tuple[dict, dict]:
        initial_time = time()
        tree = _read_json(tree_filename)
        stats = _stats_init(model_filename, tree_filename, tree)

        # Run DropCompression first
        drop_stats = _stats_init(model_filename, tree_filename, tree)
        DropCompression()._compress_inplace(tree, drop_stats)

        model = _read_model(model_filename)
        vtype = model.getAttr("vtype", model.getVars())
        sense = model.modelSense
        model = model.relax()
        globalbnd = tree["nodes"]["0"]["subtree_bound"]

        def callback(node_id: str) -> int:
            # Check available time
            elapsed_time = time() - initial_time
            if elapsed_time > self.time_limit:
                raise TimeoutError()

            # Check if node can be dropped
            if sense * tree["nodes"][node_id]["obj"] >= sense * globalbnd:
                _drop(tree, node_id)
                return COMPRESSED

            # Solve LP relaxation of the node
            original_bounds = _set_bounds(model, tree, node_id)
            _optimize(model, initial_time, self.time_limit)

            # Find foundational constraints
            constrs = _find_foundational_constrs(model, vtype)
            if constrs is None:
                return UNCOMPRESSABLE

            # Build parallel directions
            paral_dirs = []
            for constr in constrs:
                pi = gp.LinExpr()
                expr = model.getRow(constr)
                for i in range(expr.size()):
                    var = expr.getVar(i)
                    if vtype[var.index] == "C":
                        continue
                    if expr.getCoeff(i) > 1e-3:
                        pi += var
                    elif expr.getCoeff(i) < -1e-3:
                        pi -= var
                paral_dirs.append(pi)

            # Build perpendicular directions
            directions: List[Direction] = []
            for expr in paral_dirs:
                flip = 1
                pi = gp.LinExpr()
                for i in range(expr.size()):
                    # If we have an odd number of terms, one of them has to be
                    # set to zero. Here, we just set the first one. In the paper,
                    # they have a more complex rule.
                    if i == 0 and (expr.size() % 2) == 1:
                        continue
                    if expr.getCoeff(i) > 0:
                        pi -= expr.getVar(i) * flip
                    else:
                        pi += expr.getVar(i) * flip
                    flip *= -1

                pi_0 = pi.getValue()
                if frac(pi_0) > 1e-5:
                    directions.append(Direction(pi=pi, pi_0=floor(pi_0)))

                if len(directions) >= self.max_directions:
                    break

            # Evaluate directions
            elapsed_time = time() - initial_time
            best_dir, best_val, best_vals = _evaluate(
                model, directions, stats, self.time_limit - elapsed_time
            )

            # Restore bounds
            _restore_bounds(model, original_bounds)

            if sense * best_val >= sense * globalbnd:
                assert best_dir is not None
                assert best_vals is not None
                _replace(tree, node_id, best_dir.pi, best_dir.pi_0, best_vals)
                return COMPRESSED
            else:
                return UNCOMPRESSABLE

        _down_search(tree, stats, callback)
        _stats_finalize(stats, tree)
        return tree, stats


# -----------------------------------------------------------------------------


def _down_search(tree: dict, stats: dict, callback: Callable) -> None:
    initial_time = time()
    pending: Queue = Queue()
    pending.put("0")
    while not pending.empty():
        node_id = pending.get()
        initial_node_time = time()
        logger.debug(f"Visiting node {node_id}")
        stats["nodes_visited"] += 1
        try:
            result = callback(node_id)
        except TimeoutError:
            logger.debug("Time limit reached. Aborting.")
            return
        logger.debug(
            f"Visit to node {node_id} took {time() - initial_node_time:.2f} seconds"
        )
        if result == COMPRESSED:
            logger.debug(f"Node {node_id} compressed")
            pass
        elif result == UNCOMPRESSABLE:
            logger.debug(f"Node {node_id} is uncompressable.")
            for child_id in tree["nodes"][node_id]["children"]:
                # Skip leaf nodes
                if len(tree["nodes"][child_id]["children"]) == 0:
                    continue
                pending.put(child_id)
        else:
            raise Exception(f"Unknown callback result: {result}")
    logger.debug(f"Search finished in {time() - initial_time:,.2f} seconds")
    logger.debug(f"Compressed tree has {len(tree['nodes']):,d} nodes")


def _priority_search(tree: dict, stats: dict, callback: Callable) -> None:
    initial_time = time()
    queue = [
        ((len(node["subtree_support"]), -node["depth"]), node["id"])
        for node in tree["nodes"].values()
    ]
    queue.sort(reverse=True)
    for (score, node_id) in queue:
        initial_node_time = time()
        logger.debug(f"Visiting node {node_id} (score {score})")
        if node_id not in tree["nodes"]:
            logger.debug(f"Node {node_id} has been previously removed.")
            continue
        stats["nodes_visited"] += 1
        try:
            result = callback(node_id)
        except TimeoutError:
            logger.debug("Time limit reached. Aborting.")
            return
        logger.debug(
            f"Visit to node {node_id} took {time() - initial_node_time:.2f} seconds"
        )
        if result == COMPRESSED:
            logger.debug(f"Node {node_id} compressed")
            pass
        elif result == UNCOMPRESSABLE:
            logger.debug(f"Node {node_id} is uncompressable.")
        else:
            raise Exception(f"Unknown callback result: {result}")
    logger.debug(f"Search finished in {time() - initial_time:,.2f} seconds")
    logger.debug(f"Compressed tree has {len(tree['nodes']):,d} nodes")


def _drop(tree: dict, root_id: str) -> None:
    # Drop all descendents of the root node
    stack = []
    for child in tree["nodes"][root_id]["children"]:
        stack.append(child)
    while len(stack) > 0:
        node_id = stack.pop()
        for child in tree["nodes"][node_id]["children"]:
            stack.append(child)
        del tree["nodes"][node_id]

    # Update root node
    root = tree["nodes"][root_id]
    root["children"] = []
    root["subtree_bound"] = root["obj"]
    root["subtree_support"] = []
    root["compressed"] = True


def _replace(
    tree: dict, root_id: str, pi: Expr, pi_0: float, vals: Tuple[float, float]
) -> None:
    _drop(tree, root_id)

    # Add left branch
    left_id = _next_id(tree)
    tree["nodes"][left_id] = {
        "id": left_id,
        "parent": root_id,
        "children": [],
        "depth": tree["nodes"][root_id]["depth"] + 1,
        "obj": vals[0],
        "branch_lhs": _expr_to_dict(pi),
        "branch_lb": -float("inf"),
        "branch_ub": pi_0,
    }

    # Add right branch
    right_id = _next_id(tree)
    tree["nodes"][right_id] = {
        "id": right_id,
        "parent": root_id,
        "children": [],
        "depth": tree["nodes"][root_id]["depth"] + 1,
        "obj": vals[1],
        "branch_lhs": _expr_to_dict(pi),
        "branch_lb": pi_0 + 1,
        "branch_ub": float("inf"),
    }

    tree["nodes"][root_id]["children"] = [left_id, right_id]
    logger.debug(tree["nodes"][left_id])
    logger.debug(tree["nodes"][right_id])


def _next_id(tree: dict) -> str:
    return str(1 + max([int(node["id"]) for node in tree["nodes"].values()]))


def _expr_to_dict(expr: Expr) -> Dict[str, float]:
    if type(expr) == gp.Var:
        return {expr.varName: 1.0}
    else:
        lhs = {}
        for i in range(expr.size()):
            lhs[expr.getVar(i).varName] = expr.getCoeff(i)
        return lhs


def _stats_init(model_filename: str, tree_filename: str, tree: dict) -> dict:
    return {
        "model_filename": model_filename,
        "tree_filename": tree_filename,
        "nodes_visited": 0,
        "nodes_before": len(tree["nodes"]),
        "evaluated_directions": 0,
    }


def _stats_finalize(stats: dict, tree: dict) -> None:
    stats["nodes_after"] = len(tree["nodes"])
    stats["compression_ratio"] = stats["nodes_before"] / stats["nodes_after"]


def _set_bounds(
    model: gp.Model, tree: dict, node_id: str
) -> Dict[str, Tuple[float, float]]:
    """
    This function goes up the tree collecting branching bounds, and setting them
    in the model. This allows us to solve the LP relaxation at a given node. The
    function also returns the original bounds, so that they can be restored.
    """
    original_bounds = {}
    while node_id != "0":
        var = model.getVarByName(tree["nodes"][node_id]["branch_var"])
        if var not in original_bounds:
            original_bounds[var] = (var.lb, var.ub)
        var.lb = max(var.lb, tree["nodes"][node_id]["branch_lb"])
        var.ub = min(var.ub, tree["nodes"][node_id]["branch_ub"])
        node_id = tree["nodes"][node_id]["parent"]
        model.update()
    return original_bounds


def _restore_bounds(
    model: gp.Model, original_bounds: Dict[gp.Var, Tuple[float, float]],
) -> None:
    for (var, (lb, ub)) in original_bounds.items():
        var.lb = lb
        var.ub = ub
    model.update()


def frac(x: float) -> float:
    return x - floor(x)


def _evaluate(
    model: gp.Model,
    directions: List[Direction],
    stats: dict,
    time_limit: float,
    target: float,
) -> Tuple[Direction, float, Tuple[float, float]]:
    logger.debug(f"Evaluating {len(directions)} directions...")
    initial_time = time()
    sense = model.modelSense
    model.params.TimeLimit = time_limit
    model.optimize()
    if model.status == GRB.TIME_LIMIT:
        raise TimeoutError()
    all_vars = model.getVars()
    all_constrs = model.getConstrs()
    vbasis = model.getAttr("vbasis", all_vars)
    cbasis = model.getAttr("cbasis", all_constrs)

    def _value(constr: gp.TempConstr) -> float:
        elapsed_time = time() - initial_time
        if elapsed_time > time_limit:
            raise TimeoutError()
        stats["evaluated_directions"] += 1
        model.setAttr("vbasis", all_vars, vbasis)
        model.setAttr("cbasis", all_constrs, cbasis)
        c = model.addConstr(constr)
        model.params.TimeLimit = time_limit - elapsed_time
        model.optimize()
        if model.status == GRB.TIME_LIMIT:
            raise TimeoutError()
        elif model.status == GRB.OPTIMAL:
            val = model.objBound
        elif model.status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
            val = float("inf") * sense
        else:
            raise Exception(f"Unknown model status: {model.status}")
        model.remove(c)
        return val

    best_val, best_dir, best_vals = float("-inf"), None, None
    for (i, d) in enumerate(directions):
        left_val = _value(d.pi <= d.pi_0)
        right_val = _value(d.pi >= d.pi_0 + 1)
        dir_val = sense * min(sense * left_val, sense * right_val)
        if sense * dir_val > sense * best_val:
            best_val, best_dir, best_vals = dir_val, d, (left_val, right_val)

            # Early stop
            if sense * dir_val >= sense * target:
                logger.debug(
                    f"Direction found. Stopping early, after only {i} evaluations."
                )
                break

    assert best_dir is not None
    assert best_vals is not None
    return best_dir, best_val, best_vals


def _read_json(filename: str) -> dict:
    logger.debug(f"Reading: {filename}")
    with open(filename) as f:
        return json.load(f)


def _read_model(filename: str) -> gp.Model:
    logger.debug(f"Reading: {filename}")
    model = gp.read(filename)
    model.params.OutputFlag = 0
    model.params.Threads = 1
    return model


def _optimize(model: gp.Model, initial_time: float, time_limit: float) -> None:
    elapsed_time = time() - initial_time
    model.params.TimeLimit = max(0, time_limit - elapsed_time)
    model.optimize()
    if model.status == GRB.TIME_LIMIT:
        raise TimeoutError()


def _find_frac_vars(
    xv: List[float], vtype: List[str], atol: float = 1e-5,
) -> List[Tuple[float, int]]:
    return [
        (frac(xv[i]), i)
        for i in range(len(xv))
        if vtype[i] != "C" and frac(xv[i]) > atol
    ]


def _find_foundational_constrs(
    model: gp.Model, vtype: List[str], atol: float = 1e-5
) -> Optional[gp.Constr]:
    """
    Select a foundational constraint, which will serve as basis for the general
    direction, based on the method proposed by  Mahmoud, H., & Chinneck, J. W.
    (2013).
    """

    xv = model.getAttr("x", model.getVars())
    candidate_vars = set(
        var
        for (i, var) in enumerate(model.getVars())
        if vtype[i] != "C" and frac(var.x) > atol
    )

    def count_candidate_vars(constr: List[gp.Constr]) -> int:
        """Count number of fractional variables in the constraints"""
        count = 0
        expr = model.getRow(constr)
        for i in range(expr.size()):
            if expr.getVar(i) in candidate_vars:
                count += 1
        return count

    def count_int_vars(constr: List[gp.Constr]) -> int:
        """Count number of integer variables in the constraint."""
        count = 0
        expr = model.getRow(constr)
        for i in range(expr.size()):
            if vtype[expr.getVar(i).index] != "C":
                count += 1
        return count

    candidate_constrs = []
    for constr in model.getConstrs():
        if constr.sense == "=" or abs(constr.slack) < atol:
            candidate_constrs.append(constr)

    if len(candidate_constrs) == 0:
        return None

    candidates = sorted(
        [
            (count_candidate_vars(constr), count_int_vars(constr), i, constr,)
            for (i, constr) in enumerate(candidate_constrs)
        ],
        reverse=True,
    )

    return [c[3] for c in candidates]


def _write_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------------------------------------------------------


def run(time_limit: float = 900) -> None:
    def _sample(args: Tuple) -> dict:
        model_filename, method_name, method = args

        # Compute filenames
        instance = re.search("instances/models/(.*)\.mps\.gz", model_filename).group(1)
        in_tree_filename = f"instances/trees/RB/{instance}.tree.json"
        stats_filename = f"results/{method_name}/{instance}.stats.json"
        out_tree_filename = f"results/{method_name}/{instance}.tree.json"
        log_filename = f"results/{method_name}/{instance}.log"

        # Skip instances that do not have the corresponding tree file
        if not exists(in_tree_filename):
            return

        # Skip instances that have already been processed
        if exists(stats_filename):
            return

        # Create folder
        mkpath(dirname(stats_filename))

        # Set up logger
        file_handler = logging.FileHandler(log_filename, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(file_handler)

        try:
            # Run method
            initial_time = time()
            tree, stats = method.compress(model_filename, in_tree_filename)

            # Record additional information
            stats["time"] = time() - initial_time
            stats["method"] = method_name
            stats["time_limit"] = time_limit

            # Write all output
            _write_json(stats, stats_filename)
            _write_json(tree, out_tree_filename)
        except:
            logger.exception(f"Failed sample: {model_filename} {in_tree_filename}")

        # Tear down logging
        logger.removeHandler(file_handler)

    # Run benchmarks
    combinations = [
        (model_filename, method_name, method)
        for model_filename in glob("instances/models/miplib2017/*.mps.gz")
        for (method_name, method) in {
            "drop": DropCompression(),
            # "StrongBranch": StrongBranchCompression(time_limit=time_limit),
            "revised": OweMeh2001Compression(
                time_limit=time_limit,
                max_vars=5,
                max_iterations=3,
                tree_search=_priority_search,
            ),
            # "best": OweMeh2001Compression(
            #     time_limit=time_limit,
            #     max_vars=1_000_000,
            #     max_iterations=1_000_000,
            #     tree_search=_priority_search,
            # ),
            "original": OweMeh2001Compression(
                time_limit=time_limit,
                max_vars=1_000_000,
                max_iterations=1_000_000,
                tree_search=_down_search,
            ),
            # "Pairs": PairsCompression(time_limit=time_limit),
            # "MahChi2013": MahChi2013Compression(time_limit=time_limit),
        }.items()
    ]
    shuffle(combinations)
    p_umap(_sample, combinations, smoothing=0)


if __name__ == "__main__":
    run()
