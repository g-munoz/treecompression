import gurobipy as gp
from gurobipy import GRB
import json
from copy import deepcopy
from math import floor, ceil
from dataclasses import dataclass
from typing import Union, Callable, Tuple, List, Dict
from p_tqdm import p_umap
from glob import glob
import pandas as pd
from time import time
from os.path import exists
from random import shuffle

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
        stats["method"] = "Drop"
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
        return tree, stats


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
        stats["method"] = "StrongBranch"

        model = gp.read(model_filename)
        model.params.OutputFlag = 0
        model.params.Threads = 1
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
            elapsed_time = time() - initial_time
            model.params.timeLimit = self.time_limit - elapsed_time
            model.optimize()
            if model.status == GRB.TIME_LIMIT:
                raise TimeoutError()
            x = model.getVars()
            xv = model.getAttr("x", x)
            n = len(xv)

            # Find fractional vars
            candidates = [
                (frac(xv[i]), i)
                for i in range(n)
                if vtype[i] != "C" and frac(xv[i]) > 1e-3
            ]
            if len(candidates) == 0:
                _restore_bounds(model, original_bounds)
                return UNCOMPRESSABLE

            # Build candidate directions
            directions = []
            for (_, i) in candidates:
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


class OweMeh2001Compression:
    """
    Compression method based on the heuristic procedure by:

        Owen, J. H., & Mehrotra, S. (2001). Experimental results on using
        general disjunctions in branch-and-bound for general-integer linear
        programs. Computational optimization and applications, 20(2), 159-170.

    """

    def __init__(self, time_limit: float = 30):
        self.time_limit = time_limit

    def compress(self, model_filename: str, tree_filename: str) -> Tuple[dict, dict]:
        initial_time = time()
        tree = _read_json(tree_filename)
        stats = _stats_init(model_filename, tree_filename, tree)
        stats["method"] = "OweMeh2001"

        model = gp.read(model_filename)
        model.params.OutputFlag = 0
        model.params.Threads = 1
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
            elapsed_time = time() - initial_time
            model.params.TimeLimit = self.time_limit - elapsed_time
            model.optimize()
            if model.status == GRB.TIME_LIMIT:
                raise TimeoutError()
            x = model.getVars()
            xv = model.getAttr("x", x)
            n = len(xv)

            # Find best single-variable branching direction
            candidates = [
                (frac(xv[i]), i)
                for i in range(n)
                if vtype[i] != "C" and frac(xv[i]) > 0.01
            ]

            # Create directions
            directions = []
            for (_, i) in candidates:
                directions.append(Direction(pi=x[i], pi_0=floor(xv[i])))

            best_val, best_dir, best_vals = float("-inf"), None, None

            for iteration in range(10):
                if len(directions) == 0:
                    break

                elapsed_time = time() - initial_time
                curr_dir, curr_val, (curr_left_val, curr_right_val) = _evaluate(
                    model, directions, stats, self.time_limit - elapsed_time
                )
                if sense * curr_val > sense * best_val:
                    best_val = curr_val
                    best_dir = curr_dir
                    best_vals = (curr_left_val, curr_right_val)
                else:
                    break

                # Re-solve LP relaxation of the worst side of the best direction
                if curr_val == curr_left_val:
                    c = model.addConstr(curr_dir.pi <= curr_dir.pi_0)
                else:
                    c = model.addConstr(curr_dir.pi >= curr_dir.pi_0 + 1)

                elapsed_time = time() - initial_time
                model.params.TimeLimit = self.time_limit - elapsed_time
                model.optimize()
                if model.status == GRB.TIME_LIMIT:
                    raise TimeoutError()
                xv = model.getAttr("x", x)

                # Construct new set of candidate directions
                candidates = [
                    (frac(xv[i]), i)
                    for i in range(n)
                    if vtype[i] != "C" and frac(xv[i]) > 0.01
                ]

                directions = []
                for (_, i) in candidates:
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

        _down_search(tree, stats, callback)
        _stats_finalize(stats, tree)
        return tree, stats


# -----------------------------------------------------------------------------


def run(time_limit: float = 3600) -> None:
    def _sample(args: Tuple) -> dict:
        model_filename, method = args
        tree_filename = model_filename.replace("models/", "trees/RB/").replace(
            ".mps.gz", ".tree.json"
        )
        try:
            initial_time = time()
            _, stats = method.compress(model_filename, tree_filename)
            stats["time"] = time() - initial_time
            return stats
        except:
            return {}

    combinations = [
        (model_filename, method)
        for model_filename in glob("instances/models/miplib3/*.mps.gz")
        for method in [
            DropCompression(),
            StrongBranchCompression(time_limit=time_limit),
            OweMeh2001Compression(time_limit=time_limit),
        ]
    ]
    shuffle(combinations)
    stats = p_umap(_sample, combinations, smoothing=0)
    # stats = [_sample(args) for args in combinations]
    pd.DataFrame(stats).to_csv("results.csv")


# -----------------------------------------------------------------------------


def _down_search(tree: dict, stats: dict, callback: Callable) -> None:
    pending = ["0"]
    while len(pending) > 0:
        node_id = pending.pop()
        stats["nodes_visited"] += 1
        try:
            result = callback(node_id)
        except TimeoutError:
            return
        if result == COMPRESSED:
            pass
        elif result == UNCOMPRESSABLE:
            for child_id in tree["nodes"][node_id]["children"]:
                # Skip leaf nodes
                if len(tree["nodes"][child_id]["children"]) == 0:
                    continue
                pending.append(child_id)
        else:
            raise Exception(f"Unknown callback result: {result}")


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
    model: gp.Model, directions: List[Direction], stats: dict, time_limit: float,
) -> Tuple[Direction, float, Tuple[float, float]]:
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
    for d in directions:
        left_val = _value(d.pi <= d.pi_0)
        right_val = _value(d.pi >= d.pi_0 + 1)
        dir_val = sense * min(sense * left_val, sense * right_val)
        if dir_val > best_val:
            best_val, best_dir, best_vals = dir_val, d, (left_val, right_val)

    assert best_dir is not None
    assert best_vals is not None
    return best_dir, best_val, best_vals


def _read_json(filename: str) -> dict:
    with open(filename) as f:
        return json.load(f)
