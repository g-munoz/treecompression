import gurobipy as gp
from gurobipy import GRB
import json
from copy import deepcopy
from math import floor, ceil
from dataclasses import dataclass
from typing import Union, Callable, Tuple, List

COMPRESSED = 0
UNCOMPRESSABLE = 1


@dataclass
class Direction:
    pi: Union[gp.Var, gp.LinExpr]
    pi_0: float


class DropCompression:
    """
    Compression method that deletes all children of a node if the value of its LP
    relaxation is already as good the dual bound provided by the entire tree.
    """

    def compress(self, mps_filename: str, tree: dict) -> Tuple[dict, dict]:
        tree = deepcopy(tree)
        stats = _stats_init(tree)
        globalbnd = tree["nodes"]["0"]["subtree_bound"]

        def callback(node_id: str) -> int:
            subtreebnd = float(tree["nodes"][node_id]["obj"])
            if (tree["sense"] == "min" and subtreebnd >= globalbnd) or (
                tree["sense"] == "max" and subtreebnd <= globalbnd
            ):
                tree["nodes"][node_id]["compression_method"] = "drop"
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

    def compress(self, mps_filename: str, tree: dict) -> Tuple[dict, dict]:
        tree = deepcopy(tree)
        stats = _stats_init(tree)

        model = gp.read(mps_filename)
        model.params.OutputFlag = 0
        model.params.Threads = 1
        vtype = model.getAttr("vtype", model.getVars())
        sense = model.modelSense
        model = model.relax()
        globalbnd = tree["nodes"]["0"]["subtree_bound"]

        def callback(node_id: str) -> int:
            # Solve LP relaxation of the node
            original_bounds = _set_bounds(model, tree, node_id)
            model.optimize()
            x = model.getVars()
            xv = model.getAttr("x", x)
            n = len(xv)

            # Find fractional vars
            frac_idx = [i for i in range(n) if vtype[i] != "C" and frac(xv[i]) > 0.01]

            # Build candidate directions
            directions = []
            for i in frac_idx:
                directions.append(Direction(pi=x[i], pi_0=frac(xv[i],)))

            # Evaluate directions
            best_dir, best_val = _evaluate(model, directions)

            # Restore bounds
            _restore_bounds(model, original_bounds)

            if (tree["sense"] == "min" and best_val >= globalbnd) or (
                tree["sense"] == "max" and best_val <= globalbnd
            ):
                tree["nodes"][node_id]["compression_method"] = "sb"
                tree["nodes"][node_id]["compression_bound"] = best_val
                tree["nodes"][node_id]["compression_var"] = best_dir.pi.varName
                return COMPRESSED
            else:
                return UNCOMPRESSABLE

        _down_search(tree, stats, callback)
        _stats_finalize(stats, tree)
        return tree, stats


# -----------------------------------------------------------------------------


def _down_search(tree: dict, stats: dict, callback: Callable) -> None:
    pending = ["0"]
    while len(pending) > 0:
        node_id = pending.pop()
        stats["nodes_visited"] += 1
        result = callback(node_id)
        if result == COMPRESSED:
            _drop_subtree(tree, node_id, stats)
        elif result == UNCOMPRESSABLE:
            for child_id in tree["nodes"][node_id]["children"]:
                # Skip leaf nodes
                if len(tree["nodes"][child_id]["children"]) == 0:
                    continue
                pending.append(child_id)
        else:
            raise Exception(f"Unknown callback result: {result}")


def _drop_subtree(tree: dict, root_id: str, stats: dict) -> None:
    # Drop all descendents of the root node
    stack = []
    for child in tree["nodes"][root_id]["children"]:
        stack.append(child)
    while len(stack) > 0:
        node_id = stack.pop()
        for child in tree["nodes"][node_id]["children"]:
            stack.append(child)
        del tree["nodes"][node_id]
        stats["nodes_dropped"] += 1

    # Update root node
    root = tree["nodes"][root_id]
    root["children"] = []
    root["subtree_bound"] = root["obj"]
    root["subtree_support"] = []
    root["compressed"] = True


def _stats_init(tree: dict) -> dict:
    return {
        "nodes_visited": 0,
        "nodes_dropped": 0,
        "nodes_before": len(tree["nodes"]),
    }


def _stats_finalize(stats: dict, tree: dict) -> None:
    stats["nodes_after"] = len(tree["nodes"])
    stats["compression_ratio"] = stats["nodes_before"] / stats["nodes_after"]


def _set_bounds(
    model: gp.Model, tree: dict, node_id: str
) -> List[Tuple[gp.Var, float, float]]:
    """
    This function goes up the tree collecting branching bounds, and setting them
    in the model. This allows us to solve the LP relaxation at a given node. The
    function also returns the original bounds, so that they can be restored.
    """
    original_bounds = []
    while node_id != "0":
        var = model.getVarByName(tree["nodes"][node_id]["branch_var"])
        original_bounds.append((var, var.lb, var.ub))
        var.lb = tree["nodes"][node_id]["branch_lb"]
        var.ub = tree["nodes"][node_id]["branch_ub"]
        node_id = tree["nodes"][node_id]["parent"]
    return original_bounds


def _restore_bounds(
    model: gp.Model, original_bounds: List[Tuple[gp.Var, float, float]],
) -> None:
    for (var, lb, ub) in original_bounds:
        var.lb = lb
        var.ub = ub


def frac(x: float) -> float:
    return x - floor(x)


def _evaluate(model: gp.Model, directions: List[Direction]) -> Tuple[Direction, float]:
    sense = model.modelSense
    model.optimize()
    all_vars = model.getVars()
    all_constrs = model.getConstrs()
    vbasis = model.getAttr("vbasis", all_vars)
    cbasis = model.getAttr("cbasis", all_constrs)

    def _value(constr: gp.TempConstr) -> float:
        model.setAttr("vbasis", all_vars, vbasis)
        model.setAttr("cbasis", all_constrs, cbasis)
        c = model.addConstr(constr)
        model.optimize()
        if model.status == GRB.OPTIMAL:
            val = model.objBound
        elif model.status == GRB.INFEASIBLE:
            val = float("inf") * sense
        else:
            raise Exception(f"Unknown model status: {model.status}")
        model.remove(c)
        return val

    best_val, best_dir = float("-inf"), None
    for d in directions:
        left_val = _value(d.pi <= floor(d.pi_0))
        right_val = _value(d.pi >= ceil(d.pi_0))
        d_val = sense * min(sense * left_val, sense * right_val)
        if d_val > best_val:
            best_val = d_val
            best_dir = d

    assert best_dir is not None
    return best_dir, best_val
