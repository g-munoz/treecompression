import json
from copy import deepcopy
import matplotlib.pyplot as plt
from random import shuffle, random
from os.path import basename
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
import numpy as np
from tqdm.auto import tqdm
from xgboost import XGBClassifier
from p_tqdm import p_umap

SHOULD_MAKE_TIMES_UNIFORM = False

_precision = 2


def highlight_max(row, props=""):
    return np.where(
        row.round(_precision) == np.nanmax(row.values).round(_precision),
        props,
        "",
    )


def highlight_min(row, props=""):
    return np.where(
        row.round(_precision) == np.nanmin(row.values).round(_precision),
        props,
        "",
    )


def default_style(s):
    s.format(
        precision=_precision,
        na_rep="&mdash;",
        thousands=",",
    )
    s.set_table_styles(
        [
            {
                "selector": "td,th",
                "props": "border: 1px solid black; text-align: center",
            },
            {
                "selector": "td.data",
                "props": "text-align: right; font-family: 'monospace'",
            },
            {
                "selector": "th",
                "props": "background-color: #D0D0D0 !important; font-weight: normal",
            },
            {
                "selector": "th.row_heading.level1, th.col_heading.level1",
                "props": "background-color: #E0E0E0 !important;",
            },
            {
                "selector": ".index_name,.col_heading.level0",
                "props": "font-weight: bold",
            },
            {
                "selector": ".blank",
                "props": "background-color: white",
            },
        ],
    )
    return s


def _next_id(tree: dict) -> str:
    return str(1 + max([int(node["id"]) for node in tree["nodes"].values()]))


def _drop(tree: dict, root_id: str) -> None:
    # Drop all descendents of the root node
    stack = []
    for child in tree["nodes"][root_id]["children"]:
        stack.append(child)
    while len(stack) > 0:
        node_id = stack.pop()
        if node_id not in tree["nodes"]:
            continue
        for child in tree["nodes"][node_id]["children"]:
            stack.append(child)
        del tree["nodes"][node_id]

    # Update root node
    root = tree["nodes"][root_id]
    root["children"] = []
    root["subtree_bound"] = root["obj"]
    root["compressed?"] = False


def _replace(tree: dict, root_id: str) -> None:
    _drop(tree, root_id)
    left_id = _next_id(tree)
    right_id = _next_id(tree)
    tree["nodes"][left_id] = {
        "id": left_id,
        "parent": root_id,
        "children": [],
    }
    tree["nodes"][right_id] = {
        "id": right_id,
        "parent": root_id,
        "children": [],
    }
    tree["nodes"][root_id]["children"] = [left_id, right_id]


def simulate(
    tree,
    node_seq,
    time_limit,
):
    times = [0]
    sizes = [len(tree["nodes"])]

    tree = deepcopy(tree)
    current_time = 0
    for node_id in node_seq:
        # Skip nodes that have already been removed
        if node_id not in tree["nodes"]:
            continue

        node = tree["nodes"][node_id]

        # Skip leaf nodes
        if len(node["children"]) == 0:
            continue

        # Process the node
        current_time += node["processing time"]
        if current_time > time_limit:
            times.append(time_limit)
            sizes.append(len(tree["nodes"]))
            break

        # If compressible, shrink the tree
        if node["compressed?"]:
            _replace(tree, node_id)

        times.append(current_time)
        sizes.append(len(tree["nodes"]))

    return times, sizes


class RandomSequence:
    def fit(self, trees):
        pass

    def predict(self, tree, stats):
        seq = list(tree["nodes"].keys())
        shuffle(seq)
        return seq


class ExpertSequence:
    @staticmethod
    def _score(tree, node_id):
        node = tree["nodes"][node_id]
        if node["compressed?"]:
            return node["subtree_size"] / node["processing time"]
        else:
            return 0

    def fit(self, trees):
        pass

    def predict(self, tree, stats):
        seq = list(tree["nodes"].keys())
        shuffle(seq)
        seq.sort(
            key=lambda node_id: self._score(tree, node_id),
            reverse=True,
        )
        return seq


class FastSequence:
    def fit(self, trees):
        pass

    def predict(self, tree, stats):
        seq = list(tree["nodes"].keys())
        shuffle(seq)
        seq.sort(
            key=lambda node_id: (tree["nodes"][node_id]["processing time"]),
        )
        return seq


class DepthFirstSequence:
    def fit(self, trees):
        pass

    def predict(self, tree, stats):
        seq = []
        stack = ["0"]
        while len(stack) > 0:
            node_id = stack.pop()
            seq.append(node_id)
            for child in tree["nodes"][node_id]["children"]:
                stack.append(child)
        return seq


class NodeIdSequence:
    def fit(self, trees):
        pass

    def predict(self, tree, stats):
        seq = [int(n) for n in tree["nodes"]]
        seq.sort(reverse=True)
        return [str(n) for n in seq]


class SubtreeSizeSequence:
    def fit(self, trees):
        pass

    def predict(self, tree, stats):
        seq = list(tree["nodes"].keys())
        shuffle(seq)
        seq.sort(key=lambda node_id: tree["nodes"][node_id]["subtree_size"])
        return seq


class GapSequence:
    @staticmethod
    def _score(tree, node_id):
        assert tree["sense"] == "min"
        global_bnd = tree["nodes"]["0"]["subtree_bound"]
        node = tree["nodes"][node_id]
        if node["obj"] >= global_bnd:
            return 0
        else:
            return abs(global_bnd - node["obj"]) / abs(global_bnd)

    def fit(self, trees):
        pass

    def predict(self, tree, stats):
        seq = list(tree["nodes"].keys())
        shuffle(seq)
        seq.sort(key=lambda node_id: GapSequence._score(tree, node_id))
        return seq


class MLSequence:
    def __init__(self, clf, proba=True):
        self.clf = clf
        self.proba = proba

    def _extract_features(self, trees):
        x = []
        for tree in trees:
            assert tree["sense"] == "min"
            for (node_id, node) in tree["nodes"].items():
                if node["dropped?"]:
                    continue
                x.append(
                    [
                        int(node_id),
                        node["depth"],
                        node["subtree_size"],
                        GapSequence._score(tree, node_id),
                        # node["compressed?"],
                        # node["subtree_size"] / node["processing time"],
                    ]
                )
        return np.array(x).astype(float)

    def _extract_labels(self, trees):
        y = []
        for tree in trees:
            for (node_id, node) in tree["nodes"].items():
                if node["dropped?"]:
                    continue
                y.append(node["compressed?"])
        return np.array(y).astype(int)

    def fit(self, trees):
        x = self._extract_features(trees)
        y = self._extract_labels(trees)
        # np.savetxt("x.csv", x)
        self.clf.fit(x, y)

        # fig = plt.figure(figsize=(12, 8))
        # plot_tree(self.clf, proportion=True)

    def predict(self, tree, stats):
        x = self._extract_features([tree])
        if self.proba:
            y_pred = self.clf.predict_proba(x)[:, 0]
        else:
            y_pred = self.clf.predict(x)
        scores = []
        y_true = []
        offset = 0

        for (node_id, node) in tree["nodes"].items():
            if node["dropped?"]:
                continue
            y_true.append(int(node["compressed?"]))
            scores.append(
                (
                    y_pred[offset],  # * node["subtree_size"],
                    node["subtree_size"],
                    random(),
                    node_id,
                )
            )
            offset += 1
        scores.sort()
        # stats["Accuracy"] = accuracy_score(y_true, y_pred)
        return [t[3] for t in scores]


def generate_chart(all_filenames, time_limit, methods):
    trees = []
    filenames = []
    for tree_filename in tqdm(all_filenames, desc="read"):
        with open(tree_filename) as tree_file:
            tree = json.load(tree_file)

            # Skip small trees
            if len(tree["nodes"]) < 100:
                continue

            # Drop all nodes that can be dropped
            compressed_count = 0
            for (node_id, node) in tree["nodes"].items():
                if node["dropped?"]:
                    _drop(tree, node_id)
                elif node["compressed?"]:
                    compressed_count += 1

            # Skip incompressible instances
            if compressed_count == 0:
                continue

            if SHOULD_MAKE_TIMES_UNIFORM:
                for (node_id, node) in tree["nodes"].items():
                    node["processing time"] = 1

            trees.append(tree)
            filenames.append(tree_filename)

    test_filenames, test_trees = filenames, trees

    # train_filenames, test_filenames, train_trees, test_trees = train_test_split(
    #     filenames,
    #     trees,
    #     test_size=0.50,
    #     random_state=42,
    # )

    # with open("split.json", "w") as file:
    #     json.dump(
    #         {
    #             "train": train_filenames,
    #             "test": test_filenames,
    #         },
    #         file,
    #     )

    # for method in tqdm(methods.values(), desc="fit"):
    #     method.fit(train_trees)

    data = []
    for (filename, tree) in zip(tqdm(test_filenames, desc="simulate"), test_trees):
        instance = basename(filename).replace(".tree.json", "")

        if len(tree["nodes"]) < 100:
            continue

        for n in tree["nodes"].values():
            n["global_bound"] = tree["nodes"]["0"]["subtree_bound"]

        fig = plt.figure(figsize=(8, 4))
        for (method_name, method) in methods.items():
            stats = {
                "Instance": instance,
                "Method": method_name,
            }
            seq = method.predict(tree, stats)
            times, sizes = simulate(tree, seq, time_limit)
            auc = 0
            for i in range(1, len(times)):
                auc += (times[i] - times[i - 1]) * sizes[i]
            auc = 100 * auc / (sizes[0] * time_limit)
            stats.update(
                {
                    "AUC (%)": auc,
                    "Time (s)": times[-1],
                    "Nodes visited": len(times) - 1,
                    "Original tree size": sizes[0],
                    "Compressed tree size": sizes[-1],
                }
            )
            data.append(stats)

            plt.step(times, sizes, where="post", linewidth=1.5)

        plt.title(f"{instance} ({time_limit})")
        plt.xlabel("Time (s)")
        plt.ylabel("Node count")
        plt.legend(methods.keys())
        plt.tight_layout()
        plt.savefig(f"chart-{time_limit}-{instance}.png", dpi=150)
        plt.close()
    return data


def generate_table(data, time_limit):
    summary = pd.DataFrame(data)
    summary = summary.groupby(["Instance", "Method"]).mean().unstack()
    # del summary[("Original tree size", "Expert")]
    # del summary[("Original tree size", "Random")]
    summary.loc["Mean", :] = summary.mean(axis=0)
    summary.round(2).to_csv(f"summary-{time_limit}.csv")
    df = summary.style.pipe(default_style)
    df.to_html(f"summary-{time_limit}.html")


def main():
    methods = {
        # "ML:Tree": MLSequence(
        #     DecisionTreeClassifier(
        #         max_depth=5,
        #         min_impurity_decrease=1e-3,
        #     )
        # ),
        # "ML:LogReg": MLSequence(
        #     make_pipeline(
        #         StandardScaler(),
        #         LogisticRegression(),
        #     ),
        # ),
        # "ML:GBoost": MLSequence(
        #     HistGradientBoostingClassifier(max_depth=5),
        # ),
        # "ML:Dummy": MLSequence(DummyClassifier()),
        # "ML": MLSequence(XGBClassifier()),
        "DFS": DepthFirstSequence(),
        "Random": RandomSequence(),
        "NodeId": NodeIdSequence(),
        "SubtreeSize": SubtreeSizeSequence(),
        "Gap": GapSequence(),
        # "Exp:Fast": FastSequence(),
        "Expert": ExpertSequence(),
    }
    for time_limit in [3600]:
        filenames = sorted(glob("../results/RB/heuristic/miplib2017/*.tree.json"))
        data = generate_chart(
            filenames,
            time_limit=time_limit,
            methods=methods,
        )
        generate_table(data, time_limit)


main()
