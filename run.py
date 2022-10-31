from miplearn.parallel import p_umap
from glob import glob
from random import shuffle
import re
import os
from distutils.dir_util import mkpath
from os.path import exists, dirname
from pathlib import Path
import sys

def process(args):
    instance, tree, method = args
    tree_filename = f"instances/trees/{tree}/miplib3/{instance}.tree.json"
    mps_filename = f"instances/models/miplib3/{instance}.mps.gz"
    out_filename = f"results/{tree}/{method}/miplib3/{instance}.out"
    done_filename = f"results/{tree}/{method}/miplib3/{instance}.done"
    
    disjcoef = 1
    nodetime = 1_200
    globaltime = 86_400
    
    if method == "drop":
        nodetime = 0
        globaltime = 0
        disjsuppsize = 1
    elif method == "supp:1":
        disjsuppsize = 1
    elif method == "supp:2":
        disjsuppsize = 2
    elif method == "supp:inf":
        disjsuppsize = -1

    if exists(done_filename):
        return

    mkpath(dirname(out_filename))
    os.system(f"python compress.py {mps_filename} --treename {tree_filename} --disjcoef {disjcoef} --disjsuppsize {disjsuppsize} --nodetime {nodetime} --globaltime {globaltime} > {out_filename} 2>&1")
    Path(done_filename).touch()


instances = [
    re.match("instances/models/miplib3/(.*).mps.gz", filename).groups()[0]
    for filename in glob("instances/models/miplib3/*.mps.gz")
]

combinations = [
    (
        instance,
        tree,
        method,
    )
    for instance in instances
    for tree in [sys.argv[1]]
    for method in ["drop", "supp:1", "supp:2", "supp:inf"]
]
shuffle(combinations)
p_umap(process, combinations, num_cpus=32)