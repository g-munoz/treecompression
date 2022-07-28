# Treecompression script

usage: python3 compress.py filename [--nodetime NODETIME] [--globaltime GLOBALTIME] [--restrictedsupp] [--upsearch] 

- filename is any file readable by Gurobi
- nodetime specifies the time limit at each node (optional, default is 300 secs)
- globaltime specifies the overall time limit (optional, default is 3600 secs)
- restrictedsupp is a flag indicating to restrict the disjunction support to the subtree rooted at the given node (optional, default uses a free support)
- upsearch is a flag indicating to do the search starting from the leaves (optional, default is downsearch)

If *filename = instance.mps*, the code will search for *instance.tree.json* for the tree.

For example: to run the code in bell3a.mps.gz with a global time limit of 120 second and doing an uptree search, you need to run:

*python3 compress.py bell3a.mps.gz --globaltime=120 --upsearch*

Or the same instance, with a downtree search, using the restricted support, and a node time limit of 120 seconds, would be:

*python3 compress.py bell3a.mps.gz --nodetime=120 --restrictedsupp*
