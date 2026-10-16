import json

with open("shclust-synthetic-experiment.ipynb", "r") as f:
    nb = json.load(f)

# Cell index 9 is LSH, 11 is inspection
for idx in [9, 11]:
    cell = nb["cells"][idx]
    print(f"Cell {idx} ({cell['cell_type']}):")
    print("".join(cell["source"]))
    print("=" * 40)
