"""Validate nbformat and parse every code cell without executing notebook code."""
import ast
from pathlib import Path

import nbformat

root = Path(__file__).resolve().parents[1]
paths = sorted((root / "notebooks").glob("*.ipynb"))
if len(paths) != 19:
    raise SystemExit("Expected nineteen project notebooks")
for path in paths:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            if cell.outputs or cell.execution_count is not None:
                raise ValueError(f"Published notebook includes execution output: {path.name}:{index}")
            ast.parse(cell.source, filename=f"{path.name}:{index}")
print(f"Validated {len(paths)} notebooks; code syntax valid; no stored outputs")
