import papermill as pm
from pathlib import Path

for nb in Path('./').glob('*.ipynb'):
    print("==============================", nb)

    pm.execute_notebook(input_path=nb, output_path=nb)
