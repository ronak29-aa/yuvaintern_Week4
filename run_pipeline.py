from pathlib import Path
import os, runpy
root=Path(__file__).resolve().parents[1]; os.chdir(root)
runpy.run_path("src/train_evaluate.py")
print("Week 4 ML pipeline completed.")
