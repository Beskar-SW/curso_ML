"""Run the complete educational course in the documented order."""
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STEPS=[
    ["generate_data.py"],
    ["visualization/theory.py"],
    ["-m","pytest","-q"],
    ["supervisado/train_all.py"],
    ["no_supervisado/train_all.py"],
    ["course_report.py"],
]
def main():
    for args in STEPS:
        print("\n=== Ejecutando:",*args,"===",flush=True)
        subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    print("\nCurso generado: results/EXPERIMENT_REPORT.md")
if __name__=="__main__":main()
