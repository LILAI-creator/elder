from pathlib import Path

root = Path(
    r"D:\my_datasets\UR-FALL"
)

for p in root.iterdir():
    print(p)