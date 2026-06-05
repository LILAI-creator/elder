import os

base = r"D:\my_datasets\Le2i\Le2i\Le2i"
scenes = [
    ("Coffee_room_01", "Coffee_room_01", "Annotation_files"),
    ("Coffee_room_02", "Coffee_room_02", "Annotations_files"),
    ("Home_01", "Home_01", "Annotation_files"),
    ("Home_02", "Home_02", "Annotations_files"),
]

total_fall = 0
total_normal = 0
for s, sub, anno in scenes:
    anno_dir = os.path.join(base, s, sub, anno)
    files = sorted([f for f in os.listdir(anno_dir) if f.endswith(".txt")])
    fc = 0
    nc = 0
    for f in files:
        lines = open(os.path.join(anno_dir, f)).readlines()
        try:
            fs = int(lines[0].strip())
            fe = int(lines[1].strip())
        except ValueError:
            print(f"  NON-STANDARD: {s}/{f} -> line0='{lines[0].strip()[:50]}' line1='{lines[1].strip()[:50]}'")
            nc += 1
            continue
        if fs > 0 and fe > 0:
            fc += 1
        else:
            nc += 1
    total_fall += fc
    total_normal += nc
    print(s, len(files), "fall:", fc, "normal:", nc)
print("Total:", total_fall + total_normal, "fall:", total_fall, "normal:", total_normal)