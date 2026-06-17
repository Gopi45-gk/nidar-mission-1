
import glob
import os

files = glob.glob(os.path.join('labels', '*.txt'))
classes = set()
for f in files:
    with open(f, 'r') as fp:
        for line in fp:
            if line.strip():
                parts = line.split()
                if parts:
                    classes.add(int(parts[0]))

print(f"Classes found: {sorted(list(classes))}")
