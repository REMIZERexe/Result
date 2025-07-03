import re
with open("example.obj", "r", encoding="utf-8") as file:
    for line in file:
        if line.startswith("v "):
            vert = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', line)
            verts_float = [float(x) for x in vert]
            print(verts_float)