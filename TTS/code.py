import nbformat as nbf
import re

nb = nbf.v4.new_notebook()
cells = []

with open("import_json.py", "r", encoding="utf-8") as f:
    text = f.read()

# Regex to capture md(...) and code(...)
pattern = re.findall(r'(md|code)\((\d+),\s*"""(.*?)"""\)', text, re.DOTALL)

# Sort by cell index
pattern = sorted(pattern, key=lambda x: int(x[1]))

for typ, idx, content in pattern:
    content = content.strip()
    if typ == "md":
        cells.append(nbf.v4.new_markdown_cell(content))
    else:
        cells.append(nbf.v4.new_code_cell(content))

nb["cells"] = cells

with open("converted.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook created: converted.ipynb")