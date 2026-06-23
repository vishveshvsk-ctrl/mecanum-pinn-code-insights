import json, sys, os

skill_path = r"C:\Users\vishv\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\83539d12-2d45-4ca4-af13-7fc0b92f0767\abfbf7f2-f4ea-4e2a-9824-8b5126d7f23a"
nb_path = os.path.join(skill_path, r"nb-read-workspace\test_notebooks\mixed_outputs.ipynb")
skill_md = os.path.join(skill_path, r"skills\nb-read\SKILL.md")
nb_read_py = os.path.join(skill_path, r"skills\nb-read\nb_read.py")

print("=== SKILL.md ===")
try:
    with open(skill_md, encoding='utf-8') as f:
        print(f.read())
except Exception as e:
    print(f"Error reading SKILL.md: {e}")

print("\n=== nb_read.py exists? ===")
print(os.path.exists(nb_read_py))

print("\n=== Notebook JSON ===")
try:
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    # Print metadata and cells
    print(f"nbformat: {nb.get('nbformat')}")
    cells = nb.get('cells', [])
    print(f"Number of cells: {len(cells)}")
    for i, cell in enumerate(cells):
        print(f"\n--- Cell {i} ({cell['cell_type']}) ---")
        src = ''.join(cell.get('source', []))
        print(f"Source:\n{src}")
        outputs = cell.get('outputs', [])
        print(f"Number of outputs: {len(outputs)}")
        for j, out in enumerate(outputs):
            print(f"  Output {j}: type={out.get('output_type')}")
            if 'text' in out:
                print(f"    text: {''.join(out['text'])}")
            if 'data' in out:
                for k, v in out['data'].items():
                    if k == 'text/plain':
                        print(f"    data[{k}]: {''.join(v) if isinstance(v, list) else v}")
                    elif k == 'image/png':
                        print(f"    data[{k}]: <image data, len={len(''.join(v) if isinstance(v, list) else v)}>")
                    else:
                        val = ''.join(v) if isinstance(v, list) else v
                        print(f"    data[{k}]: {val[:200]}")
            if 'evalue' in out:
                print(f"    error: {out.get('ename')}: {out.get('evalue')}")
except Exception as e:
    print(f"Error reading notebook: {e}")
