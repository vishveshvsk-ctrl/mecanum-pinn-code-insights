---
name: nb-read
description: Use this skill whenever the user references, mentions, uploads, or asks you to read, summarize, debug, or analyze a .ipynb (Jupyter notebook) file — even for quick questions about what a notebook does or contains. Jupyter notebooks embed outputs inline as JSON, including large base64-encoded images from matplotlib/plotly, making them extremely token-expensive to read with the standard Read tool. This skill runs a bundled script that strips image/plot outputs and truncates verbose print output (like solver logs or DataFrame dumps), leaving only code cells, markdown, and relevant text results. Always prefer this skill over the bare Read tool when any .ipynb file is involved. Also use when the user says "look at my notebook", "check this notebook", "what does this notebook do", or similar.
---

# nb-read: Token-Efficient Jupyter Notebook Reader

## What this skill does

Jupyter notebooks store ALL outputs — plots, print logs, DataFrame reprs — inline in the JSON. Reading a notebook with the bare `Read` tool includes every base64 PNG (expensive) and every line of solver output (often thousands of lines). This skill processes the notebook first, keeping only what matters for understanding the code.

**Kept:** cell source code, stdout/stderr text, `execute_result` text/plain, error tracebacks  
**Skipped:** `image/png`, `image/svg+xml`, `image/jpeg`, and other binary image MIME types  
**Truncated:** text output exceeding `--max-lines` lines per cell (default 50)

## How to read a notebook

**Step 1.** Find the bundled script using Glob:
```
Glob('**/nb_read.py')
```
Note the absolute path — call it `<SCRIPT>`.

**Step 2.** Run it on the notebook file:
```bash
python "<SCRIPT>" "path/to/notebook.ipynb"
```

**Step 3.** Read the output — it is clean markdown with cell headers, source blocks, and text results. Use that as your working view of the notebook.

### Useful options

```bash
# Tighter output truncation (good for notebooks with long solver logs)
python "<SCRIPT>" notebook.ipynb --max-lines 20

# Source code only, no outputs at all
python "<SCRIPT>" notebook.ipynb --no-outputs
```

## Handling skipped plots

When a cell only produces an image, the script inserts `*[N plot(s) skipped]*` in the output. You can still understand what was plotted from the source code that generated it. If the user needs you to see the actual figure, ask them to paste the image directly into the chat.

## Output format

```
# Notebook: filename.ipynb
Language: python | Cells: N (M code)
Plots skipped: K (paste into chat to view)

## Cell 1 [markdown]
...

## Cell 2 [code]
```python
source code here
```
**Output:**
```
text output here
```

## Cell 3 [code]
```python
plt.show()
```
*[1 plot(s) skipped]*
```
