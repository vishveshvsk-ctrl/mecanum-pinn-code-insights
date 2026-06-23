#!/usr/bin/env python3
"""
nb_read.py - Token-efficient Jupyter notebook reader.

Strips image/plot outputs and truncates verbose text output,
leaving only code cells, markdown, and text results.

Usage:
    python nb_read.py notebook.ipynb
    python nb_read.py notebook.ipynb --max-lines 30
    python nb_read.py notebook.ipynb --no-outputs
"""

import argparse
import json
import re
import sys
from pathlib import Path

IMAGE_MIME_TYPES = {
    "image/png", "image/jpeg", "image/jpg",
    "image/svg+xml", "image/gif", "image/webp",
    "application/pdf",
}

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")


def get_kernel_language(nb):
    try:
        lang = nb.get("metadata", {}).get("kernelspec", {}).get("language", "")
        if lang:
            return lang
        lang = nb.get("metadata", {}).get("language_info", {}).get("name", "")
        return lang or "python"
    except Exception:
        return "python"


def extract_text_output(output):
    """Return text from a single output dict, or None if image-only."""
    output_type = output.get("output_type", "")

    if output_type == "stream":
        text = "".join(output.get("text", []))
        return text.rstrip() or None

    if output_type in ("execute_result", "display_data"):
        data = output.get("data", {})
        # Skip entirely when an image is present — text/plain is just
        # matplotlib's "<Figure size ...>" repr, not useful content.
        if any(mime in data for mime in IMAGE_MIME_TYPES):
            return None
        if "text/plain" in data:
            return "".join(data["text/plain"]).rstrip()
        if "text/html" in data:
            html = "".join(data["text/html"])
            return f"[HTML output, {len(html)} chars]"

    if output_type == "error":
        ename = output.get("ename", "Error")
        evalue = output.get("evalue", "")
        tb_raw = output.get("traceback", [])
        tb_clean = [ANSI_ESCAPE.sub("", line) for line in tb_raw]
        if len(tb_clean) > 8:
            tb_clean = ["... [traceback truncated]"] + tb_clean[-8:]
        return f"{ename}: {evalue}\n" + "\n".join(tb_clean)

    return None


def truncate(text, max_lines):
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    omitted = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n... [{omitted} more lines omitted]"


def format_notebook(nb_path, max_lines=50, show_outputs=True):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    lang = get_kernel_language(nb)
    cells = nb.get("cells", [])
    total_cells = len(cells)
    code_cells = sum(1 for c in cells if c.get("cell_type") == "code")

    out = []
    out.append(f"# Notebook: {Path(nb_path).name}")
    out.append(f"Language: {lang} | Cells: {total_cells} ({code_cells} code)")
    out.append("Note: Image/plot outputs stripped. Paste plots manually if needed.")
    out.append("")

    images_skipped = 0

    for i, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "code")
        source = "".join(cell.get("source", []))

        if not source.strip() and not cell.get("outputs"):
            continue

        out.append(f"## Cell {i + 1} [{cell_type}]")

        if source.strip():
            fence_lang = lang if cell_type == "code" else ""
            out.append(f"```{fence_lang}")
            out.append(source)
            out.append("```")

        if show_outputs and cell_type == "code":
            outputs = cell.get("outputs", [])
            text_parts = []
            cell_images = 0

            for output in outputs:
                text = extract_text_output(output)
                if text is not None:
                    text_parts.append(text)
                else:
                    cell_images += 1
                    images_skipped += 1

            if text_parts:
                combined = "\n".join(p for p in text_parts if p)
                out.append("**Output:**")
                out.append("```")
                out.append(truncate(combined, max_lines))
                out.append("```")

            if cell_images:
                out.append(f"*[{cell_images} plot(s) skipped]*")

        out.append("")

    if images_skipped:
        out.insert(3, f"Plots skipped: {images_skipped} (paste into chat to view)")

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Token-efficient Jupyter notebook reader")
    parser.add_argument("notebook", help="Path to .ipynb file")
    parser.add_argument("--max-lines", type=int, default=50,
                        help="Max output lines per cell (default: 50)")
    parser.add_argument("--no-outputs", action="store_true",
                        help="Skip all cell outputs; show source only")
    args = parser.parse_args()

    nb_path = Path(args.notebook)
    if not nb_path.exists():
        print(f"Error: {nb_path} not found", file=sys.stderr)
        sys.exit(1)

    print(format_notebook(nb_path, max_lines=args.max_lines, show_outputs=not args.no_outputs))


if __name__ == "__main__":
    main()
