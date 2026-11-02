"""Table bodies as the paper inputs them: gray standard errors and room for the daggers."""
import os
import re

from common import DAGGER

GRAY = r"\textcolor{black!70}{"
# A parenthesized standard error after a number, as in "0.897 (.002)".
PAREN_SE = re.compile(r"(?<=\d)(\s+)(\((?:\d+(?:\.\d+)?|\.\d+)\))")
NUMBER = re.compile(r"^(?:\\textbf\{|\\textcolor\{black!70\}\{)?-?(?:\d|\.\d)")


def gray_paren_se(lines):
    """Set every parenthesized standard error of a table row in gray."""
    return [PAREN_SE.sub(lambda match: match[1] + GRAY + match[2] + "}", ln) if " & " in ln else ln
            for ln in lines]


def gray_se_columns(lines):
    """Set the entries of the columns headed SE or SE(WSC) in gray."""
    out, cols = [], []
    for ln in lines:
        cells = ln.split(" & ")
        heads = [i for i, c in enumerate(cells)
                 if re.sub(r"\s*\\\\\s*$", "", c).strip() in ("SE", "SE(WSC)")]
        if heads:
            cols = heads
        elif cols and len(cells) > max(cols):
            for i in cols:
                match = re.match(r"^(\d+(?:\.\d+)?)(.*)$", cells[i])
                if match:
                    cells[i] = GRAY + match[1] + "}" + match[2]
            ln = " & ".join(cells)
        out.append(ln)
    return out


def reserve_dagger_width(lines):
    """Pad the unmarked numbers of each column holding a dagger, so the marks keep the alignment."""
    # Tables with a fixed slot for the mark (common.pair_cell) need no padding.
    if any(r"\makebox[2.5pt][l]{" + DAGGER + "}" in ln for ln in lines):
        return lines
    rows, marked = {}, set()
    for i, ln in enumerate(lines):
        end = re.search(r"(\s*\\\\)\s*$", ln)
        if " & " not in ln or end is None:
            continue
        cells = ln[:end.start()].split(" & ")
        rows[i] = (cells, ln[end.start():])
        marked.update(j for j, c in enumerate(cells) if DAGGER in c)
    if not marked:
        return lines
    out = list(lines)
    for i, (cells, end) in rows.items():
        for j in marked:
            if j < len(cells) and NUMBER.match(cells[j]) and DAGGER not in cells[j]:
                cells[j] += r"\hphantom{" + DAGGER + "}"
        out[i] = " & ".join(cells) + end
    return out


def write_body(path, lines, se="none"):
    """Write one body; se='paren' grays parenthesized SEs and se='columns' the SE columns."""
    if se == "paren":
        lines = gray_paren_se(lines)
    elif se == "columns":
        lines = gray_se_columns(lines)
    lines = reserve_dagger_width(lines)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}")
