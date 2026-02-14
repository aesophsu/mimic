#!/usr/bin/env python3
"""Collect all figure/table legends into a single summary file."""
from pathlib import Path
from collections import OrderedDict

docs = Path(__file__).resolve().parent.parent / "docs"
tables_main = docs / "tables" / "main"
tables_supp = docs / "tables" / "supplementary"
figures_main = docs / "figures" / "main"
figures_supp = docs / "figures" / "supplementary"

def base_from_md(md_path: Path) -> str:
    """Extract base identifier: Table1_baseline, Fig2_ROC_external_pof, etc."""
    name = md_path.stem  # e.g., Table1_baseline.csv or Fig2_ROC_external_pof.pdf
    for ext in [".csv", ".pdf", ".png", ".svg"]:
        if name.endswith(ext):
            return name[: -len(ext)]
    return name

def collect_legends(root: Path) -> OrderedDict:
    seen = {}
    for md in sorted(root.rglob("*.md")):
        if md.parent == root or "SF" in str(md):
            base = base_from_md(md)
            if base not in seen:
                seen[base] = (md.relative_to(docs), md.read_text(encoding="utf-8").strip())
    return OrderedDict(sorted(seen.items(), key=lambda x: (str(x[1][0]), x[0])))

out = []
out.append("# 稿件图表说明汇总（Figure & Table Legends）")
out.append("")
out.append("> 本文档汇总所有图片和表格的说明（legend），便于复制、修改后用于论文投稿。")
out.append("> 原始说明文件位于各图表同目录下的同名 `.md` 文件中。")
out.append("")
out.append("---")
out.append("")

# 主文表格
out.append("## 一、主文表格（Main Tables）")
out.append("")
for md in sorted(tables_main.glob("*.csv.md")):
    base = base_from_md(md)
    content = md.read_text(encoding="utf-8").strip()
    out.append(f"### {base}")
    out.append("")
    out.append("```")
    out.append(content)
    out.append("```")
    out.append("")
    out.append("---")
    out.append("")

# 主文图形（去重 base）
seen_fig = {}
for md in sorted(figures_main.glob("*.md")):
    base = base_from_md(md)
    if base not in seen_fig:
        seen_fig[base] = md.read_text(encoding="utf-8").strip()

out.append("## 二、主文图形（Main Figures）")
out.append("")
for base in sorted(seen_fig.keys()):
    out.append(f"### {base}")
    out.append("")
    out.append("```")
    out.append(seen_fig[base])
    out.append("```")
    out.append("")
    out.append("---")
    out.append("")

# 补充表格
out.append("## 三、补充表格（Supplementary Tables）")
out.append("")
for md in sorted(tables_supp.glob("*.csv.md")):
    base = base_from_md(md)
    content = md.read_text(encoding="utf-8").strip()
    out.append(f"### {base}")
    out.append("")
    out.append("```")
    out.append(content)
    out.append("```")
    out.append("")
    out.append("---")
    out.append("")

# 补充图形（按 SF1-SF5 分组，去重 base）
out.append("## 四、补充图形（Supplementary Figures）")
out.append("")

def collect_supp_figures():
    seen = {}
    for md in sorted(figures_supp.rglob("*.md")):
        base = base_from_md(md)
        # Use full relative path as part of key to preserve SF1/SF2/etc order
        rel = md.relative_to(figures_supp)
        key = (str(rel.parts[0]) if rel.parts else "", base)
        if key not in seen:
            seen[key] = md.read_text(encoding="utf-8").strip()
    return seen

supp_fig = collect_supp_figures()
for (folder, base), content in sorted(supp_fig.items(), key=lambda x: (x[0][0], x[0][1])):
    out.append(f"### {folder} / {base}")
    out.append("")
    out.append("```")
    out.append(content)
    out.append("```")
    out.append("")
    out.append("---")
    out.append("")

output_path = docs / "FIGURES_TABLES_LEGENDS_SUMMARY.md"
output_path.write_text("\n".join(out), encoding="utf-8")
print(f"Written: {output_path}")
