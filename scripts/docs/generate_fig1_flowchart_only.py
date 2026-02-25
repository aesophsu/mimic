"""Generate Figure 1 flowchart only (Panel A) to docs/main/figs."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import matplotlib.pyplot as plt
import matplotlib as mpl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_04B = PROJECT_ROOT / "scripts" / "audit_eval" / "04b_fig1_study_overview.py"
OUT_DIR = PROJECT_ROOT / "docs" / "main" / "figs"


def _load_flowchart_func():
    spec = importlib.util.spec_from_file_location("fig1_overview", str(SCRIPT_04B))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_04B}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "draw_flowchart_panel"):
        raise RuntimeError("draw_flowchart_panel not found in 04b_fig1_study_overview.py")
    return mod.draw_flowchart_panel


def main() -> None:
    draw_flowchart_panel = _load_flowchart_func()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Typography: force Arial and larger readable fonts for standalone flowchart
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    # Single-panel flowchart
    # Lower aspect ratio (less wide) for manuscript-friendly layout
    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=300, facecolor="white")
    draw_flowchart_panel(
        ax,
        box_w=3.8,
        title_fontsize=10.8,
        box_fontsize=9.2,
    )
    # Increase line spacing for multi-line box text to improve readability.
    for txt in ax.texts:
        if "\n" in txt.get_text():
            txt.set_linespacing(1.35)
    fig.tight_layout()

    out_base = OUT_DIR / "Fig1_flowchart"
    fig.savefig(f"{out_base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_base}.png")
    print(f"Saved: {out_base}.pdf")


if __name__ == "__main__":
    main()
