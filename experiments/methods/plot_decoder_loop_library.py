"""Histograms of the decoder_loop failure library: support weights and mechanisms.

    python experiments/methods/plot_decoder_loop_library.py

Reads runs/decoder_loop/library.json, writes runs/decoder_loop/library_hist.png.
Three panels:
  (a) support weight w per entry, split by which decoder generated it;
  (b) mechanism reuse — how many library entries each DEM mechanism appears in;
  (c) where in the DEM those mechanisms sit (mechanism index).
"""
from __future__ import annotations

import collections
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from repo_paths import REPO_ROOT

OUT = REPO_ROOT / "runs" / "decoder_loop"

# categorical palette — the library now carries many generators (harvest, incumbent
# top-ups, per-model w3 harvests, device top-up specimens, tech2 subsets), so the
# palette CYCLES rather than assuming two.
PALETTE = ["#2a78d6", "#eb6834", "#2ca25f", "#c51b8a", "#8856a7",
           "#d9a441", "#4bb3c3", "#9c6b4f"]
C1, C2 = PALETTE[0], PALETTE[1]
SEQ = "#256abf"                        # sequential hue, mid step
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURF = "#e1e0d9", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def main():
    lib = json.loads((OUT / "library.json").read_text(encoding="utf-8"))
    entries = lib["entries"]
    gens = sorted({e["generator"] for e in entries})
    colors = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(gens)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURF)
    fig.suptitle(
        f"decoder_loop failure library — {len(entries)} entries, "
        f"w_min={lib['w_min']}, {len({m for e in entries for m in e['mechs']})} distinct mechanisms",
        color=INK, fontsize=12, fontweight="bold", y=1.0,
    )

    # (a) support-weight histogram, stacked by generator -----------------------
    ax = axes[0]
    ws = np.array([e["w"] for e in entries])
    bins = np.arange(ws.min(), ws.max() + 2) - 0.5
    bottom = np.zeros(len(bins) - 1)
    for g in gens:
        h, _ = np.histogram([e["w"] for e in entries if e["generator"] == g], bins=bins)
        ax.bar(bins[:-1] + 0.5, h, bottom=bottom, width=0.82, color=colors[g],
               label=g, zorder=3, linewidth=1.2, edgecolor=SURF)
        bottom += h
    ax.axvline(lib["w_min"], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.annotate(f"w_min = {lib['w_min']}", (lib["w_min"], bottom.max() * 0.92),
                xytext=(6, 0), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xlabel("support weight w (faults per failing config)")
    ax.set_ylabel("library entries")
    ax.set_title("(a) failure weights", loc="left", fontsize=11)
    ax.set_xticks(range(int(ws.min()), int(ws.max()) + 1))
    ax.legend(frameon=False, labelcolor=INK2, fontsize=7, ncol=2)
    style(ax)

    # (b) mechanism reuse ------------------------------------------------------
    ax = axes[1]
    per_mech = collections.Counter(m for e in entries for m in e["mechs"])
    reuse = collections.Counter(per_mech.values())
    ks = sorted(reuse)
    ax.bar(ks, [reuse[k] for k in ks], width=0.82, color=SEQ, zorder=3)
    for k in ks[:3]:
        ax.annotate(f"{reuse[k]}", (k, reuse[k]), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=INK2, fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(range(min(ks), max(ks) + 1, 2))
    ax.set_xlabel("entries a mechanism appears in")
    ax.set_ylabel("distinct mechanisms (log)")
    ax.set_title("(b) mechanism reuse across the library", loc="left", fontsize=11)
    style(ax)

    # (c) where the mechanisms sit in the DEM ---------------------------------
    ax = axes[2]
    occ = np.array([m for e in entries for m in e["mechs"]])
    ax.hist(occ, bins=60, color=SEQ, zorder=3)
    ax.set_xlabel("DEM mechanism index")
    ax.set_ylabel("occurrences in library")
    ax.set_title("(c) mechanism index distribution", loc="left", fontsize=11)
    style(ax)

    fig.tight_layout()
    p = OUT / "library_hist.png"
    fig.savefig(p, dpi=160, facecolor=SURF)
    print(f"wrote {p}")

    # table view (accessibility: never color/geometry alone)
    print("\nw    entries  " + "  ".join(f"{g[:14]:>14s}" for g in gens))
    for w in sorted(set(ws)):
        row = [sum(1 for e in entries if e["w"] == w and e["generator"] == g) for g in gens]
        print(f"{w:<4d} {sum(row):<8d} " + "  ".join(f"{v:>14d}" for v in row))
    print(f"\nmechanism reuse: " +
          ", ".join(f"{k}x:{reuse[k]}" for k in ks))
    print("top mechanisms: " +
          ", ".join(f"{m}({c})" for m, c in per_mech.most_common(8)))


if __name__ == "__main__":
    main()
