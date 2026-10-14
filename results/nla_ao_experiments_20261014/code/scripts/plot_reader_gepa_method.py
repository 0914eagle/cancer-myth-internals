"""Render an editable method figure for the current reader-GEPA pilot."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT = Path(__file__).resolve().parents[1] / "results/nla_ao_parallel_20261012/figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(16, 10))
ax.set(xlim=(0, 16), ylim=(0, 10))
ax.axis("off")
fig.patch.set_facecolor("white")

INK = "#172438"
MUTED = "#526477"
BLUE = "#dfeafa"
TEAL = "#dff3ec"
PURPLE = "#eee7fa"


def label(x, y, text, size=11, color=INK, ha="center", weight="normal"):
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va="center",
            fontweight=weight, linespacing=1.45)


def box(x, y, w, h, text, color=BLUE, size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 facecolor=color, edgecolor="#b8c9d8", linewidth=1))
    label(x + w / 2, y + h / 2, text, size)


def arrow(start, end, color=MUTED):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                 mutation_scale=14, linewidth=1.6, color=color))


def path(points, color=MUTED):
    ax.plot([p[0] for p in points], [p[1] for p in points], color=color, lw=1.5)
    arrow(points[-2], points[-1], color)


def panel(y, h, heading):
    ax.add_patch(FancyBboxPatch((0.25, y), 15.5, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 facecolor="#f8fafc", edgecolor="#d5dee7", linewidth=1))
    label(0.55, y + h - 0.31, heading, 13, ha="left", weight="bold")


label(0.3, 9.68, "Can activation-derived feedback improve prompt optimization?",
      20, ha="left", weight="bold")
label(0.3, 9.27, "Current pilot design  |  Extra descriptions enter training reflection only",
      11, MUTED, ha="left")

panel(4.60, 4.33, "A. Collect paired answer and pre-answer activations on training questions")
box(0.60, 7.18, 2.20, 1.0, "Training question\n+ current prompt")
box(3.20, 7.18, 2.55, 1.0, "Source model\nGemma 3 27B")
box(6.15, 7.18, 2.10, 1.0, "Actual answer", BLUE)
box(8.65, 7.18, 2.75, 1.0, "GPT Well judge\nFPQ / NFP criteria", BLUE)
box(11.85, 7.10, 3.20, 1.15, "Shared text feedback\nQuestion, answer, score,\njudge rationale", BLUE, 10.5)
for a, b in [((2.80, 7.68), (3.20, 7.68)), ((5.75, 7.68), (6.15, 7.68)),
             ((8.25, 7.68), (8.65, 7.68)), ((11.40, 7.68), (11.85, 7.68))]:
    arrow(a, b)
box(8.78, 8.38, 2.50, 0.39, "Gold / reference (judge only)", "#f2f0e8", 9)
arrow((10.03, 8.38), (10.03, 8.18))

box(3.20, 5.51, 2.55, 1.20,
    "Same generation's\npre-answer states\nQuestion-end positions", TEAL, 10.5)
arrow((4.475, 7.18), (4.475, 6.71))
box(6.15, 5.51, 2.60, 1.20,
    "Fixed NLA or AO reader\nActivation input only\nNo question / gold text", TEAL, 10.5)
arrow((5.75, 6.11), (6.15, 6.11))
box(9.20, 5.51, 2.75, 1.20,
    "Extra train feedback\nNLA / AO description\nCorresponding arm only", PURPLE, 10.5)
arrow((8.75, 6.11), (9.20, 6.11))
label(1.70, 6.02, "States are captured\nbefore the first\nanswer token.", 10, MUTED)
label(7.9, 4.98,
      "NLA: decoder block 41, one question-end vector  |  AO: block 31, last up to 20 question-token vectors",
      10, MUTED)

panel(2.50, 1.83, "B. Compare four GEPA arms with the same seed prompt, data and search limits")
for x, text, color in [
    (0.65, "GEPA\nShared feedback", BLUE),
    (4.35, "+ Text diagnostic\nFrom shared text feedback", PURPLE),
    (8.05, "+ NLA description\nFrom activations", TEAL),
    (11.75, "+ AO description\nFrom activations", TEAL),
]:
    box(x, 3.01, 3.10, 0.68, text, color, 10)
label(8, 2.73,
      "GPT reflection  →  Candidate prompt  →  Fresh source answer + states  →  Train feedback  ↺",
      11, INK)
arrow((8, 4.60), (8, 4.33))

panel(0.48, 1.74, "C. Select on GPT dev scores, freeze, then evaluate independently")
box(0.65, 0.85, 3.20, 0.72, "GPT candidate-dev scores\nSelect best prompt", BLUE, 10.5)
box(4.30, 0.85, 2.40, 0.72, "Freeze prompt\nNo Haiku reselection", "#f2f0e8", 10.5)
box(7.15, 0.85, 3.60, 0.72, "Held-out answers → Haiku\nSource uses selected prompt only", BLUE, 10.5)
box(11.20, 0.85, 3.55, 0.72, "FPQ correction + NFP preservation\nReport both separately", TEAL, 10.5)
for a, b in [((3.85, 1.21), (4.30, 1.21)), ((6.70, 1.21), (7.15, 1.21)),
             ((10.75, 1.21), (11.20, 1.21))]:
    arrow(a, b)
arrow((8, 2.50), (8, 2.22))
label(0.30, 0.14,
      "Pilot: 100 train questions across two datasets; 8 candidate-dev + 8 separate held-out questions per dataset. Held-out pilot ≠ official test.",
      9, MUTED, ha="left")

fig.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.01)
for extension in ("png", "svg", "pdf"):
    fig.savefig(OUTPUT / f"reader_gepa_method_v1.{extension}", dpi=180,
                facecolor="white", bbox_inches="tight", pad_inches=0.15)
plt.close(fig)
print(OUTPUT / "reader_gepa_method_v1.png")
