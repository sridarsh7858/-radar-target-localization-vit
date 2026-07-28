"""RD-ViT architecture diagram (Fig 3.3): the localizer and detector drawn as two
separate, independent models side by side (no shared weights)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

A = os.environ.get("REPORT_ASSETS", "report_assets")
os.makedirs(A, exist_ok=True)


def box(ax, cx, y, w, h, title, sub, fc, title_fs=10, sub_fs=7.4):
    ax.add_patch(FancyBboxPatch((cx - w / 2, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec="#33415c", lw=1.3))
    if sub:
        ax.text(cx, y + h * 0.70, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold")
        ax.text(cx, y + h * 0.32, sub, ha="center", va="center", fontsize=sub_fs)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold")


def arrow(ax, x, y1, y2):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>",
                                 mutation_scale=13, color="#33415c", lw=1.4))


fig, ax = plt.subplots(figsize=(12.6, 9.6))
ax.set_xlim(0, 14); ax.set_ylim(0, 12.6); ax.axis("off")

LX, RX, W = 3.6, 10.4, 6.0

ax.text(LX, 12.25, "LOCALIZER  (independent model)", ha="center",
        fontsize=12, fontweight="bold", color="#1f3b63")
ax.text(RX, 12.25, "DETECTOR  (independent model)", ha="center",
        fontsize=12, fontweight="bold", color="#1f3b63")

for cx, inp in ((LX, "Input range Doppler map   (512 x 14 x 4)"),
                (RX, "Input range Doppler map   (512 x 14 x 1, HH)")):
    box(ax, cx, 11.0, W, 0.8, inp, "", "#fce5cd", title_fs=9.5)
    arrow(ax, cx, 11.0, 10.6)
    box(ax, cx, 9.25, W, 1.35, "CNN Patch Embedder",
        "Conv 3x3, BN, ReLU, MaxPool (Doppler only)  x4\n"
        "Doppler axis compressed 512 -> 8   ->   8 x 14 x 128", "#cfe2f3")
    arrow(ax, cx, 9.25, 8.85)

box(ax, LX, 7.55, W, 1.3, "Tokenize + Embed",
    "Permute & Reshape -> 14 tokens x 1024\n"
    "Dense(128) + position embedding -> 14 x 128", "#cfe2f3")
box(ax, RX, 7.55, W, 1.3, "Tokenize + Embed + [DET] token",
    "Permute & Reshape -> Dense(128) -> 14 x 128\n"
    "learnable [DET] token prepended -> 15 x 128", "#cfe2f3")
arrow(ax, LX, 7.55, 7.15); arrow(ax, RX, 7.55, 7.15)

box(ax, LX, 5.85, W, 1.3, "Transformer Encoder  x4",
    "LayerNorm -> MHA (4 heads) -> residual\n"
    "LayerNorm -> MLP(128,256,128) -> residual", "#cfe2f3")
box(ax, RX, 5.85, W, 1.3, "Transformer Encoder  x4",
    "same block design, own weights\n"
    "attention over 15 tokens (14 gates + [DET])", "#cfe2f3")
arrow(ax, LX, 5.85, 5.45); arrow(ax, RX, 5.85, 5.45)

box(ax, LX, 4.6, W, 0.85, "Encoded range tokens   (14 x 128)", "", "#d9ead3", title_fs=9.5)
box(ax, RX, 4.6, W, 0.85, "Encoded [DET] token   (128)", "", "#d9ead3", title_fs=9.5)
arrow(ax, LX, 4.6, 4.2); arrow(ax, RX, 4.6, 4.2)

box(ax, LX, 2.9, W, 1.3, "Localization Head",
    "Dropout -> Dense(64, GELU) -> Dense(1)\n"
    "-> Reshape(14) -> Softmax over 14 gates", "#dbe5f1")
box(ax, RX, 2.9, W, 1.3, "Detection Head",
    "Dense(64, GELU) -> Dropout\n-> Dense(1) -> Sigmoid", "#dbe5f1")
arrow(ax, LX, 2.9, 2.5); arrow(ax, RX, 2.9, 2.5)

box(ax, LX, 1.65, W, 0.85, "Range gate localization   (argmax of 14)", "", "#f2f2f2", title_fs=9.5)
box(ax, RX, 1.65, W, 0.85, "Presence probability   (target / clutter)", "", "#f2f2f2", title_fs=9.5)

ax.text(7, 0.9, "The two models share the same backbone design but are trained separately "
                "with their own weights, so the objectives never compete.",
        ha="center", fontsize=8.5, style="italic", color="#555")

plt.tight_layout()
plt.savefig(f"{A}/fig_arch_v2.png", dpi=150, bbox_inches="tight")
print("wrote fig_arch_v2.png")
