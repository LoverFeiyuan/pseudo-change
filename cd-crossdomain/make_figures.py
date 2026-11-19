"""make_figures.py —— 生成论文插图（v2）

修正点：
  fig1  样本挑选更典型（LEVIR 取建筑密集、S2L 取亮度较高者）
  fig2  横轴改为「相对源域的亮度偏移百分比」，而不是绝对亮度
        ——用绝对值时 WHU 比 LEVIR 更亮但 F1 更低，看不出单调关系
  fig3  图例移到图外，避免遮挡柱子顶部的数值
  fig5  改为相对基线的差值图，差异更直观
"""
import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path("figures")
FIG.mkdir(exist_ok=True)
DATA = Path(r"data")

SRC_BRIGHT = 102.7

MAIN = [
    ("LEVIR", 102.7, 0.8290, 0.0101, "source"),
    ("WHU", 133.1, 0.7020, 0.0019, "target"),
    ("S2Looking", 38.5, 0.0567, None, "target"),
]
S2L_INSTNORM = 0.283

ERR = {
    "LEVIR\n(source, 0%)": {"Pseudo-change": 4.0, "Border (≤3 px)": 70.0, "Far (>3 px)": 25.9},
    "WHU\n(target, +30%)": {"Pseudo-change": 40.7, "Border (≤3 px)": 11.4, "Far (>3 px)": 47.8},
    "S2Looking\n(target, −63%)": {"Pseudo-change": 45.0, "Border (≤3 px)": 6.5, "Far (>3 px)": 48.5},
}

MECH = [
    ("Baseline", 0.7020, 0.0019),
    ("Consistency\n(global)", 0.6317, 0.0516),
    ("Consistency\n(local)", 0.6522, 0.0224),
    ("Zero-change\nsupervision", 0.6601, 0.0276),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})


def load_patch(root: Path, split: str, key: str):
    a = np.load(root / split / "A" / f"{key}.npy")[:, :, ::-1]
    b = np.load(root / split / "B" / f"{key}.npy")[:, :, ::-1]
    l = np.load(root / split / "L" / f"{key}.npy")
    return a, b, l


def pick_by_csv(csv_path: Path, split: str, lo: float, hi: float):
    with open(csv_path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    cand = [r for r in rows if lo <= float(r["change_ratio"]) <= hi]
    if not cand:
        cand = sorted(rows, key=lambda r: -float(r["change_ratio"]))[:20]
    return cand[len(cand) // 2]


def fig1() -> None:
    r_levir = DATA / "patches"
    r_whu = DATA / "patches_whu"

    # LEVIR：取变化比例偏大的块，多为建筑密集区
    row_l = pick_by_csv(r_levir / "patch_stats.csv", "test", 0.05, 0.12)
    a_l, b_l, m_l = load_patch(r_levir, "test", row_l["key"])

    row_w = pick_by_csv(r_whu / "patch_stats_whu.csv", "test", 0.03, 0.08)
    a_w, b_w, m_w = load_patch(r_whu, "test", row_w["key"])

    # S2Looking：挑亮度较高且有明显变化的样本，避免整幅过暗看不清
    s2 = DATA / "S2Looking" / "S2Looking" / "test"
    best, best_score = None, -1
    for n in sorted(p.name for p in (s2 / "Image1").glob("*.png"))[:300]:
        lab = cv2.imread(str(s2 / "label" / n), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(s2 / "Image1" / n), cv2.IMREAD_COLOR)
        if lab is None or img is None:
            continue
        ratio = float((lab > 127).mean())
        bright = float(img.mean())
        if 0.03 <= ratio <= 0.08:
            score = bright
            if score > best_score:
                best, best_score = n, score
    if best is None:
        best = sorted(p.name for p in (s2 / "Image1").glob("*.png"))[0]
    a_s = cv2.cvtColor(cv2.imread(str(s2 / "Image1" / best)), cv2.COLOR_BGR2RGB)
    b_s = cv2.cvtColor(cv2.imread(str(s2 / "Image2" / best)), cv2.COLOR_BGR2RGB)
    m_s = (cv2.imread(str(s2 / "label" / best), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)

    rows = [
        ("LEVIR-CD\n(source, 102.7)", a_l, b_l, m_l),
        ("WHU-CD\n(target, 133.1)", a_w, b_w, m_w),
        ("S2Looking\n(target, 38.5)", a_s, b_s, m_s),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(9.6, 8.0))
    titles = ["Pre-event", "Post-event", "Ground truth", "GT overlay"]
    for i, (name, a, b, m) in enumerate(rows):
        ov = b.copy()
        ov[m > 0] = [255, 0, 0]
        for j, img in enumerate([a, b, (m * 255).astype(np.uint8), ov]):
            ax = axes[i, j]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_domains.png", bbox_inches="tight")
    plt.close(fig)
    print(f"fig1 -> {row_l['key']} / {row_w['key']} / {best}")


def fig2() -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))

    for name, bright, f1, std, role in MAIN:
        off = abs(bright - SRC_BRIGHT) / SRC_BRIGHT * 100
        color = "#1f77b4" if role == "source" else "#d62728"
        if std is not None:
            ax.errorbar(off, f1, yerr=std, fmt="o", ms=8, color=color,
                        capsize=4, label=name)
        else:
            ax.plot(off, f1, "o", ms=8, color=color, label=name)
        ax.annotate(f"{name}\n{f1:.3f}", (off, f1),
                    textcoords="offset points", xytext=(10, 4), fontsize=8)

    off_s2 = abs(38.5 - SRC_BRIGHT) / SRC_BRIGHT * 100
    ax.plot(off_s2, S2L_INSTNORM, "s", ms=8, mfc="none", mew=1.6,
            color="#2ca02c", label="S2Looking + instance norm")
    ax.annotate(f"+inst. norm\n{S2L_INSTNORM:.3f}", (off_s2, S2L_INSTNORM),
                textcoords="offset points", xytext=(12, -18), fontsize=8, color="#2ca02c")

    ax.set_xlabel("Relative brightness deviation from source (%)")
    ax.set_ylabel("Test F1")
    ax.set_xlim(-5, 75)
    ax.set_ylim(0, 0.95)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.set_title("Cross-domain drop vs. imaging statistic deviation", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_brightness.png", bbox_inches="tight")
    plt.close(fig)
    print("fig2 -> ok")


def fig3() -> None:
    cats = ["Pseudo-change", "Border (≤3 px)", "Far (>3 px)"]
    colors = ["#d62728", "#ff7f0e", "#1f77b4"]
    names = list(ERR.keys())

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bottoms = np.zeros(len(names))
    for ci, (cat, color) in enumerate(zip(cats, colors)):
        vals = np.array([ERR[n][cat] for n in names])
        ax.bar(names, vals, bottom=bottoms, label=cat, color=color, width=0.52)
        for i, v in enumerate(vals):
            ax.text(i, bottoms[i] + v / 2, f"{v:.1f}%", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
        bottoms += vals
    ax.set_ylabel("Share of false positives (%)")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              frameon=False)
    ax.set_title("Error structure: border-dominated vs pseudo-change-dominated", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_errorstruct.png", bbox_inches="tight")
    plt.close(fig)
    print("fig3 -> ok")


def fig5() -> None:
    base = MECH[0][1]
    labels = ["Consistency\n(global)", "Consistency\n(local)", "Zero-change\nsupervision"]
    deltas = [m[1] - base for m in MECH[1:]]
    stds = [m[2] for m in MECH[1:]]

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    xs = np.arange(len(labels))
    ax.bar(xs, deltas, yerr=stds, capsize=5, width=0.55, color="#d62728")
    ax.axhline(0, color="black", lw=0.9)

    upper_bound = 0.0707
    ax.axhline(upper_bound, ls="--", lw=1.2, color="#2ca02c")
    ax.text(len(labels) - 0.45, upper_bound + 0.004,
            f"upper bound if pseudo-change fully removed (+{upper_bound:.2f})",
            fontsize=7.5, color="#2ca02c", ha="right")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("ΔF1 on WHU (vs. baseline, 3 seeds)")
    ax.set_ylim(-0.14, 0.09)
    for x, d in zip(xs, deltas):
        # 数值放在柱内，避免和长误差棒重叠
        ax.text(x, d / 2, f"{d:+.3f}", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
    ax.set_title("Suppression cost exceeds its theoretical benefit", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_mechanisms.png", bbox_inches="tight")
    plt.close(fig)
    print("fig5 -> ok")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig5()
