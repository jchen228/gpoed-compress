#!/usr/bin/env python3
"""
hybrid_gp_diagram.py
4-panel figure illustrating Hybrid GP dyadic compression for levels 0 & 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

# ── colour palette (matches slides) ──────────────────────────────────────────
CNAVY  = '#0D2137'
CTEAL  = '#1A6B8A'
CMINT  = '#02C39A'
CICE   = '#E4F2F8'
CMID   = '#7BB8CC'
CAMBER = '#F0A500'
CDGRAY = '#2E4057'
CMUTED = '#5A7385'

# ── synthetic smooth 2-D field on the 0–8 domain ─────────────────────────────
res = 400
xs  = np.linspace(0, 8, res)
ys  = np.linspace(0, 8, res)
X, Y = np.meshgrid(xs, ys)
field = (1.4 * np.sin(X * np.pi / 4.5) * np.cos(Y * np.pi / 5.5)
       + 0.9 * np.cos(X * np.pi / 6.0 + np.pi / 3) * np.sin(Y * np.pi / 4.0)
       + 0.5 * np.cos((X + Y) * np.pi / 7.0))

# ── dyadic grid coordinates ───────────────────────────────────────────────────
coords0 = [0, 4, 8]        # stride-4 (level 0)
coords1 = [0,2,4,6,8]     # stride-2 (level 1)

pts0     = [(x, y) for x in coords0 for y in coords0]   # 9
pts1_all = [(x, y) for x in coords1 for y in coords1]   # 25
pts0_set = set(pts0)
pts1_new = [p for p in pts1_all if p not in pts0_set]   # 16 new midpoints

# ── illustrative anchor selections (spread out, as RPCholesky would) ──────────
anchors0 = [(0, 0), (8, 4), (4, 8)]                          # 3 / 9
gp_pred0 = [p for p in pts0 if p not in set(anchors0)]       # 6

anchors1 = [(2, 0), (0, 6), (6, 2), (8, 6)]                  # 4 / 16
gp_pred1 = [p for p in pts1_new if p not in set(anchors1)]   # 12

def xs_(pts): return [p[0] for p in pts]
def ys_(pts): return [p[1] for p in pts]

# ── axes helpers ──────────────────────────────────────────────────────────────
def draw_field(ax):
    ax.imshow(field, extent=[0, 8, 0, 8], origin='lower',
              cmap='Blues', alpha=0.20, aspect='equal')
    # faint grid lines at stride-4
    for c in coords0:
        ax.axvline(c, color=CMID, lw=0.5, alpha=0.35, zorder=0)
        ax.axhline(c, color=CMID, lw=0.5, alpha=0.35, zorder=0)

def style_ax(ax, title, subtitle):
    ax.set_xlim(-0.7, 8.7)
    ax.set_ylim(-0.7, 8.7)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=10.5, fontweight='bold', color=CNAVY, pad=7)
    ax.text(0.5, -0.03, subtitle, transform=ax.transAxes,
            fontsize=7.8, color=CMUTED, ha='center', va='top', style='italic')

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15.5, 4.8), facecolor='white')
gs  = GridSpec(1, 4, figure=fig, wspace=0.22,
               left=0.03, right=0.97, top=0.84, bottom=0.18)
axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

# ─────────────────────────────────────────────────────────────────────────────
# (a) Level 0 — all 9 candidate points
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0]
draw_field(ax)
ax.scatter(xs_(pts0), ys_(pts0), s=110, color=CTEAL,
           edgecolors=CNAVY, linewidths=0.8, zorder=5)
style_ax(ax,
    'Level 0 — All Candidates',
    '9 coarse-grid points (stride = 4)')
ax.text(0.03, 0.97, '(a)', transform=ax.transAxes,
        fontsize=9, color=CNAVY, va='top', fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# (b) Level 0 — after GP: anchors vs GP-predicted
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1]
draw_field(ax)
ax.scatter(xs_(gp_pred0), ys_(gp_pred0), s=110, color=CAMBER,
           edgecolors=CNAVY, linewidths=0.8, zorder=5)
ax.scatter(xs_(anchors0), ys_(anchors0), s=220, color=CMINT,
           edgecolors=CNAVY, linewidths=0.9, marker='*', zorder=6)
style_ax(ax,
    'Level 0 — GP Applied',
    '3 anchors (raw float32)  ·  6 GP-predicted (quantised residual)')
ax.text(0.03, 0.97, '(b)', transform=ax.transAxes,
        fontsize=9, color=CNAVY, va='top', fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# (c) Level 1 — all 16 new midpoint candidates
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[2]
draw_field(ax)
# show level-0 resolved points faded
ax.scatter(xs_(pts0), ys_(pts0), s=45, color=CMID,
           edgecolors=CDGRAY, linewidths=0.5, alpha=0.50, zorder=3)
# new level-1 candidates
ax.scatter(xs_(pts1_new), ys_(pts1_new), s=80, color=CTEAL,
           edgecolors=CNAVY, linewidths=0.8, marker='D', zorder=5)
style_ax(ax,
    'Level 1 — New Midpoints',
    '16 new candidates (stride = 2)  ·  conditioned on all level-0 values')
ax.text(0.03, 0.97, '(c)', transform=ax.transAxes,
        fontsize=9, color=CNAVY, va='top', fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# (d) Level 1 — after GP: anchors vs GP-predicted
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[3]
draw_field(ax)
# level-0 points faded
ax.scatter(xs_(pts0), ys_(pts0), s=45, color=CMID,
           edgecolors=CDGRAY, linewidths=0.5, alpha=0.50, zorder=3)
# GP-predicted level-1
ax.scatter(xs_(gp_pred1), ys_(gp_pred1), s=80, color=CAMBER,
           edgecolors=CNAVY, linewidths=0.8, marker='D', zorder=5)
# anchors level-1
ax.scatter(xs_(anchors1), ys_(anchors1), s=200, color=CMINT,
           edgecolors=CNAVY, linewidths=0.9, marker='*', zorder=6)
style_ax(ax,
    'Level 1 — GP Applied',
    '4 anchors (raw float32)  ·  12 GP-predicted (quantised residual)')
ax.text(0.03, 0.97, '(d)', transform=ax.transAxes,
        fontsize=9, color=CNAVY, va='top', fontweight='bold')

# ── shared legend ─────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=CMID, edgecolor=CDGRAY, alpha=0.55,
                   label='Previously resolved (prior level)'),
    mpatches.Patch(facecolor=CTEAL, edgecolor=CNAVY,
                   label='Candidate — to be resolved this level'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=CMINT,
               markersize=14, markeredgecolor=CNAVY, markeredgewidth=0.9,
               label='Anchor — stored as raw float32 (RPCholesky selection)'),
    mpatches.Patch(facecolor=CAMBER, edgecolor=CNAVY,
                   label='GP-predicted — only quantised residual stored'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           fontsize=8.5, framealpha=0.95, edgecolor=CMID,
           bbox_to_anchor=(0.5, 0.01))

# ── section labels above panels ───────────────────────────────────────────────
fig.text(0.255, 0.895, 'Level 0 (GP)', fontsize=9, color=CTEAL,
         ha='center', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor=CICE, edgecolor=CMID, lw=0.8))
fig.text(0.745, 0.895, 'Level 1 (GP)', fontsize=9, color=CTEAL,
         ha='center', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor=CICE, edgecolor=CMID, lw=0.8))
fig.text(0.97, 0.895, 'Levels 2+ →\nCubic Spline', fontsize=7.5,
         color=CMUTED, ha='right', va='center', style='italic')

# ── main title ────────────────────────────────────────────────────────────────
fig.suptitle(
    'Hybrid GP Compression  —  Dyadic Schedule, Levels 0 & 1  '
    '(GP_LEVELS = 2)',
    fontsize=12.5, fontweight='bold', color=CNAVY, y=0.99)

out = '/sessions/serene-keen-ride/mnt/Argonne/hybrid_gp_diagram.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved → {out}')
