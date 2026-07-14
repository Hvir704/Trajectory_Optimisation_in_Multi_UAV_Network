"""
replot_fixed.py  —  Replot objective and nodes graphs from existing result values.
No re-evaluation. Numbers read directly from your optimality_gap_table.txt
and the existing scalability graph.

Run:
    python replot_fixed.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

OUT_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA  — read from your optimality_gap_table.txt and existing graph
# MLP-greedy and MLP+pp are EXACT from the table.
# Baselines are read from the scalability graph (approximate ±2).
# If you have exact baseline numbers, replace them here.
# ══════════════════════════════════════════════════════════════════════════════
M_LIST = [20, 30, 40, 50, 60, 70, 80, 90, 100]

# Exact from plot_scalability.py console output
MLP_GREEDY = [ -9.73, -19.26, -26.42, -33.07, -37.54, -39.52, -45.91, -45.50, -50.88]
MLP_PP     = [-28.59, -35.27, -43.06, -49.35, -55.05, -61.04, -67.17, -67.81, -75.61]

MLP_GREEDY_CI = [1.06, 1.14, 1.31, 1.41, 1.40, 1.45, 1.50, 1.58, 1.67]
MLP_PP_CI     = [0.99, 1.04, 1.13, 1.29, 1.27, 1.28, 1.29, 1.58, 1.47]

RANDOM          = [37.85, 37.22, 38.31, 34.60, 34.39, 34.71, 34.09, 36.21, 32.16]
GREEDY_PRIORITY = [70.71, 70.28, 72.60, 71.07, 69.58, 69.83, 67.75, 70.19, 69.68]
NEAREST_NEIGH   = [37.34, 104.02, 173.91, 217.64, 243.61, 274.86, 288.44, 309.66, 331.26]
PDR             = [93.79, 170.12, 214.15, 241.53, 269.60, 287.34, 314.07, 337.40, 351.23]

RANDOM_CI       = [2.17, 2.18, 2.18, 2.36, 2.07, 2.21, 2.25, 2.17, 2.26]
GP_CI           = [2.34, 2.61, 2.69, 2.73, 2.64, 2.73, 2.29, 3.01, 2.75]
NN_CI           = [3.41, 6.01, 5.61, 5.10, 5.23, 5.16, 5.66, 6.33, 6.37]
PDR_CI          = [3.98, 3.67, 4.23, 4.76, 4.67, 4.85, 5.23, 5.23, 5.78]

# Nodes — exact from console
NODES_MLP_PP     = [15.4, 18.8, 20.6, 21.9, 23.7, 24.1, 26.4, 27.8, 28.7]
NODES_MLP_GREEDY = [14.2, 17.1, 19.0, 19.9, 21.5, 20.2, 22.6, 24.7, 23.5]
NODES_GP         = [13.3, 13.5, 14.0, 14.2, 14.4, 14.4, 14.4, 14.6, 14.8]
NODES_PDR        = [19.9, 28.7, 35.7, 41.5, 47.2, 51.8, 56.5, 60.4, 64.5]
NODES_NN         = [20.0, 30.0, 39.5, 48.0, 54.8, 61.0, 66.7, 71.3, 76.5]
NODES_RANDOM     = [13.1, 13.6, 14.2, 14.0, 14.4, 14.5, 14.6, 14.5, 14.9]

NODES_MLP_PP_CI     = [0.2, 0.3, 0.3, 0.4, 0.3, 0.4, 0.4, 0.4, 0.4]
NODES_MLP_GREEDY_CI = [0.2, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.4, 0.5]
NODES_GP_CI         = [0.3, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
NODES_PDR_CI        = [0.0, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5]
NODES_NN_CI         = [0.0, 0.0, 0.1, 0.2, 0.4, 0.5, 0.5, 0.6, 0.6]
NODES_RANDOM_CI     = [0.3, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

# ══════════════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════════════
STYLE = {
    'Random':             dict(color='#888888', marker='s', ls='--', lw=1.4, ms=5),
    'Nearest-Neighbor':   dict(color='#E69F00', marker='^', ls='--', lw=1.4, ms=5),
    'Greedy-Priority':    dict(color='#56B4E9', marker='D', ls='--', lw=1.4, ms=5),
    'PDR':                dict(color='#CC79A7', marker='v', ls='--', lw=1.4, ms=5),
    'MLP greedy':         dict(color='#4ec994', marker='o', ls='-',  lw=2.0, ms=7),
    'MLP + post-process': dict(color='#004d36', marker='o', ls='-',  lw=2.6, ms=8),
}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — SPLIT-AXIS OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════
def plot_objective():
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={'height_ratios': [1, 1.6]},
        sharex=True
    )
    fig.subplots_adjust(hspace=0.06)

    # ── TOP: baselines ────────────────────────────────────────────────────────
    baselines = {
        'Random':           RANDOM,
        'Nearest-Neighbor': NEAREST_NEIGH,
        'Greedy-Priority':  GREEDY_PRIORITY,
        'PDR':              PDR,
    }
    for name, vals in baselines.items():
        st = STYLE[name]
        ax_top.plot(M_LIST, vals, color=st['color'], marker=st['marker'],
                    ls=st['ls'], lw=st['lw'], markersize=st['ms'], label=name)

    ax_top.axhline(0, color='black', lw=0.8, ls=':')
    ax_top.set_ylim(0, max(PDR) * 1.12)
    ax_top.set_ylabel('Composite Objective', fontsize=11)
    ax_top.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_top.grid(alpha=0.3); ax_top.grid(which='minor', alpha=0.10)
    ax_top.legend(fontsize=9, ncol=2, loc='upper left', framealpha=0.9,
                  title='Baselines (higher = worse)', title_fontsize=8.5)
    ax_top.set_title(
        'Composite Objective vs Network Size  (200 instances/M, 95% CI)',
        fontsize=12, pad=8
    )
    ax_top.spines['bottom'].set_visible(False)
    ax_top.tick_params(bottom=False)

    # broken-axis marks
    d  = 0.012
    kw = dict(color='k', clip_on=False, lw=1.5)
    ax_top.plot((-d, +d), (-d*1.6, +d*1.6),
                transform=ax_top.transAxes, **kw)
    ax_top.plot((1-d, 1+d), (-d*1.6, +d*1.6),
                transform=ax_top.transAxes, **kw)
    ax_bot.plot((-d, +d), (1-d*1.6, 1+d*1.6),
                transform=ax_bot.transAxes, **kw)
    ax_bot.plot((1-d, 1+d), (1-d*1.6, 1+d*1.6),
                transform=ax_bot.transAxes, **kw)

    # ── BOTTOM: MLP curves ────────────────────────────────────────────────────
    ax_bot.axhline(0, color='black', lw=0.8, ls=':')
    ax_bot.spines['top'].set_visible(False)

    mlp_data = {
        'MLP greedy':         MLP_GREEDY,
        'MLP + post-process': MLP_PP,
    }
    for name, vals in mlp_data.items():
        st  = STYLE[name]
        is_pp = 'post' in name
        ax_bot.plot(M_LIST, vals, color=st['color'], marker=st['marker'],
                    ls=st['ls'], lw=st['lw'], markersize=st['ms'],
                    label=name, zorder=4)

        # value label on every point
        for M, v in zip(M_LIST, vals):
            v_off = -10 if is_pp else +5
            va    = 'top' if is_pp else 'bottom'
            ax_bot.annotate(
                f'{v:.1f}',
                xy=(M, v),
                xytext=(0, v_off),
                textcoords='offset points',
                ha='center', va=va,
                fontsize=8.5,
                fontweight='bold' if is_pp else 'normal',
                color=st['color'],
            )

    # y limits with room for labels
    y_lo = min(MLP_PP) * 1.20
    y_hi = 8   # small positive headroom so zero line is visible
    ax_bot.set_ylim(y_lo, y_hi)
    ax_bot.set_ylabel('Composite Objective  (lower = better)', fontsize=11)
    ax_bot.set_xlabel('Number of Sensor Nodes M', fontsize=11)
    ax_bot.set_xticks(M_LIST)
    ax_bot.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_bot.grid(alpha=0.3); ax_bot.grid(which='minor', alpha=0.10)
    ax_bot.legend(fontsize=9.5, loc='upper left', framealpha=0.9,
                  title='Learned policy (lower = better)', title_fontsize=8.5)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_objective_split.png')
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — NODES VISITED
# ══════════════════════════════════════════════════════════════════════════════
def plot_nodes():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # reference line
    ax.plot(M_LIST, M_LIST, color='#AAAAAA', lw=1.0, ls=':',
            label='All nodes (upper bound)', zorder=1)

    nodes_data = [
        ('MLP + post-process', NODES_MLP_PP),
        ('MLP greedy',         NODES_MLP_GREEDY),
        ('Greedy-Priority',    NODES_GP),
        ('Random',             NODES_RANDOM),
        ('Nearest-Neighbor',   NODES_NN),
        ('PDR',                NODES_PDR),
    ]
    for name, vals in nodes_data:
        st    = STYLE[name]
        is_mlp = 'MLP' in name
        is_pp  = 'post' in name
        ax.plot(M_LIST, vals, color=st['color'], marker=st['marker'],
                ls=st['ls'], lw=st['lw'], markersize=st['ms'],
                label=name, zorder=4 if is_mlp else 3)

        if is_mlp:
            for M, v in zip(M_LIST, vals):
                v_off = 5 if is_pp else -11
                va    = 'bottom' if is_pp else 'top'
                ax.annotate(f'{v:.0f}',
                            xy=(M, v),
                            xytext=(0, v_off),
                            textcoords='offset points',
                            ha='center', va=va,
                            fontsize=8,
                            fontweight='bold' if is_pp else 'normal',
                            color=st['color'])

    ax.set_xlabel('Number of Sensor Nodes M', fontsize=12)
    ax.set_ylabel('Average Nodes Visited per Episode', fontsize=12)
    ax.set_title('Average Nodes Visited vs Network Size\n'
                 '(200 instances per M, 95% CI)', fontsize=12)
    ax.set_xticks(M_LIST)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(alpha=0.3); ax.grid(which='minor', alpha=0.10)
    ax.legend(fontsize=9, ncol=2, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_nodes_visited.png')
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved -> {out}')


if __name__ == '__main__':
    plot_objective()
    plot_nodes()
    print('Done.')