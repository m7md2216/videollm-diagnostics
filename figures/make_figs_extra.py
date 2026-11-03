"""Figures 2-4 and the answer-letter figure. Numbers are taken from the analysis outputs.

fig3_answer_destination: valid-option pairs correct for X, what the model answers for Y
fig4_visual_shift:       video minus black-frame shift, episode-bootstrap 95% intervals
fig5_mc_vs_oe:           multiple-choice accuracy vs open-ended scores
figA_letters:            predicted answer letters vs the gold distribution
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e4e1'
X_C, Y_C, OTHER = '#2a78d6', '#eb6834', '#c9c8c3'   # same roles as Figure 1: X = original, Y = new
plt.rcParams.update({'font.family': 'serif', 'font.size': 8.3, 'axes.edgecolor': MUTED,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK})
W = 3.1


def clean(ax, grid_axis='x'):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    if grid_axis:
        getattr(ax, f'{grid_axis}axis').grid(True, color=GRID, lw=0.6, zorder=0)


# ------------------------------------------------------------------ fig3
rows = [  # (label, n, stays on X, moves to Y)  pooled BBT+Friends+Castle, video
    ('InternVL2, same', 40, 36, 3),
    ('InternVL2, cross', 50, 24, 21),
    ('Qwen, same', 37, 34, 0),
    ('Qwen, cross', 56, 18, 29),
]
fig, ax = plt.subplots(figsize=(3.0, 1.8))
ys = np.array([3.15, 2.25, 1.0, 0.1])
for y, (lab, n, st, mv) in zip(ys, rows):
    ot = n - st - mv
    segs = [(st, X_C, 'white'), (mv, Y_C, INK), (ot, OTHER, INK)]
    left = 0.0
    for k, col, tc in segs:
        w = 100 * k / n
        if k:
            ax.barh(y, w - 0.6 if w > 1 else w, left=left, height=0.62, color=col, lw=0, zorder=3)
            if w >= 7:
                ax.text(left + w / 2, y, str(k), ha='center', va='center', fontsize=7.6,
                        color=tc, zorder=4)
        left += w
    ax.text(-2, y, f'{lab} ($n$={n})', ha='right', va='center', fontsize=7.8, color=INK)
ax.set_xlim(0, 100); ax.set_ylim(-0.45, 3.7)
ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(['0', '25', '50', '75', '100'])
ax.set_xlabel('% of pairs', fontsize=7.6)
clean(ax)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (X_C, Y_C, OTHER)]
fig.legend(handles, ["keeps original's answer", "new char.'s answer", 'other'],
           loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7.3,
           handlelength=0.9, columnspacing=0.9, handletextpad=0.4)
fig.subplots_adjust(left=0.43, right=0.97, top=0.84, bottom=0.22)
fig.savefig('fig3_answer_destination.pdf', bbox_inches='tight', pad_inches=0.02); fig.savefig('fig3_answer_destination.png', dpi=220, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

# ------------------------------------------------------------------ fig4
data = {  # (same: est, lo, hi), (cross: est, lo, hi); episode-level bootstrap
    'InternVL2-8B': {'BBT': ((0.0, -7.4, 8.0), (20.7, 8.5, 34.5)),
                     'Friends': ((0.0, -5.9, 6.1), (32.1, 20.0, 45.1)),
                     'Castle': ((5.0, -11.1, 25.0), (5.9, -13.9, 26.5)),
                     'Pooled': ((0.8, -4.0, 5.7), (21.4, 13.1, 30.0))},
    'Qwen2.5-VL-7B': {'BBT': ((9.3, 0.0, 22.2), (29.3, 12.5, 46.0)),
                      'Friends': ((0.0, -10.3, 10.4), (32.1, 18.4, 46.5)),
                      'Castle': ((-5.0, -17.6, 0.0), (8.8, -9.7, 28.1)),
                      'Pooled': ((3.3, -3.5, 10.7), (25.5, 15.8, 35.4))},
}
ns = {'BBT': '112', 'Friends': '100', 'Castle': '54', 'Pooled': '266'}
fig, axes = plt.subplots(1, 2, figsize=(W, 1.95), sharey=True)
order = ['BBT', 'Friends', 'Castle', 'Pooled']
yb = {s: 3 - i for i, s in enumerate(order)}
yb['Pooled'] = -0.35
for ax, (m, d) in zip(axes, data.items()):
    ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
    for s in order:
        (se, sl, sh), (ce, cl, ch) = d[s]
        y = yb[s]
        ax.plot([sl, sh], [y + 0.17] * 2, color=MUTED, lw=1.2, zorder=2, solid_capstyle='round')
        ax.plot(se, y + 0.17, 'o', ms=4.6, mfc='white', mec=MUTED, mew=1.1, zorder=3)
        ax.plot([cl, ch], [y - 0.17] * 2, color=INK, lw=1.4, zorder=2, solid_capstyle='round')
        ax.plot(ce, y - 0.17, 'D', ms=4.3, color=INK, mec='white', mew=0.6, zorder=3)
    ax.axhline(0.32, color=GRID, lw=0.8)
    ax.set_title(m, fontsize=8.3, color=INK, pad=3)
    ax.set_xlim(-18, 50); ax.set_xticks([0, 20, 40])
    clean(ax)
axes[0].set_yticks([yb[s] for s in order])
axes[0].set_yticklabels([f'{s} ({ns[s]})' for s in order], fontsize=7.8)
fig.supxlabel('Video $-$ black shift toward new character (pp)', fontsize=7.8, y=0.02)
leg = [Line2D([0], [0], marker='o', mfc='white', mec=MUTED, color=MUTED, ms=4.6, label='same gender'),
       Line2D([0], [0], marker='D', color=INK, mec='white', ms=4.3, label='cross gender')]
fig.legend(handles=leg, loc='upper center', ncol=2, frameon=False, fontsize=7.6,
           bbox_to_anchor=(0.6, 1.0), handlelength=1.6)
fig.subplots_adjust(left=0.25, right=0.98, top=0.8, bottom=0.2, wspace=0.12)
fig.savefig('fig4_visual_shift.pdf', bbox_inches='tight', pad_inches=0.02); fig.savefig('fig4_visual_shift.png', dpi=220, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

# ------------------------------------------------------------------ fig5
mods = [('LLaVA-NeXT-Video', 23.1, 0.0, 1.9, 3.8), ('InternVL2-8B', 40.4, 0.0, 5.8, 11.5),
        ('Qwen2.5-VL-7B', 36.5, 0.0, 13.5, 26.9), ('Gemini 2.5 Flash', 65.4, 11.5, 40.4, 69.2)]
fig, ax = plt.subplots(figsize=(W, 1.55))
for i, (m, mc, st, mean, le) in enumerate(mods[::-1]):
    y = i
    ax.plot([st, le], [y, y], color=MUTED, lw=4.2, solid_capstyle='round', alpha=0.45, zorder=2)
    ax.plot(mean, y, 'o', ms=5, mfc='white', mec=INK, mew=1.1, zorder=4)
    ax.plot(mc, y, 's', ms=5, color=INK, zorder=4)
    ax.text(mc, y + 0.3, f'{mc:.1f}', va='bottom', ha='center', fontsize=7.0, color=INK)
ax.set_yticks(range(4)); ax.set_yticklabels([m[0] for m in mods[::-1]], fontsize=7.8)
ax.set_xlim(-2, 75); ax.set_xticks([0, 20, 40, 60]); ax.set_ylim(-0.6, 3.75)
ax.set_xlabel('Score on the same 52 questions (%)', fontsize=7.8)
clean(ax)
leg = [Line2D([0], [0], marker='s', color=INK, lw=0, ms=5, label='MC'),
       Line2D([0], [0], marker='o', mfc='white', mec=INK, lw=0, ms=5, label='open-ended mean'),
       Line2D([0], [0], color=MUTED, lw=4.2, alpha=0.45, label='strict to lenient')]
ax.legend(handles=leg, loc='lower left', bbox_to_anchor=(-0.42, 1.0), ncol=3, frameon=False,
          fontsize=7.2, handlelength=1.3, columnspacing=1.2, handletextpad=0.5)
fig.subplots_adjust(left=0.31, right=0.98, top=0.83, bottom=0.25)
fig.savefig('fig5_mc_vs_oe.pdf', bbox_inches='tight', pad_inches=0.02); fig.savefig('fig5_mc_vs_oe.png', dpi=220, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

# ------------------------------------------------------------------ figA (appendix)
gold = [18.1, 18.9, 17.3, 21.3, 24.4]
pred = {'LLaVA': [69.8, 8.5, 3.1, 18.6, 0.0], 'InternVL2': [3.1, 13.2, 14.0, 27.9, 41.9],
        'Qwen2.5-VL': [11.6, 14.7, 11.6, 26.4, 35.7], 'Gemini': [25.6, 10.9, 11.6, 14.7, 37.2]}
fig, axes = plt.subplots(1, 4, figsize=(W, 1.35), sharey=True)
for ax, (m, v) in zip(axes, pred.items()):
    xs = np.arange(5)
    ax.bar(xs, v, 0.72, color=X_C, lw=0, zorder=3)
    ax.plot(xs, gold, '_', color=INK, ms=8, mew=1.4, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels(list('ABCDE'), fontsize=7.3)
    ax.set_title(m, fontsize=7.8, color=INK, pad=2)
    ax.set_ylim(0, 75)
    clean(ax, 'y')
axes[0].set_ylabel('% of answers', fontsize=7.8)
axes[0].set_yticks([0, 25, 50, 75])
leg = [plt.Rectangle((0, 0), 1, 1, color=X_C, label='predicted'),
       Line2D([0], [0], marker='_', color=INK, lw=0, ms=8, mew=1.4, label='gold')]
fig.legend(handles=leg, loc='upper right', ncol=2, frameon=False, fontsize=7.2,
           bbox_to_anchor=(0.99, 1.02), handlelength=1.0)
fig.subplots_adjust(left=0.13, right=0.99, top=0.76, bottom=0.15, wspace=0.12)
fig.savefig('figA_letters.pdf', bbox_inches='tight', pad_inches=0.02); fig.savefig('figA_letters.png', dpi=220, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)
print('ok')
