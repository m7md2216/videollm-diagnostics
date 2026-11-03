"""Figure: accuracy lift from padded-oracle frames over uniform frames, per show and model.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

shows = ['BBT\n(77 s spacing)', 'Friends\n(100 s)', 'Castle\n(165 s)']
lift = {'InternVL2-8B': [3.1, 6.2, 10.7], 'Qwen2.5-VL-7B': [14.0, 6.8, 17.9]}
pval = {'InternVL2-8B': [0.541, 0.053, 0.0011], 'Qwen2.5-VL-7B': [0.0009, 0.052, 0.00005]}
color = {'InternVL2-8B': '#2a78d6', 'Qwen2.5-VL-7B': '#eb6834'}
hatch = {'InternVL2-8B': None, 'Qwen2.5-VL-7B': '////'}
INK, MUTED, GRID = '#1f1f1e', '#6b6b68', '#e4e4e1'

def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''

plt.rcParams.update({'font.family': 'serif', 'font.size': 8,
                     'hatch.color': 'white', 'hatch.linewidth': 0.8})
fig, ax = plt.subplots(figsize=(3.25, 2.05))
x = np.arange(3); w = 0.36; gap = 0.02
for i, (m, vals) in enumerate(lift.items()):
    xs = x + (i - 0.5) * (w + gap)
    ax.bar(xs, vals, w, color=color[m], hatch=hatch[m], edgecolor='white',
           linewidth=0, label=m, zorder=3)
    for xi, v, p in zip(xs, vals, pval[m]):
        ax.text(xi, v + 0.4, f'+{v:.1f}{stars(p)}', ha='center', va='bottom',
                fontsize=6.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels(shows, color=INK)
ax.set_ylabel('Accuracy lift (pp)', color=INK)
ax.set_ylim(0, 21)
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(MUTED)
ax.tick_params(axis='y', colors=MUTED, length=0)
ax.tick_params(axis='x', length=0)
ax.legend(frameon=False, loc='upper left', fontsize=7, handlelength=1.2,
          labelcolor=INK)
fig.tight_layout(pad=0.3)
fig.savefig('fig2_oracle_lift.pdf')
fig.savefig('fig2_oracle_lift.png', dpi=220)
