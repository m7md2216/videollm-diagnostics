"""Appendix figure: annotated outfit windows vs 16 uniform frames for one BBT episode
(the episode of Figure 1's example, season 10 episode 6). Data: data/timeline_data.json,
exported by export_timeline_data.py from the InfiniBench annotations (no video frames)."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e4e1'
HIT, MISS = '#2a78d6', '#d9d8d3'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8.3, 'xtick.color': INK2})

d = json.load(open('data/timeline_data.json'))['example']
ft = [t / 60 for t in d['frame_times_s']]
rows = [q for q in d['questions'] if q['character']]
order = ['leonard', 'sheldon', 'howard', 'raj', 'penny', 'bernadette', 'amy']
rows.sort(key=lambda q: order.index(q['character']))

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3.05, 3.25), gridspec_kw=dict(height_ratios=[2.05, 1.0], hspace=0.62))
tot_o = tot_h = 0
for i, q in enumerate(rows):
    y = len(rows) - 1 - i
    hits = 0
    for o in q['outfits']:
        hit = any(s <= t * 60 <= e for (s, e) in o for t in ft)
        hits += hit
        for (s, e) in o:
            ax.add_patch(Rectangle((s / 60, y - 0.32), (e - s) / 60, 0.64, color=HIT if hit else MISS,
                                   lw=0, zorder=3))
    tot_o += len(q['outfits']); tot_h += hits
    ax.text(-0.4, y, q['character'].title(), ha='right', va='center', fontsize=7.6, color=INK)
    ax.text(d['duration_s'] / 60 + 0.35, y, f"{hits}/{len(q['outfits'])}", ha='left', va='center',
            fontsize=7.2, color=INK2)
for t in ft:
    ax.axvline(t, color=INK, lw=0.6, alpha=0.55, zorder=4)
ax.set_xlim(0, d['duration_s'] / 60); ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_yticks([]); ax.set_xticks([0, 5, 10, 15])
ax.set_xlabel('Episode time (min)', fontsize=7.6)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(MUTED); ax.tick_params(length=0)
leg = [Rectangle((0, 0), 1, 1, color=HIT, label='outfit window sampled'),
       Rectangle((0, 0), 1, 1, color=MISS, label='missed'),
       Line2D([0], [0], color=INK, lw=0.8, alpha=0.6, label='uniform frame')]
fig.legend(handles=leg, loc='upper center', ncol=3, frameon=False, fontsize=7,
           bbox_to_anchor=(0.55, 1.01), handlelength=1.0, columnspacing=0.8, handletextpad=0.4)
ax.text(-0.3, 1.02, '(a)', transform=ax.transAxes, fontsize=8, color=INK, fontweight='bold')

# (b) coverage per show, from the same export (all questions used in the paper)
cov = json.load(open('data/timeline_data.json'))['coverage']
shows = ['BBT', 'Friends', 'Castle']
win = [100 * cov[s]['outfits_hit'] / cov[s]['outfits'] for s in shows]
allq = [100 * cov[s]['questions_all_hit'] / cov[s]['questions'] for s in shows]
import numpy as np
yy = np.arange(3)[::-1]
ax2.barh(yy + 0.19, win, 0.36, color=HIT, lw=0, zorder=3, label='outfit windows sampled')
ax2.barh(yy - 0.19, allq, 0.36, color='#1b4f8f', lw=0, zorder=3, label='questions with all outfits sampled')
for y, a, b in zip(yy, win, allq):
    ax2.text(a + 1, y + 0.19, f'{a:.0f}%', va='center', fontsize=6.9, color=INK2)
    ax2.text(b + 1, y - 0.19, f'{b:.0f}%', va='center', fontsize=6.9, color=INK2)
ax2.set_yticks(yy); ax2.set_yticklabels([f"{s} ({cov[s]['mean_spacing_s']:.0f} s)" for s in shows], fontsize=7.4)
ax2.set_xlim(0, 100); ax2.set_xticks([0, 25, 50, 75, 100])
for s_ in ('top', 'right', 'left'):
    ax2.spines[s_].set_visible(False)
ax2.spines['bottom'].set_color(MUTED); ax2.tick_params(length=0)
ax2.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
ax2.text(-0.3, 1.05, '(b)', transform=ax2.transAxes, fontsize=8, color=INK, fontweight='bold')
ax2.legend(loc='upper center', bbox_to_anchor=(0.4, -0.3), ncol=1, frameon=False, fontsize=6.8,
           handlelength=0.9, columnspacing=0.8, handletextpad=0.4)
fig.savefig('figB_timeline.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig('figB_timeline.png', dpi=220, bbox_inches='tight', pad_inches=0.02)
print(f'example episode: {tot_h}/{tot_o} outfits sampled, {len(rows)} characters, '
      f'{d["duration_s"]/60:.1f} min, spacing {(ft[1]-ft[0])*60:.1f}s')
