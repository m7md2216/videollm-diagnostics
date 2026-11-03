"""Figure 1: the valid-option name swap on two real BBT pairs (video mode).

Assets (fig1_assets/, exported by prepare_fig1_assets.py on the evaluation machine):
  question texts, the exact option set the models saw, gold letters, model answers,
  and still frames from the middle of annotated outfit windows.
  Q1878  Howard -> Leonard      (S10E06)  options D (Howard), A (Leonard); both models D / D
  Q1886  Bernadette -> Sheldon  (S10E08)  options D (Bernadette), B (Sheldon); both models D / B
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

A = json.load(open('fig1_assets/fig1_assets.json', encoding='utf-8'))
INK, INK2, MUTED, RULE = '#111111', '#4a4a47', '#8a8984', '#bdbcb6'
X_C, Y_C = '#2a78d6', '#d9572b'          # original character / new character (same as Figs. 2-3)
OK, BAD = '#1a7f37', '#c62828'
SERIF = 'TeX Gyre Termes'
plt.rcParams.update({'font.family': SERIF, 'font.size': 8})

FW, FH = 6.3, 2.78
fig = plt.figure(figsize=(FW, FH))


def T(x, y, s, **kw):            # text in inches from bottom-left
    kw.setdefault('fontsize', 8); kw.setdefault('color', INK)
    return fig.text(x / FW, y / FH, s, **kw)


def mmss(t):
    return f'{int(t // 60)}:{int(t % 60):02d}'


R = fig.canvas.get_renderer()


def tw(t):
    """rendered width of a Text artist, in inches"""
    return t.get_window_extent(renderer=R).width / fig.dpi


def runs(x, y, parts, size, maxx=None, lh=0.15):
    """Inline runs [(text, color, weight)] with word wrap at maxx (inches). Returns (x, y) end."""
    x_start = x
    for txt, col, wt in parts:
        for i, word in enumerate(txt.split(' ')):
            piece = word if i == 0 else ' ' + word
            if not piece:
                continue
            t = T(x, y, piece, fontsize=size, color=col, fontweight=wt, va='center')
            w = tw(t)
            if maxx is not None and x + w > maxx and x > x_start:
                t.remove(); y -= lh; x = x_start
                t = T(x, y, piece.lstrip(), fontsize=size, color=col, fontweight=wt, va='center')
                w = tw(t)
            x += w
    return x, y


def short(s, n):
    return s if len(s) <= n else s[:n].rsplit(' ', 1)[0].rstrip(',') + ' …'


PANELS = [
    dict(qid='1878', x0=0.05, title='Same gender', frames=[('q1878_x1.jpg', 'x'), ('q1878_y1.jpg', 'y')],
         labels={'q1878_x1.jpg': 'Howard', 'q1878_y1.jpg': 'Leonard'}),
    dict(qid='1886', x0=3.25, title='Cross gender', frames=[('q1886_x1.jpg', 'x'), ('q1886_y1.jpg', 'y')],
         labels={'q1886_x1.jpg': 'Bernadette (left), Sheldon', 'q1886_y1.jpg': 'Sheldon'}),
]
PW = 3.0                                      # panel width (in)

for P in PANELS:
    d = A[P['qid']]; x0 = P['x0']
    X, Y = d['x'].title(), d['y'].title()
    # ---- header
    runs(x0, 2.66, [(P['title'] + ':', INK, 'bold'), (' ' + X, X_C, 'bold'), (' →', INK2, 'normal'),
                   (' ' + Y, Y_C, 'bold')], 8.6)

    # ---- frames
    fw, fh = 1.45, 0.816
    for k, (fn, who) in enumerate(P['frames']):
        fx = x0 + k * (fw + 0.1)
        ax = fig.add_axes([fx / FW, 1.62 / FH, fw / FW, fh / FH])
        ax.imshow(Image.open(f'fig1_assets/{fn}'))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(X_C if who == 'x' else Y_C); s.set_linewidth(1.4)
        t = next(f['t'] for f in d['frames'] if f['file'] == fn)
        T(fx, 1.53, f"{P['labels'][fn]}, {mmss(t)}", fontsize=6.9, color=INK2, style='italic', va='center')

    # ---- question (only the name changes)
    pre, post = d['question_x'].split(X, 1)
    poss = "'s" if post.startswith("'s") else ''
    post = post[len(poss):]
    runs(x0, 1.33, [('Q: ' + pre.strip(), INK2, 'normal'), (' ' + X + poss, X_C, 'bold'), (' /', MUTED, 'normal'),
                    (' ' + Y + poss, Y_C, 'bold'), (post, INK2, 'normal')],
         7.4, maxx=x0 + PW, lh=0.13)

    # ---- the two relevant options (same option set for both questions)
    xl, yl = d['x_label'], d['y_label']
    rows = sorted([(xl, X_C, d['options']['ABCDE'.index(xl)], X),
                   (yl, Y_C, d['options']['ABCDE'.index(yl)], Y)])
    for i, (lab, col, txt, who) in enumerate(rows):
        yy = 0.99 - i * 0.16
        T(x0 + 0.05, yy, lab, fontsize=7.8, fontweight='bold', color=col, va='center')
        T(x0 + 0.2, yy, short(txt, 52), fontsize=7.2, color=INK, va='center')
        T(x0 + PW - 0.02, yy, f'{who}', fontsize=6.9, color=col, va='center', ha='right', style='italic')
    T(x0 + 0.2, 0.67, '+ 3 other options; identical option set and frames for both questions',
      fontsize=6.6, color=MUTED, va='center', style='italic')

    # ---- answers
    cx = [x0 + 1.55, x0 + 2.5]
    T(cx[0], 0.5, f'asked about {X}', fontsize=6.9, color=X_C, ha='center', va='center')
    T(cx[1], 0.5, f'asked about {Y}', fontsize=6.9, color=Y_C, ha='center', va='center')
    fig.add_artist(plt.Line2D([x0 / FW, (x0 + PW) / FW], [0.41 / FH] * 2, color=RULE, lw=0.6))
    for j, (m, ans) in enumerate(d['answers'].items()):
        yy = 0.29 - j * 0.15
        T(x0 + 0.05, yy, m, fontsize=7.4, va='center')
        for c, key, gold in ((cx[0], 'asked_x', xl), (cx[1], 'asked_y', yl)):
            a = ans[key]; ok = a == gold
            T(c - 0.07, yy, a, fontsize=7.8, fontweight='bold', ha='center', va='center',
              color=X_C if a == xl else Y_C if a == yl else INK)
            T(c + 0.07, yy, '✓' if ok else '✗', fontsize=7.6, ha='center', va='center',
              fontfamily='DejaVu Sans', color=OK if ok else BAD)

fig.add_artist(plt.Line2D([3.15 / FW] * 2, [0.05 / FH, 2.75 / FH], color=RULE, lw=0.6))
fig.savefig('fig1_teaser.pdf', dpi=300)
fig.savefig('fig1_teaser.png', dpi=220)
print('ok')
