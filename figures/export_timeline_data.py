"""Data for the episode-timeline figure (reads the annotations; no video frames).

1. For the BBT episode of Q1878 (Figure 1), every global-appearance question in the episode:
   character, annotated outfit windows, episode length and the 16 uniform frame times.
2. For the BBT, Friends and Castle question sets, how often 16 uniform frames fall inside
   the annotated outfit windows.
Writes timeline_data.json.
"""
import json, ast, os, re
from pathlib import Path
import numpy as np
import decord

VAL_ANN = r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json'
TR_ANN = r'D:\eval_workspace\infinibench_annotations\train\skills_json\global_appearance.json'
BASE_DIR = r'D:\InifiniBenchVideos\train_videos'
QWEN = Path(r'D:\eval_workspace\results\global_appearance\Qwen25VL')
EXAMPLE_QID = '1878'
CAST = ['sheldon', 'leonard', 'howard', 'raj', 'penny', 'bernadette', 'amy']


def secs(t):
    h, m, s = str(t).split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def outfits(tg):
    """List of outfits; each outfit is a list of (start, end) windows in seconds."""
    out = []
    for group in ast.literal_eval(tg):
        if isinstance(group, list) and group and isinstance(group[0], list):
            out.append([(secs(w[0]), secs(w[1])) for w in group])
        elif isinstance(group, list) and group and isinstance(group[0], str):
            out.append([(secs(group[0]), secs(group[1]))])
    return out


_len = {}
def video_info(rel):
    if rel not in _len:
        vr = decord.VideoReader(os.path.join(BASE_DIR, rel))
        n, fps = len(vr), vr.get_avg_fps()
        idx = np.linspace(0, n - 1, 16, dtype=int)
        _len[rel] = (n / fps, [float(i / fps) for i in idx])
    return _len[rel]


val = json.load(open(VAL_ANN, encoding='utf-8'))
tr = json.load(open(TR_ANN, encoding='utf-8'))
by_id = {str(d['question_id']): d for d in tr}
by_id.update({str(d['question_id']): d for d in val})

# ---------------------------------------------------------------- 1. example episode
ex = by_id[EXAMPLE_QID]
ep = ex['video_path_mp4']
dur, frame_t = video_info(ep)
rows = []
for d in val:
    if d['video_path_mp4'] != ep or not d.get('is_visual', False):
        continue
    hits = [c for c in CAST if re.search(rf'\b{c}\b', d['question'].lower())]
    rows.append(dict(qid=str(d['question_id']), character=hits[0] if len(hits) == 1 else None,
                     outfits=outfits(d['temporal_grounding'])))
print(f'Example episode {ep}: {dur/60:.1f} min, {len(rows)} questions')
for r in rows:
    print(f"  Q{r['qid']} {r['character']}: {len(r['outfits'])} outfits")

# ---------------------------------------------------------------- 2. coverage
sets = {
    'BBT': [str(d['question_id']) for d in val if d.get('is_visual', False) and 'bbt' in d['video_path_mp4'].lower()],
    'Friends': [str(r['question_id']) for r in json.load(open(QWEN / 'predictions_friends.json', encoding='utf-8'))],
    'Castle': [str(r['question_id']) for r in json.load(open(QWEN / 'predictions_castle.json', encoding='utf-8'))],
}
coverage = {}
for show, qids in sets.items():
    n_out = hit_out = n_q = all_hit = 0
    spacing = []
    for q in qids:
        d = by_id.get(q)
        if d is None:
            continue
        try:
            outs = outfits(d['temporal_grounding'])
            dur_q, ft = video_info(d['video_path_mp4'])
        except Exception as e:
            print(f'  skip Q{q}: {e}'); continue
        if not outs:
            continue
        spacing.append(dur_q / 15)
        hit = [any(s <= t <= e for (s, e) in o for t in ft) for o in outs]
        n_out += len(hit); hit_out += sum(hit); n_q += 1; all_hit += all(hit)
    coverage[show] = dict(questions=n_q, outfits=n_out, outfits_hit=hit_out,
                          questions_all_hit=all_hit, mean_spacing_s=float(np.mean(spacing)))
    print(f'{show:8s} questions {n_q}: outfit windows hit by >=1 uniform frame {hit_out}/{n_out} '
          f'= {100*hit_out/max(n_out,1):.1f}%; questions with every outfit hit {all_hit}/{n_q} '
          f'= {100*all_hit/max(n_q,1):.1f}%; mean spacing {np.mean(spacing):.0f}s')

json.dump(dict(example=dict(qid=EXAMPLE_QID, video=ep, duration_s=dur, frame_times_s=frame_t,
                            questions=rows), coverage=coverage),
          open(Path(__file__).with_name('timeline_data.json'), 'w', encoding='utf-8'), indent=1)
print('\nWrote timeline_data.json')
