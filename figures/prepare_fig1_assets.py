"""Exports what Figure 1 needs for its two pairs (Q1878 Howard -> Leonard, Q1886
Bernadette -> Sheldon): question texts, the option set, correct letters, model answers and
still frames from the annotated outfit windows. Output: fig1_assets.zip (not released,
because it contains video frames).
"""
import json, ast, os, shutil
from pathlib import Path
import decord
from PIL import Image

VAL_ANN = r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json'
BASE_DIR = r'D:\InifiniBenchVideos\train_videos'
RES = Path(r'D:\eval_workspace\results\global_appearance')
PAIRS = ['1878', '1886']
OUT = Path(__file__).with_name('fig1_assets')
OUT.mkdir(exist_ok=True)

ann = {str(d['question_id']): d for d in json.load(open(VAL_ANN, encoding='utf-8'))}
cands = {c['qid']: c for c in
         json.load(open(RES / 'valid_option_candidates.json', encoding='utf-8'))['existing_pairs']['InternVL2-8B']}
preds = {}
for m, folder in (('InternVL2-8B', 'InternVL2-8B'), ('Qwen2.5-VL-7B', 'Qwen25VL')):
    for r in json.load(open(RES / folder / 'predictions_valid_option_swap.json', encoding='utf-8')):
        if r['mode'] == 'video' and r['question_id'] in PAIRS:
            preds.setdefault(r['question_id'], {})[m] = r


def secs(t):
    h, m, s = str(t).split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def windows(tg):
    out = []
    for g in ast.literal_eval(tg):
        if isinstance(g, list) and g and isinstance(g[0], list):
            out.append([(secs(w[0]), secs(w[1])) for w in g])
        elif isinstance(g, list) and g and isinstance(g[0], str):
            out.append([(secs(g[0]), secs(g[1]))])
    return out


info = {}
for qid in PAIRS:
    a, c = ann[qid], cands[qid]
    opts = ast.literal_eval(a['options'])[:5]
    p = preds[qid]
    any_r = next(iter(p.values()))
    xi, yi = 'ABCDE'.index(any_r['x_label']), 'ABCDE'.index(any_r['y_label'])
    opts[yi] = c['y_gold']                     # the modified option set the models saw
    assert int(a['answer_idx']) == xi, 'gold letter mismatch'
    y_ann = ann.get(str(c['target_qid']))
    vr = decord.VideoReader(os.path.join(BASE_DIR, a['video_path_mp4']))
    fps = vr.get_avg_fps()
    frames = []
    for who, q in (('x', a), ('y', y_ann)):
        if q is None:
            continue
        for k, o in enumerate(windows(q['temporal_grounding'])[:4]):
            s, e = o[0]
            t = (s + e) / 2
            img = Image.fromarray(vr[min(len(vr) - 1, int(t * fps))].asnumpy())
            img.thumbnail((640, 640))
            fn = f'q{qid}_{who}{k + 1}.jpg'
            img.save(OUT / fn, quality=92)
            frames.append(dict(file=fn, who=who, outfit=k + 1, t=round(t, 1)))
    info[qid] = dict(
        episode=a['video_path_mp4'], duration_s=len(vr) / fps,
        x=c['x'], y=c['y'], question_x=a['question'], question_y=any_r['question_y'],
        options=[', '.join(o) if isinstance(o, list) else str(o) for o in opts],
        x_label=any_r['x_label'], y_label=any_r['y_label'],
        answers={m: dict(asked_x=r['pred_x'], asked_y=r['pred_y']) for m, r in p.items()},
        frames=frames)
    print(f"Q{qid} {c['x']} -> {c['y']}: X={any_r['x_label']} Y={any_r['y_label']} "
          f"answers={info[qid]['answers']}  frames={len(frames)}")

json.dump(info, open(OUT / 'fig1_assets.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
shutil.make_archive(str(OUT), 'zip', OUT)
print(f'\nWrote {OUT}.zip  -> upload this file')
