"""LLM judge for the open-ended answers (requires the Claude Code CLI).

Put this script, rubric.txt and the judge input file in the same folder and run it. Each
answer is scored once with a pinned model and the rubric in rubric.txt; the raw reply is
saved. Resumable. Output: rejudge_results.json.
"""
import json, re, shutil, subprocess, sys, time
from pathlib import Path

MODEL = 'claude-sonnet-5-5'       # pinned model; reported in the paper
HERE = Path(__file__).resolve().parent
DATA = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'openended_for_rejudge.json')
RUBRIC = HERE / 'rubric.txt'
OUT = HERE / (sys.argv[2] if len(sys.argv) > 2 else 'rejudge_results.json')

import os
claude = os.environ.get('CLAUDE_BIN') or shutil.which('claude')
if not claude:   # common install locations (native installer, npm, Homebrew)
    home = Path.home()
    for c in [home / '.local/bin/claude', home / '.claude/local/claude', Path('/opt/homebrew/bin/claude'),
              Path('/usr/local/bin/claude'), home / '.npm-global/bin/claude',
              home / 'AppData/Roaming/npm/claude.cmd', home / '.local/bin/claude.exe']:
        if c.exists():
            claude = str(c); break
if not claude:
    sys.exit('Cannot find the Claude Code command-line tool ("claude"). Install Claude Code or set CLAUDE_BIN.')
print(f'Using Claude Code at: {claude}')
base_cmd = (['cmd', '/c', claude] if claude.lower().endswith(('.cmd', '.bat')) else [claude])

data = json.load(open(DATA, encoding='utf-8'))
items, human = data['items'], data.get('human', [])
done = {}
if OUT.exists():
    for r in json.load(open(OUT, encoding='utf-8')):
        done[(r['model'], r['question_id'])] = r
print(f'{len(items)} answers, {len(done)} already judged, model = {MODEL}')


def ask(item):
    user = (f"QUESTION: {item['question']}\n\n"
            f"GROUND-TRUTH OUTFIT SEQUENCE (in order): {item['ground_truth']}\n\n"
            f"MODEL RESPONSE:\n{item['model_answer']}\n")
    cmd = base_cmd + ['-p', 'Score the response on stdin using the rubric. Reply with the JSON object only.',
                      '--model', MODEL, '--system-prompt-file', str(RUBRIC),
                      '--output-format', 'json', '--disallowedTools', '*',
                      '--max-turns', '1', '--no-session-persistence']
    p = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding='utf-8', timeout=300)
    raw = p.stdout.strip()
    try:
        meta = json.loads(raw)
    except Exception:
        meta = {'result': raw, 'stderr': p.stderr[-500:]}
    text = meta.get('result', '') if isinstance(meta, dict) else str(meta)
    m = re.search(r'"score"\s*:\s*"?(0\.5|1\.0|1|0\.0|0)"?', text)
    score = float(m.group(1)) if m else None
    rat = re.search(r'"rationale"\s*:\s*"(.*?)"\s*[,}]', text, re.S)
    used = list(meta.get('modelUsage', {}).keys()) if isinstance(meta, dict) and isinstance(meta.get('modelUsage'), dict) else []
    return dict(score=score, rationale=rat.group(1) if rat else '', raw_result=text,
                model_reported=used, returncode=p.returncode)


results = list(done.values())
for i, it in enumerate(items, 1):
    key = (it['model'], it['question_id'])
    if key in done:
        continue
    for attempt in range(3):
        try:
            r = ask(it)
            if r['score'] is not None:
                break
            print(f'   unparseable reply, retry {attempt + 1}: {r["raw_result"][:120]!r}')
        except subprocess.TimeoutExpired:
            r = dict(score=None, rationale='', raw_result='TIMEOUT', model_reported=[], returncode=-1)
        time.sleep(3)
    rec = dict(model=it['model'], question_id=it['question_id'], old_score=it['old_score'],
               mc_correct=it.get('mc_correct'),
               new_score=r['score'], rationale=r['rationale'], raw_result=r['raw_result'],
               model_reported=r['model_reported'], judge_model_requested=MODEL)
    results.append(rec)
    json.dump(results, open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'[{i}/{len(items)}] {it["model"]:20s} Q{it["question_id"]}  old={it["old_score"]}  new={r["score"]}')


# ------------------------------------------------------------------ comparison
def wkappa(pairs, cats=(0.0, 0.5, 1.0)):
    n = len(pairs)
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    O = [[0] * k for _ in range(k)]
    for a, b in pairs:
        O[idx[a]][idx[b]] += 1
    ra = [sum(O[i]) for i in range(k)]
    cb = [sum(O[i][j] for i in range(k)) for j in range(k)]
    raw = sum(O[i][i] for i in range(k)) / n
    pe = sum(ra[i] * cb[i] for i in range(k)) / n / n
    unw = (raw - pe) / (1 - pe) if pe < 1 else float('nan')
    w = lambda i, j: ((i - j) / (k - 1)) ** 2
    do = sum(w(i, j) * O[i][j] for i in range(k) for j in range(k)) / n
    de = sum(w(i, j) * ra[i] * cb[j] for i in range(k) for j in range(k)) / n / n
    return raw, unw, (1 - do / de) if de else float('nan')


def norm(x):
    try:
        x = float(x)
    except Exception:
        return None
    return x if x in (0.0, 0.5, 1.0) else None


print('\n' + '=' * 70)
print(f'RESULTS  (judge requested: {MODEL}; reported by Claude Code: '
      f'{sorted({m for r in results for m in r.get("model_reported", [])}) or "n/a"})')
print('=' * 70)
failed = [r for r in results if r['new_score'] is None]
print(f'unscored after 3 tries: {len(failed)}')
by_model = {}
for r in results:
    by_model.setdefault(r['model'], []).append(r)
mcs = {(it['model'], it['question_id']): it.get('mc_correct') for it in items}
print(f'\n{"model":22s} {"n":>3}  {"MC":>5} {"new mean":>8} {"gap":>6}  {"strict":>6} {"lenient":>7}  {"full":>7}')
for m, rs in by_model.items():
    ok = [r for r in rs if norm(r['new_score']) is not None]
    mcv = [mcs.get((m, r['question_id'])) for r in ok if mcs.get((m, r['question_id'])) is not None]
    if ok:
        nm = 100 * sum(norm(r['new_score']) for r in ok) / len(ok)
        st = 100 * sum(norm(r['new_score']) == 1.0 for r in ok) / len(ok)
        le = 100 * sum(norm(r['new_score']) > 0 for r in ok) / len(ok)
        mc = 100 * sum(mcv) / len(mcv) if mcv else float('nan')
        print(f'{m:22s} {len(ok):3d}  {mc:5.1f} {nm:8.1f} {nm - mc:+6.1f}  {st:6.1f} {le:7.1f}  '
              f'{sum(norm(r["new_score"]) == 1.0 for r in ok):3d}/{len(ok):<3d}')
print(f'\n{"model":22s} {"n":>3}  {"old mean":>8} {"new mean":>8}  {"old full":>8} {"new full":>8}')
for m, rs in by_model.items():
    ok = [r for r in rs if norm(r['new_score']) is not None and norm(r['old_score']) is not None]
    if not ok:
        continue
    om = 100 * sum(norm(r['old_score']) for r in ok) / len(ok)
    nm = 100 * sum(norm(r['new_score']) for r in ok) / len(ok)
    of = sum(norm(r['old_score']) == 1.0 for r in ok)
    nf = sum(norm(r['new_score']) == 1.0 for r in ok)
    print(f'{m:22s} {len(ok):3d}  {om:8.1f} {nm:8.1f}  {of:4d}/{len(ok):<3d} {nf:4d}/{len(ok):<3d}')

pairs = [(norm(r['old_score']), norm(r['new_score'])) for r in results
         if norm(r['old_score']) is not None and norm(r['new_score']) is not None]
raw, unw, qw = wkappa(pairs)
print(f'\nNEW vs OLD chat judge (n={len(pairs)}): raw agreement {100*raw:.1f}%, '
      f'kappa {unw:.2f} unweighted, {qw:.2f} quadratic-weighted')

if human:
    new = {(r['model'], r['question_id']): norm(r['new_score']) for r in results}
    old = {(r['model'], r['question_id']): norm(r['old_score']) for r in results}
    new_q = {}
    for (m, q), v in new.items():
        new_q.setdefault(q, []).append(((m, q), v))
    hp_new, hp_old = [], []
    for h in human:
        hs, q, m = norm(h['human_score']), str(h['question_id']), h.get('model', '')
        key = (m, q) if (m, q) in new else (new_q[q][0][0] if len(new_q.get(q, [])) == 1 else None)
        if hs is None or key is None:
            continue
        if new.get(key) is not None:
            hp_new.append((hs, new[key]))
        if old.get(key) is not None:
            hp_old.append((hs, old[key]))
    for lab, pp in (('NEW judge vs human', hp_new), ('OLD judge vs human', hp_old)):
        raw, unw, qw = wkappa(pp)
        print(f'{lab} (n={len(pp)}): raw {100*raw:.1f}%, kappa {unw:.2f} unweighted, {qw:.2f} quadratic')
else:
    print('\nNo human scores in the data file.')
print(f'\nSaved every reply to {OUT}')
