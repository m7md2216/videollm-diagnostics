"""Re-queries the Gemini full-episode questions whose API call returned an error
(rate limit), writing predictions_v3.json. A failed call is recorded as 'X'
(unparseable), never as a letter. Reads the API key from GEMINI_API_KEY.
"""
import os, json, ast, time
from pathlib import Path
from google import genai
from google.genai import types

API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    raise SystemExit('Set GEMINI_API_KEY first.')

# Hard cost cap so this cannot run away
MAX_TOTAL_USD = 15.00

PRICE_INPUT_PER_1M  = 0.30   # Gemini 2.5 Flash
PRICE_OUTPUT_PER_1M = 2.50

client   = genai.Client(api_key=API_KEY)
MODEL_ID = 'gemini-2.5-flash'

ANN_PATH = r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json'
BASE_DIR = r'D:\InifiniBenchVideos\train_videos'
OUT_DIR  = Path(r'D:\eval_workspace\results\global_appearance\Gemini')

INPUT_FILE  = OUT_DIR / 'predictions_v2.json'   # has 47 bad rows
OUTPUT_FILE = OUT_DIR / 'predictions_v3.json'   # clean version
COST_FILE   = OUT_DIR / 'cost_log_patch.json'

# ---- Load existing predictions ----
with open(INPUT_FILE) as f:
    existing = json.load(f)
print(f'Loaded {len(existing)} existing predictions from predictions_v2.json')

# Identify error rows
error_qids = {str(r['question_id']) for r in existing
              if 'ERROR' in str(r.get('raw_output', ''))}
print(f'Found {len(error_qids)} rows with API errors that need re-running')

# Load annotations to look up videos and options
with open(ANN_PATH) as f:
    val_data = [d for d in json.load(f) if d.get('is_visual', False)]
items_by_id = {str(d['question_id']): d for d in val_data}

to_redo = [items_by_id[qid] for qid in error_qids if qid in items_by_id]
print(f'Will re-run {len(to_redo)} questions on paid tier\n')

# ---- Cost tracking ----
cost_log = {'total_usd': 0.0, 'queries': []}
if COST_FILE.exists():
    with open(COST_FILE) as f:
        cost_log = json.load(f)
    print(f'Resuming with cumulative patch cost: ${cost_log["total_usd"]:.4f}\n')

def save_cost():
    with open(COST_FILE, 'w') as f:
        json.dump(cost_log, f, indent=2)

def record_cost(qid, in_tok, out_tok):
    c_in  = in_tok  * PRICE_INPUT_PER_1M  / 1_000_000
    c_out = out_tok * PRICE_OUTPUT_PER_1M / 1_000_000
    total = c_in + c_out
    cost_log['total_usd'] += total
    cost_log['queries'].append({
        'qid': qid, 'input_tokens': in_tok,
        'output_tokens': out_tok, 'cost_usd': total,
    })
    save_cost()
    return total

def check_budget():
    if cost_log['total_usd'] >= MAX_TOTAL_USD:
        print(f'\n*** BUDGET CAP REACHED: ${cost_log["total_usd"]:.2f} ***')
        raise SystemExit('Budget cap reached')

# ---- Helpers ----
def parse_options(opts_field):
    try:
        return ast.literal_eval(opts_field)
    except:
        return eval(opts_field)

def parse_gap(tg):
    try:
        ts = ast.literal_eval(tg)
        starts = []
        for group in ts:
            if isinstance(group, list) and isinstance(group[0], list):
                for w in group:
                    h,m,s = w[0].split(':')
                    starts.append(int(h)*3600+int(m)*60+float(s))
            elif isinstance(group, list) and isinstance(group[0], str):
                h,m,s = group[0].split(':')
                starts.append(int(h)*3600+int(m)*60+float(s))
        return (max(starts)-min(starts))/60 if len(starts)>=2 else 0
    except:
        return 0

def build_prompt(item):
    label_chars = 'ABCDE'
    opts = parse_options(item['options'])
    options_text = ''
    for i, opt in enumerate(opts):
        if i >= len(label_chars):
            break
        text = ', '.join(opt) if isinstance(opt, list) else str(opt)
        options_text += f'\n{label_chars[i]}. {text}'
    return (
        f"Watch the full video carefully and track each character's appearance.\n\n"
        f"Question: {item['question']}\n"
        f"Options:{options_text}\n\n"
        f"Reply with ONLY a single letter: A, B, C, D, or E."
    )

def upload_video(vpath):
    vf = client.files.upload(file=vpath)
    while vf.state.name == 'PROCESSING':
        time.sleep(5)
        vf = client.files.get(name=vf.name)
    if vf.state.name == 'FAILED':
        raise Exception('Video processing failed')
    return vf

def extract_text(resp):
    try:
        if resp.text:
            return resp.text.strip()
    except:
        pass
    try:
        for cand in resp.candidates or []:
            for part in (cand.content.parts or []):
                if getattr(part, 'text', None):
                    return part.text.strip()
    except:
        pass
    return ''

def ask_gemini(video_file, prompt):
    config = types.GenerateContentConfig(
        max_output_tokens=2048,
        temperature=0.0,
    )
    resp = client.models.generate_content(
        model=MODEL_ID, contents=[video_file, prompt], config=config,
    )
    text = extract_text(resp)
    in_tok = 0
    out_tok = 0
    try:
        usage = resp.usage_metadata
        in_tok  = getattr(usage, 'prompt_token_count', 0) or 0
        out_tok = getattr(usage, 'candidates_token_count', 0) or 0
    except:
        pass
    return text, in_tok, out_tok

def is_quota_error(e):
    s = str(e).lower()
    return ('429' in s or 'quota' in s or 'resource exhausted' in s
            or 'rate limit' in s or 'exceeded' in s)

# ---- Build a dict of existing predictions for easy updating ----
preds_by_qid = {str(r['question_id']): r for r in existing}

print('Starting patch run...\n')
for item in to_redo:
    check_budget()
    qid = str(item['question_id'])
    vpath = os.path.join(BASE_DIR, item['video_path_mp4'])
    prompt = build_prompt(item)
    gap = parse_gap(item['temporal_grounding'])
    gb = ('small' if gap<5 else 'medium' if gap<10
          else 'large' if gap<15 else 'xlarge')

    # Skip if already patched with a valid prediction in v3
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            v3 = json.load(f)
        v3_map = {str(r['question_id']): r for r in v3}
        if qid in v3_map and 'ERROR' not in str(v3_map[qid].get('raw_output','')):
            print(f'  Q{qid}: already patched, skipping')
            continue

    raw_out = ''
    pred = 'X'
    attempt = 0
    max_retries = 5

    while attempt < max_retries:
        try:
            vf = upload_video(vpath)
            raw_out, in_tok, out_tok = ask_gemini(vf, prompt)
            pred = next((c for c in raw_out.upper() if c in 'ABCDE'), 'X')
            record_cost(qid, in_tok, out_tok)
            try:
                client.files.delete(name=vf.name)
            except:
                pass
            break
        except Exception as e:
            if is_quota_error(e):
                attempt += 1
                print(f'  Q{qid}: quota hit, waiting 60s and retrying ({attempt}/{max_retries})')
                time.sleep(60)
                continue
            print(f'  Q{qid}: ERROR {e}')
            raw_out = f'ERROR: {e}'
            pred = 'X'
            break

    # Update the entry; a failed call is recorded as 'X'.
    correct = pred == item['answer']
    preds_by_qid[qid] = {
        'question_id':   qid,
        'correct_label': item['answer'],
        'prediction':    pred,
        'raw_output':    raw_out,
        'correct':       correct,
        'gap_minutes':   round(gap, 2),
        'gap_bucket':    gb,
    }
    flag = 'OK' if correct else ('--' if pred != 'X' else 'ERR')
    print(f'  [{flag}] Q{qid} | GT={item["answer"]} Pred={pred} | total=${cost_log["total_usd"]:.2f}')

    # Save after every question so we can resume
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(list(preds_by_qid.values()), f, indent=2)
    time.sleep(2)

# ---- Final summary ----
with open(OUTPUT_FILE) as f:
    final = json.load(f)

n_total = len(final)
n_valid = sum(1 for r in final if r['prediction'] in 'ABCDE')
n_correct = sum(1 for r in final if r['correct'] and r['prediction'] in 'ABCDE')
n_still_error = sum(1 for r in final if 'ERROR' in str(r.get('raw_output','')))

print('\n' + '='*60)
print('  PATCH COMPLETE')
print('='*60)
print(f'  Total rows:           {n_total}')
print(f'  Valid predictions:    {n_valid}')
print(f'  Still error:          {n_still_error}')
print(f'  Correct (of valid):   {n_correct}/{n_valid} = {n_correct/max(1,n_valid)*100:.1f}%')
print(f'  Total patch cost:     ${cost_log["total_usd"]:.4f}')
print(f'  Clean output saved:   {OUTPUT_FILE}')
print()
print('  Next step: use predictions_v3.json as the official Gemini full-episode result.')

