"""Gemini 2.5 Flash: BBT name swap and the "I don't know" condition. Resumable; reads the
API key from GEMINI_API_KEY.
"""
import json
import os
import time
import ast
import random
from pathlib import Path

# Set your Gemini API key
API_KEY = os.environ.get('GEMINI_API_KEY', '')
MODEL = 'gemini-2.5-flash'

if not API_KEY:
    print('Set the GEMINI_API_KEY environment variable.')
    print('Get one from https://aistudio.google.com/app/apikey')
    raise SystemExit(1)

# Install: pip install google-genai
from google import genai
client = genai.Client(api_key=API_KEY)

ANNOT_BBT = Path(r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json')
VIDEOS_ROOT = Path(r'D:\InifiniBenchVideos\train_videos')
GEMINI_DIR = Path(r'D:\eval_workspace\results\global_appearance\Gemini')
GEMINI_DIR.mkdir(parents=True, exist_ok=True)

BBT_CAST_MALE = ['Sheldon', 'Leonard', 'Howard', 'Raj']
BBT_CAST_FEMALE = ['Penny', 'Bernadette', 'Amy']
BBT_CAST = BBT_CAST_MALE + BBT_CAST_FEMALE


def find_character(question):
    """Find which BBT character is named in the question."""
    q_lower = question.lower()
    for name in BBT_CAST:
        if name.lower() in q_lower:
            return name
    return None


def resolve_video(video_rel):
    for c in [VIDEOS_ROOT / video_rel,
              VIDEOS_ROOT / video_rel.replace('TV_shows/videos/', '')]:
        if c.exists():
            return c
    return None


def upload_video(video_path):
    """Upload video file to Gemini Files API. Returns the file object."""
    print(f'  Uploading {video_path.name}...')
    file = client.files.upload(file=str(video_path))
    # Wait for processing
    while file.state.name == 'PROCESSING':
        time.sleep(2)
        file = client.files.get(name=file.name)
    if file.state.name == 'FAILED':
        raise RuntimeError(f'Upload failed for {video_path}')
    return file


def call_gemini(video_file, prompt, max_tokens=2048):
    """Call Gemini with a video and prompt. Returns the text response."""
    response = client.models.generate_content(
        model=MODEL,
        contents=[video_file, prompt],
        config={'max_output_tokens': max_tokens, 'temperature': 0.0},
    )
    return response.text.strip() if response.text else ''


def extract_letter(response_text):
    """Extract A/B/C/D/E/F letter from Gemini's response."""
    response = response_text.strip().upper()
    # Look for first standalone letter
    for ch in response:
        if ch in 'ABCDEF':
            return ch
    return '?'


def format_options(options):
    """Format InfiniBench options list into A/B/C/D/E text."""
    lines = []
    for i, opt in enumerate(options):
        letter = chr(ord('A') + i)
        if isinstance(opt, list):
            text = ', '.join(str(x) for x in opt)
        else:
            text = str(opt)
        lines.append(f'{letter}. {text}')
    return '\n'.join(lines)


# Load annotations
with open(ANNOT_BBT) as f:
    annots = json.load(f)


# =============================================================================
# PRIORITY 1: Complete name-swap BBT to n=124
# =============================================================================

def run_name_swap_completion():
    output_file = GEMINI_DIR / 'predictions_name_swap_full.json'

    # Load existing results
    existing = []
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
    done_qids = {str(r['question_id']) for r in existing}
    print(f'Name swap: {len(existing)} already done.')

    # Find candidates: BBT questions with a main-cast character we can swap
    candidates = []
    for a in annots:
        qid = str(a['question_id'])
        if qid in done_qids:
            continue
        orig_char = find_character(a.get('question', ''))
        if not orig_char:
            continue
        video_path = resolve_video(a.get('video_path_mp4', ''))
        if not video_path:
            continue
        candidates.append((qid, orig_char, video_path, a))

    print(f'Name swap: {len(candidates)} candidates remaining.')

    # Random but deterministic swap target selection (same seed as your original run)
    rng = random.Random(42)

    for i, (qid, orig_char, video_path, annot) in enumerate(candidates):
        # Pick a random other character
        other_chars = [c for c in BBT_CAST if c != orig_char]
        swap_char = rng.choice(other_chars)

        try:
            options = ast.literal_eval(annot['options'])
        except (SyntaxError, ValueError):
            print(f'  QID {qid}: bad options, skipping')
            continue

        answer_idx = int(annot.get('answer_idx', 0))
        correct_letter = chr(ord('A') + answer_idx)

        # Upload video once, reuse for both queries
        try:
            video_file = upload_video(video_path)
        except Exception as e:
            print(f'  QID {qid}: upload failed: {e}')
            continue

        question = annot['question']
        opt_text = format_options(options)

        # Original
        orig_prompt = f'{question}\n\n{opt_text}\n\nAnswer with only the letter (A/B/C/D/E).'

        # Swapped
        swap_question = question.replace(orig_char, swap_char)
        swap_prompt = f'{swap_question}\n\n{opt_text}\n\nAnswer with only the letter (A/B/C/D/E).'

        try:
            orig_resp = call_gemini(video_file, orig_prompt, max_tokens=20)
            swap_resp = call_gemini(video_file, swap_prompt, max_tokens=20)
        except Exception as e:
            print(f'  QID {qid}: API error: {e}')
            continue

        orig_pred = extract_letter(orig_resp)
        swap_pred = extract_letter(swap_resp)

        record = {
            'question_id': qid,
            'original_char': orig_char,
            'swap_char': swap_char,
            'correct_label': correct_letter,
            'orig_pred': orig_pred,
            'swap_pred': swap_pred,
            'orig_correct': orig_pred == correct_letter,
            'answer_changed': orig_pred != swap_pred,
        }
        existing.append(record)

        # Save after each call so we don't lose progress
        with open(output_file, 'w') as f:
            json.dump(existing, f, indent=2)

        print(f'  [{len(existing)}/124] QID {qid} {orig_char}->{swap_char}: '
              f'{orig_pred}->{swap_pred} ({"changed" if record["answer_changed"] else "same"})')

        # Stop at 124 to match open-source sample size
        if len(existing) >= 124:
            print('Reached n=124, stopping.')
            break

        # Rate limit politeness
        time.sleep(1)


# =============================================================================
# PRIORITY 2: Abstention experiment on Gemini BBT
# =============================================================================

def run_abstention():
    output_file = GEMINI_DIR / 'predictions_idk.json'

    existing = []
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
    done_qids = {str(r['question_id']) for r in existing}
    print(f'\nAbstention: {len(existing)} already done.')

    # All BBT validation questions
    for i, annot in enumerate(annots):
        qid = str(annot['question_id'])
        if qid in done_qids:
            continue

        video_path = resolve_video(annot.get('video_path_mp4', ''))
        if not video_path:
            continue

        try:
            options = ast.literal_eval(annot['options'])
        except (SyntaxError, ValueError):
            continue

        answer_idx = int(annot.get('answer_idx', 0))
        correct_letter = chr(ord('A') + answer_idx)

        try:
            video_file = upload_video(video_path)
        except Exception as e:
            print(f'  QID {qid}: upload failed: {e}')
            continue

        # Build prompt with added option F
        opt_text = format_options(options)
        next_letter = chr(ord('A') + len(options))  # usually F if 5 options
        opt_text += f'\n{next_letter}. I cannot determine this from the video'

        prompt = (f'{annot["question"]}\n\n{opt_text}\n\n'
                  f'Answer with only the letter (A/B/C/D/E/{next_letter}).')

        try:
            resp = call_gemini(video_file, prompt, max_tokens=20)
        except Exception as e:
            print(f'  QID {qid}: API error: {e}')
            continue

        pred = extract_letter(resp)
        abstained = pred == next_letter

        record = {
            'question_id': qid,
            'correct_label': correct_letter,
            'predicted_letter': pred,
            'idk_letter': next_letter,
            'abstained': abstained,
            'correct': pred == correct_letter,
        }
        existing.append(record)

        with open(output_file, 'w') as f:
            json.dump(existing, f, indent=2)

        marker = 'ABSTAINED' if abstained else ('correct' if record['correct'] else 'wrong')
        print(f'  [{len(existing)}/{len(annots)}] QID {qid}: {pred} [{marker}]')

        time.sleep(1)


if __name__ == '__main__':
    print('=' * 70)
    print('Gemini priority completions')
    print('=' * 70)
    print()
    print('Priority 1: Complete name-swap BBT (current ~23, target 124)')
    print('Priority 2: Add abstention experiment on Gemini BBT (n=129)')
    print()
    print('Estimated cost: $30-60 in Gemini API calls.')
    print()
    response = input('Continue? (y/n): ')
    if response.lower() != 'y':
        raise SystemExit('Aborted.')

    print('\n--- Running name-swap completion ---')
    run_name_swap_completion()

    print('\n--- Running abstention experiment ---')
    run_abstention()

    print('\nDone. Results saved to:')
    print(f'  {GEMINI_DIR / "predictions_name_swap_full.json"}')
    print(f'  {GEMINI_DIR / "predictions_idk.json"}')
