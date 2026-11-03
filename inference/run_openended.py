"""Open-ended condition (no answer options) for InternVL2, Qwen and LLaVA on the same 52 BBT
questions: identical open-ended prompt, 16 uniform frames, greedy decoding, at most 512
new tokens. InternVL2 uses the same frame labels and input as its multiple-choice runs.
Writes <model>/predictions_openended_v2.json. Resumable.
"""
import os, json, ast, sys, torch, numpy as np
from pathlib import Path
from PIL import Image
import decord
decord.bridge.set_bridge('torch')

ANN_PATH = r'D:\eval_workspace\infinibench_annotations\validation\skills_json\global_appearance.json'
BASE_DIR = r'D:\InifiniBenchVideos\train_videos'
RES = Path(r'D:\eval_workspace\results\global_appearance')
PATHS = {'InternVL2-8B': r'D:\models\internvl2_8b', 'Qwen25VL': r'D:\models\qwen25_vl_7b',
         'LLaVA-NeXT-Video-7B': r'D:\models\llava_next_video_7b'}
MAX_NEW = 512
N_FRAMES = 16
RUN = sys.argv[1:] or ['InternVL2-8B', 'Qwen25VL', 'LLaVA-NeXT-Video-7B']   # optional: pick models

ann = {str(d['question_id']): d for d in json.load(open(ANN_PATH, encoding='utf-8'))}
ref = json.load(open(RES / 'Qwen25VL' / 'predictions_openended_judged.json', encoding='utf-8'))
QIDS = [str(r['question_id']) for r in ref]
GT = {str(r['question_id']): r['ground_truth'] for r in ref}
print(f'{len(QIDS)} open-ended questions (same set as the original Qwen/Gemini runs)')


def prompt_for(q):
    return (f"Watch the full video carefully and track each character's appearance.\n\n"
            f"Question: {q}\n\n"
            f"Answer in detail — describe each outfit in the order it appears. "
            f"Be specific about colors, clothing items, and layering.")


def load_frames(vpath, n=N_FRAMES):
    vr = decord.VideoReader(vpath)
    idx = np.linspace(0, len(vr) - 1, n, dtype=int)
    arr = vr.get_batch(idx).numpy()
    return [Image.fromarray(arr[i]) for i in range(n)]


def run(folder, infer):
    out = RES / folder / 'predictions_openended_v2.json'
    results = json.load(open(out, encoding='utf-8')) if out.exists() else []
    done = {r['question_id'] for r in results}
    todo = [q for q in QIDS if q not in done]
    print(f'\n--- {folder}: done {len(done)}, remaining {len(todo)}')
    for q in todo:
        a = ann[q]
        try:
            text, n_tok = infer(load_frames(os.path.join(BASE_DIR, a['video_path_mp4'])),
                                prompt_for(a['question']))
        except Exception as e:
            print(f'  ERROR Q{q}: {e}'); torch.cuda.empty_cache(); continue
        results.append(dict(question_id=q, question=a['question'], ground_truth=GT[q],
                            model_answer=text, new_tokens=n_tok, hit_limit=n_tok >= MAX_NEW,
                            max_new_tokens=MAX_NEW, n_frames=N_FRAMES))
        json.dump(results, open(out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print(f'  Q{q}: {n_tok} tokens{"  (HIT LIMIT)" if n_tok >= MAX_NEW else ""} | {text[:70]!r}')
        torch.cuda.empty_cache()
    hit = sum(r['hit_limit'] for r in results)
    print(f'  {folder}: {len(results)} answers, {hit} reached the {MAX_NEW}-token limit')


if 'InternVL2-8B' in RUN:
    from transformers import AutoTokenizer, AutoModel
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    tf = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC), T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    tok = AutoTokenizer.from_pretrained(PATHS['InternVL2-8B'], trust_remote_code=True, use_fast=False)
    mdl = AutoModel.from_pretrained(PATHS['InternVL2-8B'], torch_dtype=torch.float16,
                                    device_map='cuda:0', trust_remote_code=True).eval()

    def iv(frames, prompt):
        pv = torch.stack([tf(f) for f in frames]).half().cuda()
        ftxt = '\n'.join(f'Frame{i+1}: <image>' for i in range(len(frames)))
        resp = mdl.chat(tok, pv, f'{ftxt}\n{prompt}', dict(max_new_tokens=MAX_NEW, do_sample=False))
        return resp, len(tok(resp, add_special_tokens=False).input_ids)

    run('InternVL2-8B', iv)
    del mdl, tok; torch.cuda.empty_cache()

if 'Qwen25VL' in RUN:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    proc = AutoProcessor.from_pretrained(PATHS['Qwen25VL'], trust_remote_code=True)
    qm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        PATHS['Qwen25VL'], torch_dtype=torch.float16, device_map='cuda:0', trust_remote_code=True).eval()

    def qw(frames, prompt):
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': f} for f in frames] +
                 [{'type': 'text', 'text': prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ii, vi = process_vision_info(msgs)
        inputs = proc(text=[text], images=ii, videos=vi, padding=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = qm.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
        gen = out[0][inputs.input_ids.shape[1]:]
        return proc.decode(gen, skip_special_tokens=True).strip(), int(len(gen))

    run('Qwen25VL', qw)
    del qm, proc; torch.cuda.empty_cache()

if 'LLaVA-NeXT-Video-7B' in RUN:
    from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
    lproc = LlavaNextVideoProcessor.from_pretrained(PATHS['LLaVA-NeXT-Video-7B'])
    lm = LlavaNextVideoForConditionalGeneration.from_pretrained(
        PATHS['LLaVA-NeXT-Video-7B'], torch_dtype=torch.float16,
        device_map='cuda:0', low_cpu_mem_usage=True).eval()

    def lv(frames, prompt_text):   # loader and slicing as in run_llava_64frames.py
        np_frames = np.stack([np.array(f) for f in frames])
        conv = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': prompt_text}]}]
        prompt = lproc.apply_chat_template(conv, add_generation_prompt=True)
        inputs = lproc(text=[prompt], videos=[np_frames], padding=True,
                       return_tensors='pt').to('cuda', torch.float16)
        with torch.no_grad():
            out = lm.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
        gen = out[0][inputs.input_ids.shape[1]:]
        return lproc.decode(gen, skip_special_tokens=True).strip(), int(len(gen))

    run('LLaVA-NeXT-Video-7B', lv)
print('\nDone. Next: python export_openended_for_judge.py')
