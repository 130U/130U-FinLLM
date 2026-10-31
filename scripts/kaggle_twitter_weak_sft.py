# One Kaggle code cell. Exploratory WEAK supervision; not verified asset labels.
# Requires the still-running private 04e/06 Kaggle session, Internet and GPU.
# 06 best and its outputs are read only. Old held-out labels are not used.
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import copy, csv, hashlib, io, json, math, random, re, unicodedata, urllib.request, zipfile
import torch
from finllm.checkpoint import load_checkpoint, sha256_file
from finllm.evaluation import run_evaluation
from finllm.inference import validate_input
from finllm.kaggle_lite import prepare_lite_sft
from finllm.tokenizer import encode_prompt, load_tokenizer
from finllm.training import run_training

WORK = Path('/kaggle/working')
PROJECT = WORK / '130U-FinLLM'
LITE = WORK / '130u_lite_input'
BASE = WORK / '130u_balanced_sft_v1'
OUT = WORK / '130u_twitter_weak_v1'
OUT_LIVE = WORK / '130u_twitter_weak_v1_live'
BEST = BASE / 'best.pt'
REVISION = '8f939567b2e36d007725d8a860219079da6d851a'
URL = f'https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/{REVISION}/sent_train.csv'
MANIFEST_SHA = '0c7843c92dc1bad4bee9faad6f17d9c7d62bb4e9c2f405d0e6ec3ae2a35a35e4'
SEED, EPOCHS, MAX_PER_CLASS = 130, 2, 500
LABELS = ('bearish', 'neutral', 'bullish')
LABEL_MAP = {'0': 'bearish', '1': 'bullish', '2': 'neutral'}
if not torch.cuda.is_available():
    raise RuntimeError('请连接 GPU；本实验不在 CPU 上正式训练。')
required = [BEST, Path(str(BEST) + '.json'), BASE / 'balanced_best_development.json',
            LITE / 'lite_manifest.json', LITE / 'tokenizer/tokenizer.json',
            PROJECT / 'runtime_revision.txt']
required += [LITE / 'task' / (x + '.jsonl') for x in ('train','development','calibration_temperature','calibration_threshold','test')]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    raise FileNotFoundError('当前会话缺少 06 检查点或输入；请保留成果，不要重训：\n' + '\n'.join(missing))
parent = load_checkpoint(BEST, allow_previous=False)
parent_report = json.loads((BASE / 'balanced_best_development.json').read_text(encoding='utf-8'))
if parent['config']['stage'] != 'sft' or parent_report['bindings']['checkpoint_sha256'] != sha256_file(BEST):
    raise ValueError('06 模型与开发集报告不匹配。')
if parent_report['bindings']['tokenizer_sha256'] != sha256_file(LITE / 'tokenizer/tokenizer.json'):
    raise ValueError('06 模型和词表不匹配。')

def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def canonical(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r'https?://\S+', ' ', text)
    return ' '.join(re.findall(r'[a-z0-9]+', text))
def grams(text, width=8):
    words = canonical(text).split()
    return {' '.join(words[i:i+width]) for i in range(len(words)-width+1)}
def ticker_for(text):
    # A single cashtag does not prove that other, untagged assets are absent.
    no_urls = re.sub(r'https?://\S+', ' ', text)
    tags = re.findall(r'(?<!\w)\$([A-Za-z][A-Za-z0-9.\-]{0,14})(?![A-Za-z0-9_.-])', no_urls)
    if len(tags) != 1 or not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,14}', tags[0]):
        return None
    return tags[0]

original = rows(LITE / 'task/train.jsonl')
heldout = []
for name in ('development','calibration_temperature','calibration_threshold','test'):
    heldout.extend(rows(LITE / 'task' / (name + '.jsonl')))
held_keys = {canonical(r['text']) for r in heldout}
held_grams = set().union(*(grams(r['text']) for r in heldout))
existing_keys = {canonical(r['text']) for r in original}
print('下载固定版本公开 Twitter 金融数据；不上传私有标题。', flush=True)
request = urllib.request.Request(URL, headers={'User-Agent': '130U-FinLLM-research/1.0'})
with urllib.request.urlopen(request, timeout=90) as response:
    raw = response.read(5_000_001)
if len(raw) > 5_000_000:
    raise ValueError('下载大小超出预期，已停止。')
source_sha = hashlib.sha256(raw).hexdigest()
reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
if not {'text', 'label'}.issubset(reader.fieldnames or []):
    raise ValueError(f'来源 CSV 的列不符合预期：{reader.fieldnames}')
tokenizer = load_tokenizer(LITE / 'tokenizer')
groups = {k: [] for k in LABELS}
seen = {}
conflicting_ids = set()
stats = Counter()
for source in reader:
    stats['source_rows'] += 1
    text, label = (source.get('text') or '').strip(), LABEL_MAP.get(str(source.get('label', '')).strip())
    target = ticker_for(text)
    if not text or label is None or target is None:
        stats['not_single_cashtag_or_invalid'] += 1
        continue
    if validate_input({'text': text, 'target': target}, target_mode='literal_entity'):
        stats['input_not_supported'] += 1
        continue
    key = canonical(text)
    if not key or key in held_keys or (grams(text) & held_grams):
        stats['overlap_old_heldout'] += 1
        continue
    if key in existing_keys:
        stats['duplicate_train'] += 1
        continue
    if key in seen:
        if seen[key] != label:
            conflicting_ids.add('twitter_weak_' + hashlib.sha256(key.encode('utf-8')).hexdigest())
            stats['conflicting_duplicate'] += 1
        else:
            stats['duplicate_train'] += 1
        continue
    try:
        encode_prompt(tokenizer, text, target, parent['model_config']['max_position_embeddings'])
    except ValueError:
        stats['invalid_prompt'] += 1
        continue
    seen[key] = label
    digest = hashlib.sha256(key.encode('utf-8')).hexdigest()
    groups[label].append({'id': 'twitter_weak_' + digest, 'group_id': 'twitter_weak_' + digest,
                          'text': text, 'target': target, 'label': label, 'split': 'train',
                          'source': f'zeroshot/twitter-financial-news-sentiment@{REVISION}',
                          'rights_note': 'Dataset card declares MIT; original post rights not independently audited',
                          'training_allowed': True, 'synthetic': False, 'human_reviewed': False,
                          'label_scope': 'weak_single_cashtag_whole_post_sentiment_not_audited_asset_stance'})
groups = {label: [r for r in candidates if r['id'] not in conflicting_ids]
          for label, candidates in groups.items()}
if any(len(v) < 50 for v in groups.values()):
    raise ValueError('筛选后某一类别少于 50 条，避免开始偏斜训练：' + str({k: len(v) for k,v in groups.items()}))
rng = random.Random(SEED)
take = min(MAX_PER_CLASS, *(len(groups[k]) for k in LABELS))
weak = []
for label in LABELS:
    weak.extend(rng.sample(groups[label], take))
# Mix explicitly reviewed headlines with weak Twitter labels in equal class groups.
original_groups = {k: [r for r in original if r['label'] == k] for k in LABELS}
if any(not v for v in original_groups.values()):
    raise ValueError('原训练数据缺少类别。')
largest = max(map(len, original_groups.values()))
mixed = list(weak)
for label in LABELS:
    items = original_groups[label]
    mixed.extend(items)
    mixed.extend(rng.choice(items) for _ in range(largest-len(items)))
rng.shuffle(mixed)
mixed_text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in mixed)
mixed_sha = hashlib.sha256(mixed_text.encode('utf-8')).hexdigest()
pilot = prepare_lite_sft(PROJECT, LITE, OUT, expected_manifest_sha256=MANIFEST_SHA, epochs=EPOCHS)
cfg = copy.deepcopy(pilot['config'])
cfg['data'] = str(OUT / 'mixed_train.jsonl')
cfg['init_checkpoint'] = str(BEST)
cfg['lite_binding']['twitter_weak_experiment'] = {'source_url': URL, 'source_sha256': source_sha,
    'parent_checkpoint_sha256': sha256_file(BEST), 'mixed_sha256': mixed_sha, 'seed': SEED,
    'weak_per_class': take, 'reviewed_resampled_per_class': largest,
    'label_scope': 'weak whole-post labels, not audited target-asset stance'}
updates = math.ceil(len(mixed) * EPOCHS / (cfg['train']['batch_size'] * cfg['train']['grad_accumulation']))
cfg['train'].update(max_updates=updates, schedule_updates=updates, max_epochs=EPOCHS, learning_rate=1e-5)
config_path = OUT / 'experiment_config.json'
if OUT_LIVE.exists() and not OUT.exists():
    raise RuntimeError('只找到备份目录，工作目录缺失；保留备份并停止。')
if OUT.exists():
    if not config_path.is_file() or not (OUT / 'mixed_train.jsonl').is_file():
        raise RuntimeError('已有输出不完整，请保留并检查。')
    if json.loads(config_path.read_text(encoding='utf-8')) != cfg or sha256_file(OUT / 'mixed_train.jsonl') != mixed_sha:
        raise RuntimeError('已有配置或数据不一致，未覆盖。')
    latest = OUT / 'latest.pt'
    if latest.exists():
        load_checkpoint(latest, allow_previous=False)
        cfg.pop('init_checkpoint')
        cfg['resume'] = str(latest)
    elif any(OUT.glob('*.pt*')) or OUT_LIVE.exists():
        raise RuntimeError('已有检查点不完整，保留原文件。')
else:
    OUT.mkdir()
    (OUT / 'mixed_train.jsonl').write_bytes(mixed_text.encode('utf-8'))
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
manifest = {'dataset': URL, 'dataset_revision': REVISION, 'source_sha256': source_sha,
            'license': 'Dataset card declares MIT; original post rights not independently audited',
            'source_counts': dict(stats), 'available_weak_by_class': {k:len(v) for k,v in groups.items()},
            'selected_weak_per_class': take, 'reviewed_resampled_per_class': largest,
            'reviewed_unique': len(original), 'train_exposures_per_epoch': len(mixed),
            'epochs': EPOCHS, 'updates_budget': updates, 'mixed_sha256': mixed_sha,
            'parent_checkpoint_sha256': sha256_file(BEST),
            'scope': 'Weak whole-post sentiment applied to one visible cashtag; not audited asset stance',
            'limits': ['Old heldouts excluded by normalized text and shared eight-word spans only',
                       'Non-cashtag asset mentions and paraphrased event leakage are not ruled out',
                       'Twitter validation split not read; development is reused and is not a final test',
                       'Not evaluated for real trading or price prediction']}
(OUT / 'data_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
print('训练准备:', json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
print('训练从 06 best 权重开始；弱标签是实验材料，保留原模型。', flush=True)
training = run_training(cfg)
comparison = run_evaluation(OUT / 'best.pt', cfg['tokenizer'], cfg['validation_data'],
                            OUT / 'twitter_best_development.json', target_mode='literal_entity')
summary = {'training': training, 'baseline_06_development': parent_report['metrics'],
           'twitter_weak_development': comparison['metrics'], 'manifest': manifest,
           'scope': 'Exploratory reused development comparison only; no independent test claim'}
(OUT / 'experiment_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print('06 开发集 macro-F1:', parent_report['metrics']['macro_f1'], flush=True)
print('Twitter 弱监督开发集 macro-F1:', comparison['metrics']['macro_f1'], flush=True)
print('各类结果:', json.dumps(comparison['metrics']['per_class'], ensure_ascii=False), flush=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
archive = WORK / ('130u_twitter_weak_private_' + stamp + '.zip')
checks = {}
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            name = path.relative_to(WORK).as_posix()
            checks[name] = {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
            z.write(path, name)
    z.writestr('CHECKSUMS.json', json.dumps(checks, indent=2, sort_keys=True))
print('私有训练包:', archive, '大小:', archive.stat().st_size, 'SHA256:', sha256_file(archive), flush=True)
