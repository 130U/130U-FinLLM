# Run once in a NEW code cell after the existing Kaggle 04e session.
# This is an exploratory development experiment, not a fresh independent test.
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import copy, json, math, random, shutil, traceback, zipfile

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from finllm.checkpoint import load_checkpoint, sha256_file
from finllm.evaluation import run_evaluation
from finllm.kaggle_lite import prepare_lite_sft
from finllm.training import run_training

WORK = Path('/kaggle/working')
PROJECT = WORK / '130U-FinLLM'
LITE = WORK / '130u_lite_input'
OLD_RUN = WORK / '130u_lite_sft_v1'
OUT = WORK / '130u_balanced_sft_v1'
OUT_LIVE = WORK / '130u_balanced_sft_v1_live'
MANIFEST_SHA = '0c7843c92dc1bad4bee9faad6f17d9c7d62bb4e9c2f405d0e6ec3ae2a35a35e4'
SEED = 130
EPOCHS = 3
LABELS = ('bearish', 'neutral', 'bullish')

if not torch.cuda.is_available():
    raise RuntimeError('Connect a GPU first; this experiment does not train on CPU.')
if OUT_LIVE.exists() and not OUT.exists():
    raise FileExistsError('Only the live backup exists; the working directory is missing. Preserve and inspect the backup before resuming.')
for path in [LITE / 'task/train.jsonl', LITE / 'task/development.jsonl',
             OLD_RUN / 'sft_best_development.json', PROJECT / 'runtime_revision.txt']:
    if not path.is_file():
        raise FileNotFoundError(f'Missing session input: {path}. Preserve existing outputs; do not repeat training.')

# prepare_lite_sft validates the entire immutable input and fixed parent hash.
pilot = prepare_lite_sft(PROJECT, LITE, OUT, expected_manifest_sha256=MANIFEST_SHA, epochs=EPOCHS)
cfg = copy.deepcopy(pilot['config'])
train = [json.loads(x) for x in (LITE / 'task/train.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
dev = [json.loads(x) for x in (LITE / 'task/development.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
groups = {label: [row for row in train if row['label'] == label] for label in LABELS}
if any(not groups[k] for k in LABELS) or sum(map(len, groups.values())) != len(train):
    raise ValueError('The training set has invalid or missing three-class labels.')

# Oversampling is restricted to the train partition.  Duplicated rows are
# explicitly recorded as repeated exposures, not additional unique examples.
rng = random.Random(SEED)
largest = max(len(v) for v in groups.values())
balanced = []
for label in LABELS:
    values = groups[label]
    balanced.extend(values)
    balanced.extend(rng.choice(values) for _ in range(largest - len(values)))
rng.shuffle(balanced)
balanced_path = OUT / 'balanced_train.jsonl'
balanced_text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in balanced)
counts = {'unique_original': dict(Counter(row['label'] for row in train)),
          'balanced_per_epoch': dict(Counter(row['label'] for row in balanced)),
          'original_unique_rows': len(train), 'resampled_rows_per_epoch': len(balanced),
          'resampling_seed': SEED, 'max_epochs': EPOCHS,
          'note': 'Repeated exposures are not new unique data.'}
import hashlib
counts['resampled_data_sha256'] = hashlib.sha256(balanced_text.encode('utf-8')).hexdigest()
cfg['data'] = str(balanced_path)
updates = math.ceil(len(balanced) * EPOCHS / (cfg['train']['batch_size'] * cfg['train']['grad_accumulation']))
cfg['train'].update(max_updates=updates, schedule_updates=updates, max_epochs=EPOCHS)
cfg['lite_binding']['epochs'] = EPOCHS
cfg['lite_binding']['balanced_sampling'] = counts
config_path = OUT / 'experiment_config.json'
resume_path = OUT / 'latest.pt'
if OUT.exists():
    # Only continue the exact experiment. Never overwrite mismatched or partial input.
    if not config_path.is_file() or not balanced_path.is_file():
        raise RuntimeError('Existing outputs are incomplete. Preserve the files and stop; do not delete or reinitialize them.')
    if json.loads(config_path.read_text(encoding='utf-8')) != cfg or sha256_file(balanced_path) != counts['resampled_data_sha256']:
        raise RuntimeError('Existing configuration or balanced examples do not match. Preserve the files and stop.')
    if resume_path.exists():
        load_checkpoint(resume_path, allow_previous=False)
        cfg.pop('init_checkpoint')
        cfg['resume'] = str(resume_path)
        print('Resuming after checkpoint verification. The trainer will also verify data, optimizer, scheduler, and runtime state.', flush=True)
    elif any(OUT.glob('*.pt*')) or OUT_LIVE.exists():
        raise RuntimeError('Model files exist but a resumable latest.pt is missing. Preserve the files and stop.')
    else:
        print('Prepared configuration verified. No training checkpoint exists; starting from the original parent model.', flush=True)
else:
    OUT.mkdir()
    balanced_path.write_bytes(balanced_text.encode('utf-8'))
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUT / 'resampling.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
print('GPU:', torch.cuda.get_device_name(0), 'Training class counts:', counts, flush=True)

# Reference baselines see only original training labels and development labels.
def feature(row):
    return 'target ' + row['target'] + ' text ' + row['text']
def scores(truth, pred):
    return {'accuracy': accuracy_score(truth, pred),
            'macro_f1': f1_score(truth, pred, labels=list(LABELS), average='macro', zero_division=0),
            'per_class': classification_report(truth, pred, labels=list(LABELS), output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(truth, pred, labels=list(LABELS)).tolist(),
            'label_order': list(LABELS)}
y_train = [row['label'] for row in train]
y_dev = [row['label'] for row in dev]
majority = Counter(y_train).most_common(1)[0][0]
baseline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED))
baseline.fit([feature(row) for row in train], y_train)
baseline_dev = {'scope': 'development-only', 'majority_label': majority,
                'majority': scores(y_dev, [majority] * len(dev)),
                'tfidf_logistic_balanced': scores(y_dev, baseline.predict([feature(row) for row in dev]))}
(OUT / 'development_baselines.json').write_text(json.dumps(baseline_dev, indent=2), encoding='utf-8')
print('Development majority-class macro-F1:', baseline_dev['majority']['macro_f1'],
      'TF-IDF baseline macro-F1:', baseline_dev['tfidf_logistic_balanced']['macro_f1'], flush=True)

failed = False
try:
    print('Starting or resuming the class-balanced experiment; the first run initializes from the original domain weights.', flush=True)
    training_report = run_training(cfg)
    result = run_evaluation(OUT / 'best.pt', cfg['tokenizer'], cfg['validation_data'],
                            OUT / 'balanced_best_development.json', target_mode='literal_entity')
    old = json.loads((OLD_RUN / 'sft_best_development.json').read_text(encoding='utf-8'))
    summary = {'scope': 'exploratory development; previously viewed test is not independent for future selection',
               'resampling': counts, 'training': training_report,
               'old_sft_development': old['metrics'], 'balanced_sft_development': result['metrics'],
               'baselines': baseline_dev,
               'limits': ['No unseen final test in this experiment', 'Small and imbalanced data',
                          'Literal entity news headlines, not verified ticker or X/Twitter data',
                          'No proof of event independence or chronological generalization']}
    (OUT / 'experiment_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('New model development macro-F1:', result['metrics']['macro_f1'],
          'Previous model:', old['metrics']['macro_f1'], flush=True)
    print('Per-class results:', result['metrics']['per_class'], flush=True)
except Exception:
    failed = True
    error_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    (OUT / ('experiment_error_' + error_stamp + '.txt')).write_text(traceback.format_exc(), encoding='utf-8')
    print('The experiment encountered an error. Packaging existing outputs before stopping.', flush=True)

# Keep this ZIP private. Source examples in the balanced training file are included.
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
archive = WORK / ('130u_balanced_experiment_private_' + stamp + '.zip')
checksums = {}
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            name = path.relative_to(WORK).as_posix()
            checksums[name] = {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
            z.write(path, name)
    z.writestr('CHECKSUMS.json', json.dumps(checksums, indent=2))
print('Experiment backup:', archive, 'size:', archive.stat().st_size, 'SHA256:', sha256_file(archive), flush=True)
if failed:
    raise RuntimeError('The experiment did not complete. Preserve the ZIP and the full error above.')
