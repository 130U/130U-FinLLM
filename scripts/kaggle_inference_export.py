# Run in a new Kaggle code cell after 06 has completed successfully.
# Creates a private, weight-only inference package; does not start another training run.
from pathlib import Path
from datetime import datetime, timezone
import json, shutil, sys, zipfile
import torch
from finllm.checkpoint import load_checkpoint, save_checkpoint, sha256_file
import finllm.evaluation as eval_module
from finllm.evaluation import run_calibration
from finllm.inference import Predictor

WORK = Path('/kaggle/working')
PROJECT = WORK / '130U-FinLLM'
LITE = WORK / '130u_lite_input'
RUN = WORK / '130u_balanced_sft_v1'
BEST = RUN / 'best.pt'
DEV_REPORT = RUN / 'balanced_best_development.json'
for path in (BEST, Path(str(BEST) + '.json'), DEV_REPORT,
             LITE / 'tokenizer/tokenizer.json', LITE / 'task/calibration_temperature.jsonl',
             LITE / 'task/calibration_threshold.jsonl', PROJECT / 'runtime_revision.txt'):
    if not path.is_file():
        raise FileNotFoundError(f'Missing a required stage 06 output: {path}. Keep existing files; do not retrain.')

payload = load_checkpoint(BEST, allow_previous=False)
if payload.get('config', {}).get('stage') != 'sft':
    raise ValueError('The selected checkpoint is not an SFT model.')
report = json.loads(DEV_REPORT.read_text(encoding='utf-8'))
if report['bindings']['checkpoint_sha256'] != sha256_file(BEST):
    raise ValueError('The best checkpoint does not match the development report. Stopping.')
if report['bindings']['tokenizer_sha256'] != sha256_file(LITE / 'tokenizer/tokenizer.json'):
    raise ValueError('The best checkpoint does not match the tokenizer. Stopping.')

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
EXPORT = WORK / ('130u_inference_private_' + stamp)
EXPORT.mkdir()
shutil.copytree(LITE / 'tokenizer', EXPORT / 'tokenizer')
shutil.copytree(PROJECT / 'src/finllm', EXPORT / 'src/finllm',
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copyfile(PROJECT / 'pyproject.toml', EXPORT / 'pyproject.toml')

# Preserve just the model tensors and metadata needed by Predictor; optimizer and
# RNG remain in the original 06 full backup, which is required for resuming training.
small = {'model_state': payload['model_state'], 'model_config': payload['model_config'],
         'tokenizer_hash': payload['tokenizer_hash'],
         'config': {'stage': 'sft', 'mode': payload['config']['mode']},
         'progress': payload.get('progress', {}),
         'source_checkpoint_sha256': sha256_file(BEST)}
save_checkpoint(EXPORT / 'model.pt', small)
check = load_checkpoint(EXPORT / 'model.pt', allow_previous=False)
if check['model_config'] != payload['model_config'] or check['tokenizer_hash'] != payload['tokenizer_hash']:
    raise ValueError('Inference-model metadata differs from the original best checkpoint.')
if set(check['model_state']) != set(payload['model_state']) or any(
        not torch.equal(check['model_state'][k], payload['model_state'][k]) for k in check['model_state']):
    raise ValueError('Inference-model weights differ from the original checkpoint.')
del check, payload, small
print('Stage 06 best model verified. Inference weights: ', sha256_file(EXPORT / 'model.pt'), flush=True)

# Dataset event_id may be blank. A blank value denotes missing, not a common
# event; the original function still checks all text and nonempty group IDs.
strict_disjoint = eval_module.assert_disjoint_calibration
def disjoint_without_blank_ids(first, second):
    fields = ('group_id', 'duplicate_group', 'event_id', 'source_id')
    def clean(rows):
        return [{k: v for k, v in row.items()
                 if not (k in fields and isinstance(v, str) and not v.strip())} for row in rows]
    return strict_disjoint(clean(first), clean(second))
eval_module.assert_disjoint_calibration = disjoint_without_blank_ids
calibration = run_calibration(EXPORT / 'model.pt', EXPORT / 'tokenizer',
                              LITE / 'task/calibration_temperature.jsonl',
                              LITE / 'task/calibration_threshold.jsonl',
                              EXPORT / 'calibration.json', target_mode='literal_entity')
print('Calibration temperature:', calibration['temperature'], 'Review threshold:', calibration['review_threshold'],
      '; calibration uses a small sample and does not establish score reliability.', flush=True)

runner = '''from pathlib import Path
import argparse, json, sys
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'src'))
from finllm.inference import Predictor
parser = argparse.ArgumentParser(description='130U-FinLLM exploratory literal-entity stance')
parser.add_argument('--text', required=True, help='English headline or post')
parser.add_argument('--target', required=True, help='Single-word literal target present in the text')
args = parser.parse_args()
predictor = Predictor(root / 'model.pt', root / 'tokenizer', root / 'calibration.json', target_mode='literal_entity')
print(json.dumps(predictor.predict({'text': args.text, 'target': args.target}), indent=2, ensure_ascii=False))
'''
(EXPORT / 'predict.py').write_text(runner, encoding='utf-8')
(EXPORT / 'README.txt').write_text('''130U-FinLLM experimental inference package. Keep private.\nInstall dependencies from pyproject.toml, then run:\npython predict.py --text "Apple shares rise after earnings" --target Apple\nTargets must be a single word appearing literally in the input. Scores express text stance, not price returns. Training data are news headlines, not validated X/Twitter data. Calibration uses small samples; independent final evaluation is still required. This weight-only export cannot resume training; keep the full 06 backup.\n''', encoding='utf-8')

predictor = Predictor(EXPORT / 'model.pt', EXPORT / 'tokenizer', EXPORT / 'calibration.json', target_mode='literal_entity')
examples = [
    {'text': 'Apple reports stronger earnings and rising revenue.', 'target': 'Apple'},
    {'text': 'Tesla reports falling sales and cuts its forecast.', 'target': 'Tesla'},
    {'text': 'Microsoft will release its results on Thursday.', 'target': 'Microsoft'},
]
outputs = [predictor.predict(row) for row in examples]
print('Actual model outputs on synthetic examples, not market news or accuracy measurements: ', json.dumps(outputs, indent=2, ensure_ascii=False), flush=True)
meta = {'source_checkpoint_sha256': sha256_file(BEST), 'export_checkpoint_sha256': sha256_file(EXPORT / 'model.pt'),
        'tokenizer_sha256': sha256_file(EXPORT / 'tokenizer/tokenizer.json'),
        'revision': (PROJECT / 'runtime_revision.txt').read_text(encoding='utf-8').strip(),
        'development_macro_f1': report['metrics']['macro_f1'],
        'calibration_temperature': calibration['temperature'],
        'review_threshold': calibration['review_threshold'],
        'scope': 'exploratory literal entities in financial news headlines; no ticker mapping or Twitter validation',
        'limits': 'Small calibration sample; prior test exposed; requires new independent evaluation.'}
(EXPORT / 'model_info.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

archive = WORK / (EXPORT.name + '.zip')
checksums = {}
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as out:
    for path in sorted(EXPORT.rglob('*')):
        if path.is_file():
            name = path.relative_to(EXPORT).as_posix()
            checksums[name] = {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
            out.write(path, name)
    out.writestr('CHECKSUMS.json', json.dumps(checksums, indent=2, sort_keys=True))
print('Private inference package:', archive, 'bytes:', archive.stat().st_size, 'SHA256:', sha256_file(archive), flush=True)
print('Keep the full stage 06 backup. This lightweight package supports inference, not training recovery.', flush=True)
