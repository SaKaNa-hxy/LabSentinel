"""Resume the corrected run, preserving its optimizer, data hashes and log stream."""
import hashlib
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from multiprocessing import freeze_support
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/laboratory_safety_corrected_20261027'
LOGS = ROOT / 'asset/training_corrected_20261027'
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('MPLBACKEND', 'Agg')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    # Append so existing terminal viewers keep following the same files.
    sys.stdout = (LOGS/'stdout.log').open('a', encoding='utf-8', buffering=1)
    sys.stderr = (LOGS/'stderr.log').open('a', encoding='utf-8', buffering=1)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    print(f'\n=== RESUME {stamp} | PID {os.getpid()} ===', flush=True)
    print(f'\n=== RESUME {stamp} ===', file=sys.stderr, flush=True)
    from ultralytics import YOLO
    manifest = json.loads((RUN/'split_manifest.json').read_text(encoding='utf-8'))
    labels = json.loads((ROOT/'datasets/laboratory_safety/label_manifest.json').read_text(encoding='utf-8'))
    def check_data():
        assert all(Path(p).is_file() and sha(p)==digest for p,digest in manifest.items()), 'Dataset metadata changed'
        assert len(labels)==7217 and all((ROOT/p).is_file() and sha(ROOT/p)==digest for p,digest in labels.items()), 'Annotations changed'
    check_data()
    checkpoint = RUN/'weights/last.pt'
    model = YOLO(str(checkpoint))
    saved = model.ckpt
    completed = saved['epoch']+1
    assert completed > 0 and saved.get('optimizer') is not None, 'Checkpoint cannot restore optimizer/epoch'
    assert Path(saved['train_args']['data']).resolve() == (ROOT/'datasets/laboratory_safety/data.yaml').resolve()
    assert completed < saved['train_args']['epochs'] and len(model.names)==12
    backup = LOGS/f'before_resume_{stamp}'
    backup.mkdir()
    for path in [checkpoint, RUN/'weights/best.pt', RUN/'results.csv',RUN/'status.json',RUN/'args.yaml']:
        shutil.copy2(path, backup/path.name)
    event = dict(resumed_at_utc=stamp,pid=os.getpid(),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                 completed_epochs=completed,next_epoch=completed+1,target_epochs=saved['train_args']['epochs'],
                 optimizer_restored=True,script_sha256=sha(__file__),backup=str(backup))
    (LOGS/f'resume_{stamp}.json').write_text(json.dumps(event,indent=2),encoding='utf-8')
    status = dict(pid=os.getpid(),state='resuming',run_dir=str(RUN),completed_epochs=completed,resumed_at_utc=stamp)
    def write_status():
        status['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = RUN/'status.tmp'
        temporary.write_text(json.dumps(status,indent=2),encoding='utf-8')
        temporary.replace(RUN/'status.json')
    def started(trainer):
        assert Path(trainer.save_dir).resolve()==RUN.resolve()
        assert trainer.start_epoch==completed, (trainer.start_epoch,completed)
        status.update(state='training',current_epoch=trainer.start_epoch+1);write_status()
        print(f'Restored checkpoint: continuing epoch {completed+1}/{trainer.epochs}',flush=True)
    def epoch_end(trainer):
        status.update(state='training',completed_epochs=trainer.epoch+1,
                      metrics={k:float(v) for k,v in trainer.metrics.items()});write_status()
    model.add_callback('on_train_start',started)
    model.add_callback('on_fit_epoch_end',epoch_end)
    write_status()
    try:
        model.train(resume=True,device='0',workers=0)
        check_data()
        status.update(state='completed',best_checkpoint=str(RUN/'weights/best.pt'));write_status()
    except BaseException:
        status.update(state='failed_or_interrupted',error=traceback.format_exc());write_status()
        raise

if __name__ == '__main__':
    freeze_support()
    main()
