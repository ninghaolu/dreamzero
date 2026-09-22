"""Record Slurm state and training progress for an authorized ABC job."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_id', type=int)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--status-file', required=True, type=Path)
    parser.add_argument('--hours', type=float, default=12)
    args = parser.parse_args()
    deadline = time.monotonic() + args.hours*3600
    previous = None
    active = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'}
    args.status_file.parent.mkdir(parents=True, exist_ok=True)
    while time.monotonic() < deadline:
        report = dict(job_id=args.job_id, checked_utc=datetime.now(timezone.utc).isoformat())
        try:
            result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', str(args.job_id),
                                     '--format=JobIDRaw,State,ExitCode,NodeList'],
                                    check=True, capture_output=True, text=True, timeout=30)
            entries = [line.split('|') for line in result.stdout.splitlines()]
            entry = next((entry for entry in entries if entry[0] == str(args.job_id)), None)
            if entry is None:
                report['state'] = 'UNKNOWN'
            else:
                report.update(state=entry[1].split()[0], exit_code=entry[2], nodes=entry[3])
            report['progress'] = []
            for path in sorted(args.output_dir.glob('resources-rank-*.json')):
                try:
                    report['progress'].append(dict(rank_file=path.name, **json.loads(path.read_text())))
                except (OSError, ValueError):
                    pass  # The training callback may currently be writing this file.
            if report['state'] == 'COMPLETED':
                validation = args.output_dir / 'smoke_validation.json'
                report['validation'] = json.loads(validation.read_text()) if validation.exists() else {'passed': False, 'reason': 'Missing smoke validation report'}
        except (OSError, subprocess.SubprocessError) as exc:
            report.update(state='QUERY_ERROR', error=str(exc))
        tmp = args.status_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2)+'\n')
        os.replace(tmp, args.status_file)
        if report['state'] != previous:
            print(json.dumps(report), flush=True)
            previous = report['state']
        if report['state'] not in active | {'UNKNOWN', 'QUERY_ERROR'}:
            return 0 if report['state'] == 'COMPLETED' and report.get('validation', {}).get('passed') else 1
        time.sleep(60)
    print('Monitoring time limit reached; job was not cancelled.', flush=True)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
