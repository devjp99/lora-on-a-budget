"""Select four LRs only after all eight registered BillSum trials complete."""
import argparse
import json
from pathlib import Path
from lora_budget.billsum_protocol import atomic_json, choose_trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', default='results/billsum/training')
    parser.add_argument('--output', default='results/billsum/selected_lr.json')
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in sorted(Path(args.reports).glob('*.json'))]
    result = choose_trials(reports)
    result['source_reports'] = [str(p) for p in sorted(Path(args.reports).glob('*.json'))]
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
