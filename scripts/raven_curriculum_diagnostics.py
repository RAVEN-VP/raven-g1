"""Summarise actual Isaac Sim actor/camera coverage across online curriculum cycles.

Diagnostic only; does not change policy, tracker, or acceptance criteria.
Run from repository root:
 python scripts/raven_curriculum_diagnostics.py --root raven_training_results/supermimic_online_v2
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def analyse(csv_file: Path):
    with csv_file.open(newline='', encoding='utf-8') as h:
        records = list(csv.DictReader(h))
    valid = [r for r in records if str(r.get('detected','')).lower() in ('true','1','yes')]
    if not valid:
        raise ValueError(f'No valid RGB-D detections in {csv_file}')

    def values(key):
        result=[]
        for r in valid:
            try: result.append(float(r[key]))
            except (KeyError, ValueError, TypeError): continue
        return result

    desired, actual, lateral, bearing = (values(k) for k in (
        'desired_pan_deg','actual_pan_deg','actor_y_ground_truth_m','image_bearing_deg'))
    # Baseline tracker has a fixed +/-10 degree sweep. Saturation is measured
    # by the requested target, NOT by measured pan motor response.
    sat = sum(abs(v) >= 9.9 for v in desired) / max(len(desired),1)
    return {'frames':len(records),'detected':len(valid),'saturated_fraction':sat,
            'target_min_deg':min(desired), 'target_max_deg':max(desired),
            'pan_min_deg':min(actual), 'pan_max_deg':max(actual),
            'actor_y_min_m':min(lateral), 'actor_y_max_m':max(lateral),
            'image_bearing_min_deg':min(bearing), 'image_bearing_max_deg':max(bearing),
            'traversed_both_sides':min(lateral)<-0.01 and max(lateral)>0.01,
            'target_traversed_both_sides':min(desired)<-0.5 and max(desired)>0.5}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,
                    default=Path('raven_training_results/supermimic_online_v2'))
    args=ap.parse_args()
    files=sorted(args.root.glob('cycle_*/experience/raven_actor_tracking.csv'))
    if not files: ap.error(f'No completed tracking CSV found under {args.root}')
    for path in files:
        result=analyse(path)
        summary_file=path.with_name('raven_g1_state_action.summary.json')
        meta=json.loads(summary_file.read_text(encoding='utf-8')) if summary_file.is_file() else {}
        print(f'\n{path.parent.parent.name}: motion={meta.get("motion_profile", "legacy sine")} '
              f'actor excursions=({result["actor_y_min_m"]:+.3f}, {result["actor_y_max_m"]:+.3f})m')
        print(f'  detected={result["detected"]}/{result["frames"]}; '
              f'pan target=[{result["target_min_deg"]:+.2f}, {result["target_max_deg"]:+.2f}]deg; '
              f'pan physical=[{result["pan_min_deg"]:+.2f}, {result["pan_max_deg"]:+.2f}]deg')
        print(f'  bearing=[{result["image_bearing_min_deg"]:+.2f}, '
              f'{result["image_bearing_max_deg"]:+.2f}]deg; '
              f'pan target saturation={result["saturated_fraction"]:.1%}')
        if result['traversed_both_sides'] and not result['target_traversed_both_sides']:
            print('  WARNING: actor moved both ways, but pan target did not. '
                  'Check RGB-D bearing reconstruction / reference / sign before extended training.')
        if result['saturated_fraction']>0.80:
            print('  WARNING: >80% pan target saturation; this episode has poor control diversity.')


if __name__=='__main__':
    main()
