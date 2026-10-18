"""Run one checked mission from the command line and print honest metrics."""
import argparse
import json
import sys
from fpv_sim.mission import PROFILES, SCENARIOS, demo_world, fly


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=sorted(PROFILES), default='conservative')
    parser.add_argument('--scenario', choices=SCENARIOS, default='demo')
    parser.add_argument('--dt', type=float, default=.005, help='integration step in seconds')
    parser.add_argument('--duration', type=float, default=90., help='maximum simulated seconds')
    parser.add_argument('--resolution', type=float, default=.25, help='A* grid resolution in metres')
    parser.add_argument('--json', action='store_true', help='machine-readable output')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        world, start, goal = demo_world(args.scenario)
        result = fly(world, start, goal, args.dt, args.duration, args.resolution, args.mode)
    except ValueError as exc:
        parser.exit(2, f'error: {exc}\n')

    report = {
        'scenario': args.scenario,
        'mode': result.profile,
        'arrived': result.arrived,
        'collision': result.collision,
        'simulated_seconds': round(float(result.times[-1]), 3),
        'final_error_m': round(result.final_error, 4),
        'max_tracking_error_m': round(result.max_tracking_error, 4),
        'max_speed_mps': round(result.max_speed, 3),
        'max_tilt_degrees': round(result.max_tilt_degrees, 2),
        'min_point_clearance_m': round(result.min_point_clearance, 4),
        'waypoints': [[round(float(value), 4) for value in point] for point in result.path],
    }
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        status = 'ARRIVED' if result.arrived else ('COLLISION' if result.collision else 'TIMEOUT')
        print(f'{status} | {args.scenario} | {result.profile} mode')
        print(f'Simulated time: {report["simulated_seconds"]} s')
        print(f'Final error: {report["final_error_m"]} m')
        print(f'Max tracking error: {report["max_tracking_error_m"]} m')
        print(f'Peak speed: {report["max_speed_mps"]} m/s')
        print(f'Peak tilt: {report["max_tilt_degrees"]} degrees')
        print(f'Minimum point clearance: {report["min_point_clearance_m"]} m')
    return 0 if result.arrived and not result.collision else 1


if __name__ == '__main__':
    sys.exit(main())
