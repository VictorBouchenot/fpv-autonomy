"""Render a reproducible static 3D preview from a verified simulated mission."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from fpv_sim.mission import PROFILES, SCENARIOS, demo_world, fly


def draw_box(ax, box):
    x, y, z = box.lower
    X, Y, Z = box.upper
    vertices = np.array([
        [x, y, z], [X, y, z], [X, Y, z], [x, Y, z],
        [x, y, Z], [X, y, Z], [X, Y, Z], [x, Y, Z],
    ])
    face_indices = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
    ]
    faces = [[vertices[i] for i in face] for face in face_indices]
    ax.add_collection3d(Poly3DCollection(
        faces, facecolors=(.91, .42, .25, .12), edgecolors=(1, .65, .5, .65)
    ))


def render(output='artifacts/mission-preview.png', mode='dynamic', scenario='demo'):
    world, start, goal = demo_world(scenario)
    result = fly(world, start, goal, profile=mode)
    if not result.arrived or result.collision:
        raise RuntimeError('Preview mission failed')
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8), layout='constrained')
    ax = fig.add_subplot(projection='3d')
    for box in world.obstacles:
        draw_box(ax, box)
    ax.plot(*result.path.T, 'o--', color='#ffd166', label='A* + checked shortcuts', linewidth=2)
    ax.plot(*result.positions.T, color='#65dfd0', label='Physics + PID actual flight', linewidth=2)
    ax.scatter(*start, s=80, color='#80ed99', label='Start')
    ax.scatter(*goal, s=100, marker='*', color='white', label='Goal')
    ax.set(xlim=(0, 6), ylim=(0, 5), zlim=(0, 3.5), xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
    ax.set_box_aspect((6, 5, 3.5))
    ax.view_init(elev=26, azim=-58)
    ax.set_title(
        f'FPV autonomy | {result.profile} mode\n'
        f'{result.times[-1]:.2f} s to goal | final error {100*result.final_error:.1f} cm | no collision in this run',
        pad=20,
    )
    ax.legend(loc='upper left', fontsize=9)
    fig.text(
        .5, .01,
        'Educational model, not Aquila 20 calibration. Orange = obstacles. Use animate.py for the full diagnostic video.',
        ha='center', fontsize=9,
    )
    output = Path(output)
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/mission-preview.png')
    parser.add_argument('--mode', choices=sorted(PROFILES), default='dynamic')
    parser.add_argument('--scenario', choices=SCENARIOS, default='demo')
    args = parser.parse_args(argv)
    try:
        render(args.output, args.mode, args.scenario)
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f'error: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
