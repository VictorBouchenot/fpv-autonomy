"""Render a verified simulated flight, with separate conservative/dynamic modes."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
from fpv_sim.mission import PROFILES, SCENARIOS, demo_world, fly, get_profile, point_clearance
from preview_mission import draw_box


def render(output='artifacts/flight.mp4', fps=15, mode='conservative', playback=None, scenario='demo'):
    if not isinstance(fps, int) or not 1 <= fps <= 60:
        raise ValueError('fps must be an integer from 1 to 60')
    profile = get_profile(mode)
    if playback is None:
        playback = 4. if profile.stop_at_corners else 1.
    if not np.isfinite(playback) or not 0 < playback <= 8:
        raise ValueError('playback must be in (0, 8]')
    output = Path(output)
    if output.suffix.lower() not in {'.mp4', '.gif'}:
        raise ValueError('output must end in .mp4 or .gif')

    world, start, goal = demo_world(scenario)
    result = fly(world, start, goal, profile=profile)
    if not result.arrived or result.collision:
        raise RuntimeError('Mission failed; refusing to render a success animation')

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 7), facecolor='#0b1220')
    grid = fig.add_gridspec(3, 4, left=.04, right=.96, top=.85, bottom=.10, wspace=.5, hspace=.5)
    ax = fig.add_subplot(grid[:, :3], projection='3d', facecolor='#0b1220')
    tilt_ax = fig.add_subplot(grid[0, 3], facecolor='#0b1220')
    error_ax = fig.add_subplot(grid[1, 3], facecolor='#0b1220')
    motor_ax = fig.add_subplot(grid[2, 3], facecolor='#0b1220')
    fig.text(.05, .94, 'FPV AUTONOMY', fontsize=23, weight='bold', color='#65dfd0')
    fig.text(
        .05, .895,
        f'{profile.name.upper()} mode  /  A* route planning  /  rigid-body physics  /  PID + attitude PD',
        fontsize=11,
    )
    status = fig.text(.70, .94, '', fontsize=11)
    for box in world.obstacles:
        draw_box(ax, box)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((.08, .12, .18, 1))
    ax.plot(*result.path.T, '--', color='#ffd166', alpha=.7, label='Planned route')
    trail, = ax.plot([], [], [], color='#65dfd0', linewidth=2, label='Actual flight')
    arms = [ax.plot([], [], [], color=color, lw=4)[0] for color in ['#ffffff', '#ff8095']]
    rotor, = ax.plot([], [], [], linestyle='', marker='o', markersize=6, color='#65dfd0')
    ax.scatter(*start, color='#80ed99', s=40)
    ax.scatter(*goal, color='white', marker='*', s=100)
    ax.set(xlim=(0, 6), ylim=(0, 5), zlim=(0, 3.5), xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
    ax.set_box_aspect((6, 5, 3.5))
    ax.view_init(elev=29, azim=-57)
    ax.legend(loc='upper left', fontsize=9)

    tilt = np.degrees(np.arccos(np.clip(result.rotations[:, 2, 2], -1, 1)))
    clearance = np.array([point_clearance(world, point) for point in result.positions])
    speed_steps = np.linalg.norm(np.diff(result.positions, axis=0) / np.diff(result.times)[:, None], axis=1)
    cursors = []
    for plot, data, title, color in [
        (tilt_ax, tilt, 'Tilt (degrees)', '#ffd166'),
        (error_ax, clearance, 'Point clearance (m)', '#65dfd0'),
    ]:
        plot.plot(result.times, data, color=color, lw=1)
        plot.set_title(title, fontsize=10)
        plot.set_xlim(0, result.times[-1])
        plot.grid(alpha=.15)
        if title.startswith('Point clearance'):
            plot.axhline(world.clearance, color='#ff8095', linestyle=':', lw=1, label='collision radius')
            plot.legend(fontsize=7, loc='lower right')
        cursors.append(plot.axvline(0, color='white', alpha=.7, lw=1))
    bars = motor_ax.bar(range(1, 5), [0]*4, color=['#65dfd0', '#ffd166', '#ff8095', '#a8b6ff'])
    motor_ax.set(ylim=(0, .6), xticks=range(1, 5))
    motor_ax.set_title('Motor thrust (N)', fontsize=10)
    fig.text(
        .05, .025,
        f'SIMULATION ONLY  |  Generic parameters, not calibrated Aquila 20 data  |  Playback {playback:g}x  |  Same physics in both modes',
        fontsize=9, color='#aab6c7',
    )

    # Display geometry enlarged for readability; state/attitude are unmodified.
    local = np.array([[.16, .16, 0], [-.16, .16, 0], [-.16, -.16, 0], [.16, -.16, 0]])
    frame_times = np.arange(0, result.times[-1], playback / fps)
    indices = np.r_[np.searchsorted(result.times, frame_times), len(result.times)-1]

    def update(frame):
        i = indices[frame]
        p = result.positions[i]
        r = result.rotations[i]
        trail.set_data_3d(*result.positions[:i+1:10].T)
        vertices = local @ r.T + p
        for line, pair in zip(arms, [[0, 2], [1, 3]]):
            line.set_data_3d(*vertices[pair].T)
        rotor.set_data_3d(*vertices.T)
        for cursor in cursors:
            cursor.set_xdata([result.times[i]]*2)
        for bar, value in zip(bars, result.motors[i]):
            bar.set_height(value)
        final = frame == len(indices)-1
        speed = 0. if i == 0 else speed_steps[min(i-1, len(speed_steps)-1)]
        status.set_text(
            ('GOAL REACHED' if final else f'{profile.name.upper()} FLIGHT')
            + f'\nt = {result.times[i]:.1f} s  |  v = {speed:.1f} m/s'
            + f'\nclearance = {clearance[i]:.2f} m  |  {np.linalg.norm(p-goal):.2f} m to goal'
        )
        return []

    animation = FuncAnimation(fig, update, frames=len(indices), interval=1000/fps, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps) if output.suffix.lower() == '.gif' else FFMpegWriter(
        fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p']
    )
    animation.save(output, writer=writer, dpi=100)
    for fraction, label in [(0, 'start'), (.5, 'middle'), (1, 'finish')]:
        update(round((len(indices)-1)*fraction))
        fig.savefig(output.parent / f'{output.stem}-{label}.png', dpi=120)
    plt.close(fig)
    print(output)
    print(
        f'{profile.name}: {result.times[-1]:.2f}s simulated, '
        f'{result.max_speed:.2f} m/s peak, {result.max_tilt_degrees:.1f} deg peak tilt, '
        f'{result.min_point_clearance:.3f} m minimum point clearance'
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/flight.mp4')
    parser.add_argument('--mode', choices=sorted(PROFILES), default='conservative')
    parser.add_argument('--scenario', choices=SCENARIOS, default='demo')
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--playback', type=float, help='video speed multiplier (defaults: 4x conservative, 1x dynamic)')
    args = parser.parse_args(argv)
    try:
        render(args.output, args.fps, args.mode, args.playback, args.scenario)
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f'error: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
