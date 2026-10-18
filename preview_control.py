"""Generate a deterministic first-milestone plot, not a real drone flight."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from fpv_sim.physics import Quadrotor, State, rotation_vector
from fpv_sim.control import FlightController

q = Quadrotor(state=State(position=np.array([0.5, -0.3, 0.6]), rotation=rotation_vector(np.array([0.2, -0.15, 0.]))))
c = FlightController(q)
dt = 0.005
times, positions, tilts, motor_log = [], [], [], []
for i in range(2400):
    t = i * dt
    motors = c.command([0, 0, 1], dt)
    disturbance = [0.08, -0.04, 0] if 3 <= t < 4 else [0, 0, 0]
    s = q.step(motors, dt, disturbance)
    times.append(t + dt)
    positions.append(s.position)
    tilts.append(np.degrees(np.arccos(np.clip(s.rotation[2, 2], -1, 1))))
    motor_log.append(motors)
positions = np.array(positions)
plt.style.use('dark_background')
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
fig.suptitle('FPV autonomy | First physics + controller preview', fontsize=18)
for j, label in enumerate(['x', 'y', 'z']):
    axes[0].plot(times, positions[:, j], label=label + ' actual', linewidth=2)
axes[0].axhline(1, color='white', linestyle=':', label='z target = 1 m')
axes[0].axhline(0, color='gray', linestyle=':', label='x/y target = 0 m')
axes[0].set_ylabel('Position (m)'); axes[0].legend(ncol=5, fontsize=9)
axes[1].plot(times, tilts, color='#e5b567', linewidth=2)
axes[1].set_ylabel('Tilt (degrees)')
for j in range(4): axes[2].plot(times, np.array(motor_log)[:, j], label=f'Motor {j+1}', alpha=0.8)
axes[2].set_ylabel('Motor thrust (N)'); axes[2].set_xlabel('Simulation time (s)'); axes[2].legend(ncol=4)
for ax in axes:
    ax.axvspan(3, 4, alpha=0.15, color='red')
    ax.grid(alpha=0.15)
axes[1].text(3.05, max(tilts)*0.85, 'External force\n3-4 s', color='#ff9999')
fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
fig.text(0.5, 0.008, 'Generic educational model, not calibrated Aquila 20 physics. No real-flight control.', ha='center', fontsize=9)
Path('artifacts').mkdir(exist_ok=True)
fig.savefig('artifacts/controller-preview.png', dpi=150)
print('Final position error:', np.linalg.norm(positions[-1] - [0, 0, 1]))
