from pathlib import Path
import sys
import jax.numpy as jnp
import jax.random as jrd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from environments import RopeEnv
from .task import Task


class DragRopeTask(Task):
    def __init__(self, seed):
        key = jrd.key(seed)

        self._env = RopeEnv(
            time_step=0.02,
            num_segments=20,
            rope_length=0.3,
            rope_diameter=0.004,
            youngs_modulus=1e4,
            mass_density=300,
            num_floating_grippers=1,
            grip_stiffness=800,
            gripper_radius=0.005,
            contact_smoothing=1e-3,
        )

        env = self._env
        x_grip = jnp.array([[0.02, 0.0, 0.02]])
        self._initial_state = env.state(x_grip=x_grip)

        l = jnp.arange(env.params.num_nodes)[::-1] * env.params.segment_length
        x_node = jnp.stack((
            jnp.ones_like(l) * jrd.uniform(key, minval=-0.21, maxval=-0.15),
            l,
            jnp.full_like(l, 0.002),
        ), axis=1)
        self._goal_state = env.state(x_node=x_node)

        vmax = 0.2
        self._u_max = jnp.array([vmax, vmax, vmax, 10.0])
        self._u_min = -self._u_max

        self._initial_control = jnp.array([0, 0, -vmax, 1.0])

        self._obstacle_center = jnp.array([-0.1, 0.1])
        self._obstacle_radius = 0.1

    def initial_state(self):
        return self._initial_state

    def initial_control(self):
         return self._initial_control

    def cost(self, state, control):
        u_ref = self._env.control()
        state_err = state - self._goal_state
        control_err = control - u_ref
        return (
            1.0 * jnp.sum(state_err[:-6]**2)
            + 9.0 * jnp.sum(state_err[:3]**2)
            + 0.1 * jnp.sum(control_err[:-2]**2)
        )

    def constrtaints(self, state, control):
        control_constraints = jnp.concatenate((
             control - self._u_max,
             self._u_min - control,
        ))

        x_node, _, _ = self._env.unpack_state(state)
        z_constraints = -x_node[:, 2] - 0.05

        obstacle_constraints = self._obstacle_radius**2 - jnp.sum((x_node[:, :2] - self._obstacle_center)**2, axis=1)

        return jnp.concatenate([control_constraints, z_constraints, obstacle_constraints])

    def step(self, state, control, clip=True):
        self._start_timer()
        if clip:
            control = jnp.clip(control, self._u_min, self._u_max)
        state = self._env.step(state, control)
        self._stop_timer()
        return state

    def visualize(self, server, state):
        server.scene.add_grid("ground")
        server.scene.add_cylinder(
            name="obstacle",
            radius=self._obstacle_radius,
            height=0.03,
            color=(1.0, 0.4, 0.0),
            position=tuple(self._obstacle_center) + (0.0,)
        )
        server.scene.add_icosphere(
            name="goal",
            radius=self._env.params.rope_radius * 2,
            color=(0.0, 1.0, 1.0),
            opacity=0.8,
            position=tuple(self._goal_state[:3].tolist()),
        )
        self._env.visualize(server, state)

    def horizon(self):
        return 500

    def eval_metrics(self):
        metrics, traj = super().eval_metrics()

        final_state = traj[-1][0]
        final_state_error = final_state - self._goal_state
        final_state_error = float(jnp.linalg.norm(final_state_error[:3]))

        metrics["final_state_error"] = final_state_error
        return metrics, traj
