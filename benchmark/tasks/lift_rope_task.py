from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import jax.random as jrd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from environments import RopeEnv
from .task import Task


class LiftRopeTask(Task):
    def __init__(self, seed):
        key = jrd.key(seed)

        self._env = RopeEnv(
            time_step=0.02,
            num_segments=20,
            rope_length=0.3,
            rope_diameter=0.004,
            youngs_modulus=2e3,
            mass_density=100,
            num_floating_grippers=2,
            grip_stiffness=800,
            gripper_radius=0.015,
            ground_friction_coeff=0.8,
            contact_smoothing=1e-3,
        )

        env = self._env
        x_node = jnp.load(Path(__file__).resolve().parent / "curved_rope.npy")
        x_grip = jnp.array([
            [0.01, 0.03, 0.05],
            [0.14, 0.00, 0.05],
        ])
        self._initial_state = env.state(x_node=x_node, x_grip=x_grip)

        l = jnp.arange(env.params.num_nodes) * env.params.segment_length
        x_node = jnp.stack((
            l,
            jnp.zeros_like(l),
            jnp.full_like(l, jrd.uniform(key, minval=0.03, maxval=0.07))
        ), axis=1)
        self._goal_state = env.state(x_node=x_node)

        vmax = 0.2
        u_max = jnp.array([vmax, vmax, vmax, 10.0])
        self._u_max = jnp.repeat(u_max, env.params.num_floating_grippers)
        self._u_min = -self._u_max

        self._initial_control = jnp.array([0, 0, -vmax, 0, 0, -vmax, 1.0, 1.0])

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
            + 0.1 * jnp.sum(control_err[:-2]**2)
        )

    def constrtaints(self, state, control):
        control_constraints = jnp.concatenate((
             control - self._u_max,
             self._u_min - control,
        ))
        return control_constraints

    def step(self, state, control, clip=True):
        self._start_timer()
        if clip:
            control = jnp.clip(control, self._u_min, self._u_max)
        state = self._env.step(state, control)
        self._stop_timer()
        return state

    def visualize(self, server, state):
        server.scene.add_grid("ground")
        self._env.visualize(server, state)

    def horizon(self):
        return 200

    def eval_metrics(self):
        metrics, traj = super().eval_metrics()

        final_state = traj[-1][0]
        final_state_error = final_state - self._goal_state
        final_state_error = float(jnp.linalg.norm(final_state_error[:-6]))

        metrics["final_state_error"] = final_state_error
        return metrics, traj
