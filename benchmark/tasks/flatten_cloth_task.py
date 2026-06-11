from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import jax.random as jrd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from environments import ClothEnv
from .task import Task


class FlattenClothTask(Task):
    def __init__(self, seed):
        key = jrd.key(seed)

        def rand_uniform(minval, maxval):
            return float(jrd.uniform(key, minval=minval, maxval=maxval))

        self._env = ClothEnv.from_regular_grid(
            time_step=0.02,
            cloth_width=0.3,
            num_div_side=9,
            thickness=1e-3,
            youngs_modulus=rand_uniform(0.8e4, 1.6e4),
            possion_ratio=0.3,
            mass_density=20,
            num_floating_grippers=2,
            grip_stiffness=1000,
            gripper_radius=0.01,
            contact_smoothing=3e-3,
            ground_friction_coeff=rand_uniform(0.3, 0.8),
        )

        env = self._env
        x_node = jnp.load(Path(__file__).resolve().parent / "wrinkled_cloth.npy")
        x_grip = jnp.array([
            [-0.07, 0.06, 0.5e-3],
            [ 0.08, -0.07, 0.5e-3],
        ])
        self._initial_state = env.state(x_node=x_node, x_grip=x_grip)

        x_node = env.params.x_node_rest
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
        server.scene.add_grid("ground", position=(0, 0, -0.01))
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
