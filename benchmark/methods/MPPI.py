from functools import partial
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp

sys.path.append(str(Path(__file__).resolve().parents[2]))
from benchmark.tasks import *


class MPPIPlanner:
    def __init__(
        self,
        dynamics_fn,
        running_cost_fn,
        terminal_cost_fn,
        horizon: int,
        num_samples: int,
        u_ref,
        lambda_: float = 1.0,
        noise_sigma: float = 1.0,
        gamma: float = 1.0,
    ):
        self.dynamics_fn = dynamics_fn
        self.running_cost_fn = running_cost_fn
        self.terminal_cost_fn = terminal_cost_fn
        self.horizon = horizon
        self.num_samples = num_samples
        self.u_ref = u_ref
        self.lambda_ = lambda_
        self.noise_sigma = noise_sigma
        self.gamma = gamma
        self.U = jnp.tile(u_ref[None, :], (horizon, 1))

    @partial(jax.jit, static_argnums=0)
    def _rollout(self, x0, U, batch_noise):
        def rollout_one(noise):
            def step(x, inp):
                u_nom, eps = inp
                u = u_nom + eps
                x_next = self.dynamics_fn(x, u)
                control_cost = (
                    self.gamma
                    * self.lambda_
                    * jnp.sum(u_nom * eps / self.noise_sigma**2)
                )
                cost = self.running_cost_fn(x, u) + control_cost
                return x_next, cost

            x_final, costs = jax.lax.scan(step, x0, (U, noise))
            return jnp.sum(costs) + self.terminal_cost_fn(x_final)

        costs = jax.vmap(rollout_one)(batch_noise)

        weights = jnp.exp(-(costs - jnp.min(costs)) / self.lambda_)
        weights /= jnp.sum(weights) + 1e-8

        U_new = U + jnp.sum(weights[:, None, None] * batch_noise, axis=0)
        U_shifted = jnp.concatenate([U_new[1:], self.u_ref[None, :]], axis=0)
        return U_new[0], U_shifted, costs

    def act(self, x0, key):
        noise = jax.random.normal(key, shape=(self.num_samples, self.horizon, self.u_ref.size)) * self.noise_sigma
        action, self.U, costs = self._rollout(x0, self.U, noise)
        return action, costs


def run_MPPI(task, visualizer):
    nx = task.initial_state().size
    N = 20 if nx < 100 else 10

    def running_cost(x, u):
        constr = task.constrtaints(x, u)
        return task.cost(x, u) + jnp.sum(jnp.maximum(constr, 0.0)**2) * 10

    def terminal_cost(x):
        u = task.initial_control()
        return running_cost(x, u)

    def dynamics(x, u):
        return task.step(x, u)

    planner = MPPIPlanner(
        dynamics_fn=dynamics,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        horizon=N,
        num_samples=1024,
        u_ref=task.initial_control(),
        lambda_=0.1,
        noise_sigma=0.1,
        gamma=1e-2,
    )

    key   = jax.random.PRNGKey(0)
    state = task.initial_state()

    if visualizer is not None:
        task.visualize(visualizer, state)

    execution_time = 0.0
    for k in range(task.horizon()):
        key, subkey = jax.random.split(key)

        tic = time.time()
        control, costs = planner.act(state, subkey)
        elapsed = time.time() - tic

        print(f"MPPI step {k:3d} | time: {elapsed * 1e3:6.2f} ms | best cost: {float(jnp.min(costs)):.4f}")

        if k > 0:
            execution_time += elapsed

        if jnp.isnan(control).any() or (jnp.abs(control) > 1e2).any():
            control = task.initial_control()

        task.log(state, control)
        state = task.step(state, control)

        if visualizer is not None:
            task.visualize(visualizer, state)

    task.log(state)
    return execution_time
