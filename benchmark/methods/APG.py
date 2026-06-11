from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import linen as nn
from typing import Sequence

sys.path.append(str(Path(__file__).resolve().parents[2]))
from benchmark.tasks import *


# ---------------------------------------------------------------------------
# MLP policy: nx -> [hidden...] -> nu
# ---------------------------------------------------------------------------
class MLPPolicy(nn.Module):
    hidden_dims: Sequence[int]
    nu: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.tanh(x)
        return nn.Dense(self.nu)(x)


# ---------------------------------------------------------------------------
# Single trajectory rollout (no vmap here — vmap is applied outside)
# ---------------------------------------------------------------------------
def _running_cost(x, u, task):
    constr = task.constrtaints(x, u)
    return task.cost(x, u) + jnp.sum(jnp.maximum(constr, 0.0) ** 2) * 1e-3


def rollout_trajectory(params, apply_fn, x0, task, horizon):
    def step(x, _):
        u = apply_fn({"params": params}, x)
        c = _running_cost(x, u, task)
        x_next = task.step(x, u)
        return x_next, (x, u, c)          # carry, (state, action, cost)

    _, (states, actions, costs) = jax.lax.scan(step, x0, None, length=horizon)
    return states, actions, costs


def batched_cost(params, apply_fn, x0_batch, task, horizon):
    states, actions, costs = jax.vmap(
        lambda x0: rollout_trajectory(params, apply_fn, x0, task, horizon)
    )(x0_batch)
    return jnp.mean(costs)


def pretrain_policy(params, apply_fn, demos, n_steps=10000, lr=1e-2):
    demos = [(x, u) for x, u in demos if x is not None and u is not None]
    if not demos:
        return params

    xs = jnp.stack([jnp.asarray(x) for x, _ in demos])
    us = jnp.stack([jnp.asarray(u) for _, u in demos])

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        def mse(params):
            u_pred = jax.vmap(lambda x: apply_fn({"params": params}, x))(xs)
            return jnp.mean(jnp.sum((u_pred - us) ** 2, axis=-1))
        loss, grads = jax.value_and_grad(mse)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    for _ in range(n_steps):
        params, opt_state, loss = step(params, opt_state)

    print(f"[APG] pretrain done  |  {len(demos)} demos  |  MSE = {float(loss):.2e}")
    return params


def train_APG(task, visualizer, demos,
            hidden_dims=(64, 64),
            n_epochs=10000,
            batch_size=8,
            learning_rate=1e-4,
            seed=0):

    nx = task.initial_state().size
    nu = task.initial_control().size
    horizon = task.horizon()

    policy = MLPPolicy(hidden_dims=list(hidden_dims), nu=nu)
    key = jax.random.PRNGKey(seed)
    key, init_key, batch_key = jax.random.split(key, 3)
    dummy_x = jnp.zeros(nx)
    params = policy.init(init_key, dummy_x)["params"]

    x0 = task.initial_state()

    params = pretrain_policy(params, policy.apply, demos)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, x0_batch):
        cost, grads = jax.value_and_grad(batched_cost)(
            params, policy.apply, x0_batch, task, horizon
        )

        def do_update(_):
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state

        def skip_update(_):
            return params, opt_state

        grads_finite = jax.tree_util.tree_reduce(
            lambda acc, g: acc & jnp.all(jnp.isfinite(g)),
            grads,
            initializer=jnp.bool_(True),
        )

        new_params, new_opt_state = jax.lax.cond(
            jnp.isnan(cost) | ~grads_finite,
            skip_update,
            do_update,
            operand=None,
        )
        return new_params, new_opt_state, cost

    @jax.jit
    def get_trajectory(params, x0):
        return rollout_trajectory(params, policy.apply, x0, task, horizon)

    if visualizer is not None:
        task.visualize(visualizer, x0)

    # ---- Training loop ----
    execution_time = 0.0
    print(f"[APG] nx={nx}, nu={nu}, horizon={horizon}, "
          f"batch={batch_size}, epochs={n_epochs}")

    for epoch in range(n_epochs):

        # Sample a batch of slightly perturbed initial states for robustness.
        batch_key, subkey = jax.random.split(batch_key)
        noise = jax.random.normal(subkey, (batch_size, nx)) * 0.001
        x0_batch = x0[None, :] + noise       # shape [B, nx]

        tic = time.time()
        params, opt_state, cost = update(params, opt_state, x0_batch)
        execution_time += time.time() - tic

        print(f"  epoch {epoch+1:4d}/{n_epochs}  |  cost = {float(cost):.6f}")

        if ((epoch + 1) % 50 == 0 or epoch == 0) and visualizer is not None:
            states, _, _ = get_trajectory(params, x0_batch[0])
            for t in range(horizon):
                task.visualize(visualizer, states[t])

            path = Path(__file__).resolve().parent.parent / "results" / f"APG_{task.__class__.__name__.lower()}_epoch{epoch+1}"
            path.parent.mkdir(parents=True, exist_ok=True)
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(path, params, force=True)


    # ---- Final state for logging ----
    states, _, _ = get_trajectory(params, x0)
    task.log(states[-1])
    return execution_time


def run_APG(task, visualizer, checkpoint):
    nx = task.initial_state().size
    nu = task.initial_control().size

    policy = MLPPolicy(hidden_dims=[64, 64], nu=nu)

    path = Path(__file__).resolve().parent.parent / "results" / checkpoint
    checkpointer = ocp.StandardCheckpointer()
    params = checkpointer.restore(path)

    state = task.initial_state()

    if visualizer is not None:
        task.visualize(visualizer, state)

    execution_time = 0.0
    for _ in range(task.horizon()):
        tic = time.time()
        control =  policy.apply({"params": params}, state)
        execution_time += time.time() - tic

        task.log(state, control)
        state = task.step(state, control)

        if visualizer is not None:
            task.visualize(visualizer, state)

    task.log(state)
    return execution_time
