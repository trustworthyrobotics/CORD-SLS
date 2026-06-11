from pathlib import Path
import sys
import time

import gymnasium as gym
from gymnasium import spaces

import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import numpy as np

import torch
from torch import nn as tnn
from torch import optim

import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(str(Path(__file__).resolve().parents[2]))
from benchmark.tasks import *


# ---------------------------------------------------------------------------
# Gymnasium wrapper around the task
# ---------------------------------------------------------------------------
class TaskEnv(gym.Env):
    """
    Wraps a benchmark task as a Gymnasium environment for SB3.

    Important:
    - SB3/PyTorch uses float32
    - We force all simulator inputs to jnp.float32
    - We disable JAX x64 globally
    """

    metadata = {"render_modes": []}

    def __init__(self, task):
        super().__init__()

        self.task = task

        self._horizon = int(task.horizon())
        self._t = 0

        x0 = np.asarray(task.initial_state(), dtype=np.float32)
        u0 = np.asarray(task.initial_control(), dtype=np.float32)

        self._x0 = x0

        nx = x0.size
        nu = u0.size

        # ------------------------------------------------------------------
        # Observation space
        # SB3 prefers finite bounds
        # ------------------------------------------------------------------
        self.observation_space = spaces.Box(
            low=np.full(nx, -1e10, dtype=np.float32),
            high=np.full(nx, 1e10, dtype=np.float32),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Action space
        # Must be finite for PPO
        # ------------------------------------------------------------------
        self.action_space = spaces.Box(
            low=np.asarray(task._u_min, dtype=np.float32),
            high=np.asarray(task._u_max, dtype=np.float32),
            dtype=np.float32,
        )

        self._state = self._x0.copy()

    # ----------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        noise = (
            self.np_random.standard_normal(self._x0.shape)
            .astype(np.float32)
            * 1e-3
        )

        self._state = (self._x0 + noise).astype(np.float32)

        self._t = 0

        return self._state.copy(), {}

    # ----------------------------------------------------------------------
    def step(self, action):
        """
        Convert everything to JAX float32 before entering simulator.
        """

        x = jnp.asarray(self._state, dtype=jnp.float32)
        u = jnp.asarray(action, dtype=jnp.float32)

        # ------------------------------------------------------------------
        # Cost
        # ------------------------------------------------------------------
        constr = jnp.asarray(
            self.task.constrtaints(x, u),
            dtype=jnp.float32,
        )

        cost = (
            jnp.asarray(self.task.cost(x, u), dtype=jnp.float32)
            + jnp.sum(jnp.maximum(constr, 0.0) ** 2) * 1e-3
        )

        reward = -float(cost)

        # ------------------------------------------------------------------
        # Simulator step
        # ------------------------------------------------------------------
        next_state = self.task.step(x, u)

        # Force float32
        next_state = jnp.asarray(next_state, dtype=jnp.float32)

        # Convert back to NumPy for SB3
        self._state = np.asarray(next_state, dtype=np.float32)

        self._t += 1

        terminated = False
        truncated = self._t >= self._horizon

        return (
            self._state.copy(),
            reward,
            terminated,
            truncated,
            {},
        )

    # ----------------------------------------------------------------------
    def render(self):
        pass


# ---------------------------------------------------------------------------
# Optional logging callback
# ---------------------------------------------------------------------------
class EpochLogCallback(BaseCallback):
    def __init__(self, log_interval: int = 50, verbose: int = 0, visualizer=None, task=None):
        super().__init__(verbose)

        self.log_interval = log_interval
        self.rollout_count = 0
        self.visualizer = visualizer
        self.task = task

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1

        if len(self.model.ep_info_buffer) > 0:
            mean_r = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            print(f"[rollout {self.rollout_count}] mean_reward={mean_r:.6f}")

        if not(self.rollout_count % self.log_interval == 0 or self.rollout_count == 1):
            return

        if self.visualizer is None:
            return

        task = self.task
        state = np.asarray(task.initial_state(), dtype=np.float32)

        task.visualize(self.visualizer, state)

        for _ in range(task.horizon()):
            action, _ = self.model.predict(state, deterministic=True)

            state = np.asarray(
                task.step(
                    jnp.asarray(state, dtype=jnp.float32),
                    jnp.asarray(action, dtype=jnp.float32),
                ),
                dtype=np.float32,
            )

            task.visualize(self.visualizer, state)

        task_name = task.__class__.__name__.lower()

        ckpt_path = (
            Path(__file__).resolve().parent.parent
            / "results"
            / f"PPO_{task_name}_epoch{self.rollout_count}"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(ckpt_path))


# ---------------------------------------------------------------------------
# Train PPO
# ---------------------------------------------------------------------------
def train_PPO(
    task,
    visualizer,
    demos,
    hidden_dims=(64, 64),
    n_epochs=10000,
    batch_size=8,
    learning_rate=1e-4,
    seed=0,
):
    env = TaskEnv(task)

    # ----------------------------------------------------------------------
    # PPO network
    # ----------------------------------------------------------------------
    policy_kwargs = dict(
        net_arch=list(hidden_dims),
        activation_fn=sb3.common.torch_layers.nn.Tanh,
        log_std_init=-3.0,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=task.horizon(),
        batch_size=batch_size,
        n_epochs=10,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.05,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        device="cpu",   # SB3 MLP PPO is usually faster on CPU
    )

    # ----------------------------------------------------------------------
    # Optional BC warm-start
    # ----------------------------------------------------------------------
    demos = [
        (x, u)
        for x, u in demos
        if x is not None and u is not None
    ]

    if demos:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # NOTE: SB3 model must also be on same device
        model.policy.to(device)

        xs = np.stack([
            np.asarray(x, dtype=np.float32)
            for x, _ in demos
        ])

        us = np.stack([
            np.asarray(u, dtype=np.float32)
            for _, u in demos
        ])

        xs_t = torch.as_tensor(xs, dtype=torch.float32, device=device)
        us_t = torch.as_tensor(us, dtype=torch.float32, device=device)

        bc_opt = optim.Adam(model.policy.parameters(), lr=1e-2)

        for i in range(10000):

            # IMPORTANT: SB3 feature extractor expects torch tensor on same device
            features = model.policy.extract_features(
                xs_t,
                model.policy.features_extractor,
            )

            latent_pi, _ = model.policy.mlp_extractor(features)

            u_pred = model.policy.action_net(latent_pi)

            loss = tnn.functional.mse_loss(u_pred, us_t)

            bc_opt.zero_grad()
            loss.backward()
            bc_opt.step()

            if (i + 1) % 500 == 0:
                print(f"[BC] iter={i+1:4d} | mse={loss.item():.6e}")

        print(
            f"[PPO] BC warm-start done | "
            f"demos={len(demos)} | "
            f"final_mse={loss.item():.6e}"
        )
        model.policy.to("cpu")

    # ----------------------------------------------------------------------
    # Main training
    # ----------------------------------------------------------------------
    total_timesteps = (
        n_epochs
        * task.horizon()
    )

    print(
        f"[PPO] "
        f"nx={env.observation_space.shape[0]} | "
        f"nu={env.action_space.shape[0]} | "
        f"horizon={task.horizon()} | "
        f"timesteps={total_timesteps}"
    )

    callback = EpochLogCallback(log_interval=50, visualizer=visualizer, task=task)

    tic = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
    )

    execution_time = time.time() - tic
    return execution_time


# ---------------------------------------------------------------------------
# Run saved PPO checkpoint
# ---------------------------------------------------------------------------
def run_PPO(task, visualizer, checkpoint):

    ckpt_path = (
        Path(__file__).resolve().parent.parent
        / "results"
        / checkpoint
    )

    model = PPO.load(
        str(ckpt_path),
        device="cpu",
    )

    state = np.asarray(
        task.initial_state(),
        dtype=np.float32,
    )

    if visualizer is not None:
        task.visualize(visualizer, state)

    execution_time = 0.0

    for _ in range(task.horizon()):

        tic = time.time()

        action, _ = model.predict(
            state,
            deterministic=True,
        )

        execution_time += time.time() - tic

        task.log(state, action)

        state = np.asarray(
            task.step(
                jnp.asarray(state, dtype=jnp.float32),
                jnp.asarray(action, dtype=jnp.float32),
            ),
            dtype=np.float32,
        )

        if visualizer is not None:
            task.visualize(visualizer, state)

    task.log(state)

    return execution_time
