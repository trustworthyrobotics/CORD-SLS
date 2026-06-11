import time
import jax
import jax.numpy as jnp

class Task:
    def _start_timer_impl(self, _):
        self._tic = time.time()

    def _start_timer(self):
        jax.debug.callback(self._start_timer_impl, None)

    def _stop_timer_impl(self, _):
        dt = time.time() - self._tic
        if not hasattr(self, "_dynamics_eval_count"):
            self._dynamics_eval_count = 0
            self._dynamics_eval_time = 0.0
        self._dynamics_eval_count += 1
        self._dynamics_eval_time += dt

    def _stop_timer(self):
        jax.debug.callback(self._stop_timer_impl, None)

    def log(self, state, control=None):
        if not hasattr(self, "_traj"):
            self._traj = []
        self._traj.append((state, control))

    def eval_metrics(self):
        violation_count = 0
        for state, control in self._traj:
            if control is None:
                control = self.initial_control()
            if jnp.any(self.constrtaints(state, control) > 0):
                violation_count += 1

        metrics =  {
            "dynamics_eval_count": self._dynamics_eval_count,
            "dynamics_eval_time": self._dynamics_eval_time,
            "num_steps": len(self._traj),
            "num_constraint_violations_steps": violation_count,
        }

        return metrics, self._traj
