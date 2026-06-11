from pathlib import Path
import sys
import time
import jax.numpy as jnp

sys.path.append(str(Path(__file__).resolve().parents[2]))
from planners import gpu_sls
from benchmark.tasks import *


def run_SLS(task, visualizer):
    def cost(W, reference, x, u, t):
        return task.cost(x, u)

    def dynamics(x, u, t, parameter):
        return task.step(x, u, clip=False)

    def constraints(x, u, t):
        return task.constrtaints(x, u)


    state0 = task.initial_state()
    control0 = task.initial_control()
    nx = state0.size
    nu = control0.size
    dt = task._env.params.dt
    N = 20 if nx < 100 else 10

    alpha = 0.003 * dt
    def disturbance(X):
        N, nx = X.shape
        E0 = alpha * jnp.eye(nx, dtype=X.dtype)
        return jnp.broadcast_to(E0, (N, nx, nx))

    admm_cfg = gpu_sls.ADMMConfig(
        eps_abs=5e-2,
        eps_rel=1e-2,
        rho_max=1e3,
        max_iterations=100,
        rho_update_frequency=25,
        initial_rho=10.0,
    )

    sls_cfg = gpu_sls.SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1e-2,
        enable_fastsls=False,
        initialize_nominal=True,
        max_initial_sqp_iterations=0,
        warm_start=False,
        rti=False,
    )

    sqp_cfg = gpu_sls.SQPConfig(
        max_sqp_iterations=1,
        warm_start=False,
        feas_tol=1e-2,
        step_tol=1e-4,
        line_search=True,
    )

    cfg = gpu_sls.MPCConfig(
        n=nx,
        nu=nu,
        N=N,
        dt=dt,
        W=None,
        u_ref=control0,
    )

    nc = constraints(state0, control0, 0.0).size

    X_in = jnp.tile(state0[None, :], (N + 1, 1))
    U_in = jnp.tile(control0[None, :], (N, 1))
    for k in range(N):
        X_in = X_in.at[k+1].set(task.step(X_in[k], U_in[k]))

    controller = gpu_sls.GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        cost=cost,
        constraints=constraints,
        obstacles=jnp.zeros((0, 3)),
        disturbance=disturbance,
        num_constraints=nc,
        shift=1,
        X_in=X_in,
        U_in=U_in,
    )

    state = state0
    if visualizer is not None:
        task.visualize(visualizer, state)

    execution_time = 0
    for k in range(task.horizon()):
        tic = time.time()
        u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u, Phi_xw, Phi_uw, Phi_xe, Phi_ue = controller.run(
            x0=state,
            reference=None,
            parameter=None,
            Xi=jnp.zeros((nx,nx))
        )
        elapsed = time.time() - tic

        print(f"MPC step took {elapsed * 1e3:.2f} ms")
        if k > 0:
            execution_time += elapsed

        control = u0
        if jnp.isnan(u0).any() or (jnp.abs(u0) > 1e2).any():
            control = control0

        task.log(state, control)
        state = task.step(state, control)

        if visualizer is not None:
            task.visualize(visualizer, state)

        wait = dt - elapsed
        if wait > 0:
            time.sleep(wait)

    task.log(state)
    return execution_time
