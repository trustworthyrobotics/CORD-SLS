from functools import partial
from typing import NamedTuple
from itertools import accumulate
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .collision import *
from .utils import *


class RopeEnv:
    def __init__(
            self,
            time_step: float,
            num_segments: int,
            rope_length: float,
            rope_diameter: float,
            youngs_modulus: float,
            mass_density: float,
            ground_friction_coeff: float=0.1,
            ground_peneration_dist: float=0.01,
            gravity: float=9.81,
            num_floating_grippers: int=0,
            weld_to_rope_ends: bool=False,
            grip_stiffness: float=1.0,
            gripper_radius: float | None=None,
            contact_smoothing: float=0.0,
        ):
        l = rope_length / num_segments
        r = rope_diameter / 2
        E = youngs_modulus
        rho = mass_density
        A = np.pi * r**2
        I = A * r**2 / 4
        k_axial = E * A / l
        k_bend = E * I / l
        m = rho * A * l

        ground_contact_stiffness = m * gravity / ground_peneration_dist

        if gripper_radius is None:
            gripper_radius = r * 2

        self.params = RopeEnvParams(
            dt=time_step,
            segment_length=l,
            segment_mass=m,
            k_axial=k_axial,
            k_bend=k_bend,
            rope_radius=r,
            ground_friction_coeff=ground_friction_coeff,
            ground_contact_stiffness=ground_contact_stiffness,
            gravity=gravity,
            grip_stiffness=grip_stiffness,
            gripper_radius=gripper_radius,
            num_nodes=num_segments + 1,
            num_floating_grippers=num_floating_grippers,
            weld_to_rope_ends=weld_to_rope_ends,
            contact_smoothing=contact_smoothing,
        )

    def state(
            self,
            x_node: np.ndarray | None = None,
            x_weld: np.ndarray | None = None,
            x_grip: np.ndarray | None = None,
        ) -> jax.Array:
        params = self.params

        if x_node is None:
            x_node = jnp.vstack((
                jnp.arange(params.num_nodes) * params.segment_length,
                jnp.zeros(params.num_nodes),
                jnp.ones(params.num_nodes) * params.rope_radius,
            )).T
        else:
            x_node = jnp.array(x_node)
            assert x_node.ndim == 2 and x_node.shape[-1] == 3

        if x_weld is None:
            x_weld = jnp.vstack((x_node[0], x_node[-1])) if params.weld_to_rope_ends else jnp.zeros((0, 3))
        else:
            x_weld = jnp.array(x_weld)
            assert x_weld.ndim == 2 and x_weld.shape[-1] == 3
            assert x_weld.shape[0] == (2 if params.weld_to_rope_ends else 0)

        if x_grip is None:
            p = 0.2
            x_grip = jnp.vstack((
                jnp.linspace(-p, p, params.num_floating_grippers) if params.num_floating_grippers > 1 else jnp.zeros(params.num_floating_grippers),
                jnp.zeros(params.num_floating_grippers),
                jnp.full(params.num_floating_grippers, p),
            )).T
        else:
            x_grip = jnp.array(x_grip)
            assert x_grip.ndim == 2 and x_grip.shape[-1] == 3
            assert x_grip.shape[0] == params.num_floating_grippers

        return _pack_state(x_node, x_weld, x_grip)

    def control(
            self,
            v_weld: np.ndarray | None = None,
            v_grip: np.ndarray | None = None,
            c_grip: np.ndarray | None = None,
        ) -> jax.Array:
        params = self.params

        if v_weld is None:
            v_weld = jnp.zeros(((2 if params.weld_to_rope_ends else 0), 3))
        else:
            v_weld = jnp.array(v_weld)
            assert v_weld.ndim == 2 and v_weld.shape[-1] == 3
            assert v_weld.shape[0] == (2 if params.weld_to_rope_ends else 0)

        if v_grip is None:
            v_grip = jnp.zeros((params.num_floating_grippers, 3))
        else:
            v_grip = jnp.array(v_grip)
            assert v_grip.ndim == 2 and v_grip.shape[-1] == 3
            assert v_grip.shape[0] == params.num_floating_grippers

        if c_grip is None:
            c_grip = jnp.zeros(params.num_floating_grippers)
        else:
            c_grip = jnp.array(c_grip)
            assert c_grip.ndim == 1 and c_grip.size == params.num_floating_grippers

        return _pack_control(v_weld, v_grip, c_grip)

    def step(self, state: jax.Array, control: jax.Array) -> jax.Array:
        return _step(state, control, self.params)

    def unpack_state(self, state: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        x_node, x_weld, x_grip = _unpack_state(state, self.params)
        return x_node, x_weld, x_grip

    def unpack_control(self, control: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        v_weld, v_grip, c_grip = _unpack_control(control, self.params)
        return v_weld, v_grip, c_grip

    def visualize(self, server, state: jax.Array) -> None:
        return _visualize(
            server=server,
            state=state,
            params=self.params,
            rope_color=(0.0, 0.0, 1.0),
            gripper_color=(1.0, 0.0, 0.0),
        )


class RopeEnvParams(NamedTuple):
    dt: float
    segment_length: float
    segment_mass: float
    k_axial: float
    k_bend: float
    rope_radius: float
    ground_friction_coeff: float
    ground_contact_stiffness: float
    gravity: float
    grip_stiffness: float
    gripper_radius: float
    num_nodes: int
    num_floating_grippers: int
    weld_to_rope_ends: bool
    contact_smoothing: float


def _pack_state(x_node: jax.Array, x_weld: jax.Array, x_grip: jax.Array) -> jax.Array:
    assert x_node.ndim == 2 and x_node.shape[-1] == 3
    assert x_weld.ndim == 2 and x_weld.shape[-1] == 3
    assert x_grip.ndim == 2 and x_grip.shape[-1] == 3
    state = jnp.concatenate((x_node.ravel(), x_weld.ravel(), x_grip.ravel()))
    return state

@partial(jax.jit, static_argnums=1)
def _unpack_state(state: jax.Array, params: RopeEnvParams) -> tuple[jax.Array, jax.Array, jax.Array]:
    assert state.ndim == 1
    sizes = [
        params.num_nodes * 3,
        (2 if params.weld_to_rope_ends else 0) * 3,
        params.num_floating_grippers * 3,
    ]
    assert state.size == sum(sizes)
    splits = list(accumulate(sizes))[:-1]
    x_node, x_weld, x_grip = jnp.split(state, splits)
    x_node = x_node.reshape(-1, 3)
    x_weld = x_weld.reshape(-1, 3)
    x_grip = x_grip.reshape(-1, 3)
    return x_node, x_weld, x_grip

def _pack_control(v_weld: jax.Array, v_grip: jax.Array, c_grip: jax.Array) -> jax.Array:
    assert v_weld.ndim == 2 and v_weld.shape[-1] == 3
    assert v_grip.ndim == 2 and v_grip.shape[-1] == 3
    assert c_grip.ndim == 1
    control = jnp.concatenate((v_weld.ravel(), v_grip.ravel(), c_grip))
    return control

@partial(jax.jit, static_argnums=1)
def _unpack_control(control: jax.Array, params: RopeEnvParams) -> tuple[jax.Array, jax.Array, jax.Array]:
    assert control.ndim == 1
    sizes = [
        (2 if params.weld_to_rope_ends else 0) * 3,
        params.num_floating_grippers * 3,
        params.num_floating_grippers,
    ]
    assert control.size == sum(sizes)
    splits = list(accumulate(sizes))[:-1]
    v_weld, v_grip, c_grip = jnp.split(control, splits)
    v_weld = v_weld.reshape(-1, 3)
    v_grip = v_grip.reshape(-1, 3)
    return v_weld, v_grip, c_grip


@partial(jax.jit, static_argnums=1)
def _elastic_energy(x_node: jax.Array, params: RopeEnvParams):
    x = x_node.reshape(-1, 3)
    dx = x[1:] - x[:-1]
    lengths = vector_norm(dx)
    E_axial = 0.5 * params.k_axial * jnp.sum((lengths - params.segment_length)**2)

    t = dx / lengths[:, None]
    t1, t2 = t[:-1], t[1:]
    sin_ang = vector_norm(jnp.cross(t1, t2))
    cos_ang = jnp.sum(t1 * t2, axis=1)
    curvature = 2 * sin_ang / (1 + cos_ang)
    # curvature = jnp.arctan2(sin_ang, cos_ang)  # Alternative formula
    E_bend = 0.5 * params.k_bend * jnp.sum(curvature**2)

    return E_axial + E_bend

_elastic_energy_grad = jax.jit(jax.grad(_elastic_energy), static_argnums=1)

_elastic_energy_hess = jax.jit(jax.hessian(_elastic_energy), static_argnums=1)


@partial(jax.jit, static_argnums=2)
def _deformable_grip_jacobian_stiffness(
    state: jax.Array,
    control: jax.Array,
    params: RopeEnvParams
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x_node, x_weld, x_grip = _unpack_state(state, params)
    v_weld, v_grip, c_grip = _unpack_control(control, params)

    Js = []
    phis = []
    ks = []

    if params.weld_to_rope_ends:
        J = jnp.zeros((6, x_node.size))
        J = J.at[:3, :3].set(jnp.eye(3))
        J = J.at[-3:, -3:].set(jnp.eye(3))
        phi = jnp.zeros(6)
        k = jnp.ones(6) * params.grip_stiffness

        Js.append(J)
        phis.append(phi)
        ks.append(k)

    for i in range(params.num_floating_grippers):
        J, phi = LinesegmentsPointCollision(x_node, x_grip[i], closest_only=True)
        dist_squared = jnp.sum(phi**2, axis=-1)

        k = params.grip_stiffness
        if params.contact_smoothing == 0.0:
            k *= (dist_squared <= (params.rope_radius + params.gripper_radius)**2)
        else:
            k *= jnp.exp(-jnp.sqrt(dist_squared + 1e-12) / params.contact_smoothing)
        k *= jax.nn.sigmoid(15 * (c_grip[i] - 0.5))
        k = jnp.repeat(k, 3)

        Js.append(J)
        phis.append(phi)
        ks.append(k)

    if Js:
        J = jnp.concatenate(Js)
        phi = jnp.concatenate(phis)
        K = jnp.diag(jnp.concatenate(ks))
    else:
        J = jnp.zeros((0, x_node.size))
        phi = jnp.zeros(0)
        K = jnp.zeros((0,0))

    return J, phi, K


@partial(jax.jit, static_argnums=3)
def _deformable_residual(
    x_node_next: jax.Array,
    state: jax.Array,
    control: jax.Array,
    params: RopeEnvParams,
) -> jax.Array:

    x_node, x_weld, x_grip = _unpack_state(state, params)
    v_weld, v_grip, c_grip = _unpack_control(control, params)

    x_node_next = x_node_next.ravel()
    x_node = x_node.ravel()

    x_weld_next = x_weld + v_weld * params.dt
    x_grip_next = x_grip + v_grip * params.dt
    x_ref = jnp.concatenate((x_weld_next.ravel(), x_grip_next.ravel()))

    m_over_dt2 = params.segment_mass / params.dt**2
    mg = jnp.tile(jnp.array([0, 0, -params.segment_mass * params.gravity]), params.num_nodes)

    J, phi, K = _deformable_grip_jacobian_stiffness(state, control, params)

    return m_over_dt2 * (x_node_next - x_node) \
        + _elastic_energy_grad(x_node_next, params) \
        + J.T @ K @ (J @ x_node_next + phi * 0.9 - x_ref) \
        - mg

@partial(jax.jit, static_argnums=3)
def _deformable_residual_grad(
    x_node_next: jax.Array,
    state: jax.Array,
    control: jax.Array,
    params: RopeEnvParams,
) -> jax.Array:
    x_node_next = x_node_next.ravel()

    M_over_dt2 = params.segment_mass / params.dt**2 * jnp.eye(x_node_next.size)

    K_elastic = _elastic_energy_hess(x_node_next, params)
    K_elastic = jax.lax.stop_gradient(K_elastic)  # Assume ∂K/∂x=0 to avoid expensive 3rd order tensor.

    J, phi, K = _deformable_grip_jacobian_stiffness(state, control, params)

    return M_over_dt2 + K_elastic + J.T @ K @ J


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _step_deformable(
    state: jax.Array,
    control: jax.Array,
    params: RopeEnvParams
) -> jax.Array:
    max_iters, tol= 50, 1e-4
    max_ls_iters, ls_rho = 6, 0.5

    x_node, _, _ = _unpack_state(state, params)
    x_node_next = x_node

    def cond_fn(carry):
        x_node_next, i = carry
        r = _deformable_residual(x_node_next, state, control, params)
        return (jnp.linalg.norm(r, ord=jnp.inf) >= tol) & (i < max_iters)

    def body_fn(carry):
        x_node_next, i = carry
        r = _deformable_residual(x_node_next, state, control, params)
        A = _deformable_residual_grad(x_node_next, state, control, params)
        dx = jnp.linalg.solve(A, -r).reshape(-1, 3)

        merit = lambda x: jnp.linalg.norm(_deformable_residual(x, state, control, params), ord=jnp.inf)
        m_old = merit(x_node_next)

        def ls_cond_fn(ls_carry):
            alpha, ls_i = ls_carry
            m_new = merit(x_node_next + alpha * dx)
            sufficient_decrease = (m_new <= m_old).all()
            return (~sufficient_decrease) & (ls_i < max_ls_iters)

        def ls_body_fn(ls_carry):
            alpha, ls_i = ls_carry
            return alpha * ls_rho, ls_i + 1

        alpha, ls_iters = jax.lax.while_loop(ls_cond_fn, ls_body_fn, (1.0, 0))

        x_node_next = (x_node_next + alpha * dx).astype(x_node_next.dtype)
        return x_node_next, i + 1

    x_node_next, iters = jax.lax.while_loop(cond_fn, body_fn, (x_node_next, 0))

    def _check(iters):
        if iters >= max_iters:
            raise RuntimeError(
                f"Newton solve not converged after {max_iters} iterations, "
                f"consider increasing the tolerance (currently {tol})"
            )

    # jax.debug.callback(_check, iters)
    return x_node_next

def _step_deformable_fwd(state: jax.Array, control: jax.Array, params: RopeEnvParams):
    x_node_next = _step_deformable(state, control, params)
    A = _deformable_residual_grad(x_node_next, state, control, params)
    decompA = jsp.linalg.lu_factor(A)
    _, pullback = jax.vjp(
        lambda _state, _control: _deformable_residual(x_node_next, _state, _control, params),
        state,
        control,
    )
    return x_node_next, (decompA, pullback)

def _step_deformable_bwd(params, res, x_node_next_adjoint):
    (decompA, pullback) = res
    h = jsp.linalg.lu_solve(decompA, x_node_next_adjoint.ravel())
    state_adjoint, control_adjoint = pullback(-h)
    return state_adjoint, control_adjoint

_step_deformable.defvjp(_step_deformable_fwd, _step_deformable_bwd)


@partial(jax.jit, static_argnums=3)
def _step_contact(
    x_node: jax.Array,
    state: jax.Array,
    control: jax.Array,
    params: RopeEnvParams,
) -> jax.Array:
    J, phi = PolygonizeFrictionCone(
        *SpheresGroundCollision(x_node, params.rope_radius),
        params.ground_friction_coeff
    )
    f = J.T @ smoothrelu(-params.ground_contact_stiffness * phi, params.contact_smoothing * 0.0)

    A = _deformable_residual_grad(x_node, state, control, params)
    x_node_next = x_node + jnp.linalg.solve(A, f).reshape(-1,3)

    return x_node_next

@partial(jax.jit, static_argnums=2)
def _step_all(state: jax.Array, control: jax.Array, params: RopeEnvParams) -> jax.Array:
    x_node_next = _step_deformable(state, control, params)
    x_node_next = _step_contact(x_node_next, state, control, params)

    _, x_weld, x_grip = _unpack_state(state, params)
    v_weld, v_grip, _ = _unpack_control(control, params)
    x_weld_next = x_weld + v_weld * params.dt
    x_grip_next = x_grip + v_grip * params.dt

    state_next = _pack_state(x_node_next, x_weld_next, x_grip_next)
    return state_next


@partial(jax.jit, static_argnums=2)
def _step(state: jax.Array, control: jax.Array, params: RopeEnvParams) -> jax.Array:
    if params.contact_smoothing == 0.0:
        return _step_all(state, control, params)

    state_smoothed = _step_all(state, control, params)
    state_true = _step_all(state, control, params._replace(contact_smoothing=0.0))
    return jax.lax.stop_gradient(state_true) + state_smoothed - jax.lax.stop_gradient(state_smoothed)


def _visualize(
        server,
        state: jax.Array,
        params: RopeEnvParams,
        rope_color: tuple[int,int,int] | tuple[float,float,float],
        gripper_color: tuple[int,int,int] | tuple[float,float,float],
        gripper_opacity: float=0.7,
    ) -> None:
    def draw_shpere(pos, radius, **kwargs):
        server.scene.add_icosphere(radius=radius, position=pos, **kwargs)

    def draw_cylinder(pos1, pos2, radius, **kwargs):
        direction = pos2 - pos1
        length = np.linalg.norm(direction)
        if length < 1e-8:
            return
        direction /= length
        midpoint = 0.5 * (pos1 + pos2)

        z_axis = np.array([0.0, 0.0, 1.0])
        v = np.cross(z_axis, direction)
        c = np.dot(z_axis, direction)

        eps = 1e-8
        if c > 1 - eps:
            q = np.array([1.0, 0.0, 0.0, 0.0])  # aligned with +z
        elif c < -1 + eps:
            q = np.array([0.0, 1.0, 0.0, 0.0])  # aligned with -z (180° rotation)
        else:
            s = np.sqrt(2 * (1 + c))
            q = np.array([
                0.5 * s,
                v[0] / s,
                v[1] / s,
                v[2] / s,
            ])
        server.scene.add_cylinder(radius=radius, height=length, position=midpoint, wxyz=q, **kwargs)

    x_node, x_weld, x_grip = _unpack_state(state, params)

    for i in range(params.num_nodes):
        draw_shpere(x_node[i], params.rope_radius, name=f"rope/node{i}", color=rope_color)
        if i == params.num_nodes - 1:
            continue
        draw_cylinder(x_node[i], x_node[i+1], params.rope_radius, name=f"rope/seg{i}", color=rope_color)

    for i in range(x_weld.shape[0]):
        draw_shpere(x_weld[i], params.gripper_radius, name=f"end_effector{i}/ee", color=gripper_color)
        draw_cylinder(x_weld[i], x_node[0 if i==0 else -1], params.rope_radius*0.2, name=f"end_effector{i}/weld", color=gripper_color)

    for i in range(x_grip.shape[0]):
        draw_shpere(x_grip[i], params.rope_radius*2, name=f"gripper{i}", color=gripper_color, opacity=gripper_opacity)
