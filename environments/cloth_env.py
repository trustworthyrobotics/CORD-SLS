from functools import partial
from typing import NamedTuple
from itertools import accumulate
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .collision import *
from .utils import *


class ClothEnv:
    def __init__(
            self,
            x_node_rest: np.ndarray,
            triangles: np.ndarray,
            thickness: float,
            youngs_modulus: float,
            possion_ratio: float,
            mass_density: float,
            time_step: float,
            ground_friction_coeff: float=0.1,
            ground_peneration_dist: float=0.05,
            gravity: float=9.81,
            num_floating_grippers: int=0,
            grip_stiffness: float=1.0,
            gripper_radius: float | None=None,
            contact_smoothing: float=1e-3,
        ):
        adjacency = _triangles_adjacency(triangles)

        x_node_rest = jnp.array(x_node_rest, dtype=jnp.float32)
        triangles = jnp.array(triangles, dtype=jnp.int32)
        adjacency = jnp.array(adjacency, dtype=jnp.int32)

        Alpha = _deformation_Alpha(x_node_rest, triangles)
        Alpha_inv = jnp.linalg.inv(Alpha)
        area = jnp.sqrt(jnp.linalg.det(Alpha)) / 2

        Beta = _deformation_Beta(x_node_rest, triangles, adjacency)

        total_mass = jnp.sum(area) * thickness * mass_density
        node_mass = float(total_mass) / x_node_rest.shape[0]

        ground_contact_stiffness = node_mass * gravity / ground_peneration_dist

        if gripper_radius is None:
            gripper_radius = thickness * 3

        self.params = ClothEnvParams(
            dt=time_step,
            x_node_rest=x_node_rest,
            triangles=triangles,
            adjacency=adjacency,
            Alpha_inv_rest=Alpha_inv,
            Beta_rest=Beta,
            area_rest=area,
            node_mass=node_mass,
            thickness=thickness,
            youngs_modulus=youngs_modulus,
            possion_ratio=possion_ratio,
            ground_friction_coeff=ground_friction_coeff,
            ground_contact_stiffness=ground_contact_stiffness,
            gravity=gravity,
            grip_stiffness=grip_stiffness,
            gripper_radius=gripper_radius,
            num_floating_grippers=num_floating_grippers,
            contact_smoothing=contact_smoothing,
        )

    @staticmethod
    def from_regular_grid(cloth_width: float, num_div_side: int, **kwargs):
        n = num_div_side + 1

        x = np.linspace(-cloth_width / 2, cloth_width / 2, n)
        y = np.linspace(-cloth_width / 2, cloth_width / 2, n)

        X, Y = np.meshgrid(x, y, indexing="ij")
        x_node_rest = np.stack([
            X.ravel(),
            Y.ravel(),
            np.zeros(n * n),
        ], axis=1)

        def vid(i, j):
            return i * n + j

        triangles = []
        for i in range(num_div_side):
            for j in range(num_div_side):
                v00 = vid(i, j)
                v10 = vid(i + 1, j)
                v01 = vid(i, j + 1)
                v11 = vid(i + 1, j + 1)

                triangles.append([v00, v10, v11])
                triangles.append([v00, v11, v01])

        return ClothEnv(
            x_node_rest=x_node_rest,
            triangles=triangles,
            **kwargs,
        )

    @staticmethod
    def from_obj_file(filename: str, **kwargs):
        nodes, triangles = _load_obj_mesh(filename)
        return ClothEnv(
            x_node_rest=nodes,
            triangles=triangles,
            **kwargs,
        )

    def state(
            self,
            x_node: np.ndarray | None = None,
            x_grip: np.ndarray | None = None,
        ) -> jax.Array:
        params = self.params

        if x_node is None:
            x_node = self.params.x_node_rest
        else:
            x_node = jnp.array(x_node)
            assert x_node.ndim == 2 and x_node.shape[-1] == 3

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

        return _pack_state(x_node, x_grip)

    def control(
            self,
            v_grip: np.ndarray | None = None,
            c_grip: np.ndarray | None = None,
        ) -> jax.Array:
        params = self.params

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

        return _pack_control(v_grip, c_grip)

    def step(self, state: jax.Array, control: jax.Array) -> jax.Array:
        return _step(state, control, self.params)

    def unpack_state(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_node, x_grip = _unpack_state(state, self.params)
        return x_node, x_grip

    def unpack_control(self, control: jax.Array) -> tuple[jax.Array, jax.Array]:
        v_grip, c_grip = _unpack_control(control, self.params)
        return v_grip, c_grip

    def visualize(self, server, state: jax.Array) -> None:
        return _visualize(
            server=server,
            state=state,
            params=self.params,
            cloth_color=(0.4, 0.7, 1.0),
            gripper_color=(1.0, 0.0, 0.0),
        )


class ClothEnvParams(NamedTuple):
    dt: float
    x_node_rest: jax.Array
    triangles: jax.Array
    adjacency: jax.Array
    Alpha_inv_rest: jax.Array
    Beta_rest: jax.Array
    area_rest: jax.Array
    node_mass: float
    thickness: float
    youngs_modulus: float
    possion_ratio: float
    ground_friction_coeff: float
    ground_contact_stiffness: float
    gravity: float
    grip_stiffness: float
    gripper_radius: float
    num_floating_grippers: int
    contact_smoothing: float

    def __hash__(self):
        return hash((
            self.dt,
            id(self.x_node_rest),
            id(self.triangles),
            id(self.adjacency),
            id(self.Alpha_inv_rest),
            id(self.Beta_rest),
            id(self.area_rest),
            self.node_mass,
            self.thickness,
            self.youngs_modulus,
            self.possion_ratio,
            self.ground_friction_coeff,
            self.ground_contact_stiffness,
            self.gravity,
            self.grip_stiffness,
            self.gripper_radius,
            self.num_floating_grippers,
            self.contact_smoothing,
        ))

    def __eq__(self, other):
        if not isinstance(other, ClothEnvParams):
            return False

        return (
            self.dt == other.dt and
            self.x_node_rest is other.x_node_rest and
            self.triangles is other.triangles and
            self.adjacency is other.adjacency and
            self.Alpha_inv_rest is other.Alpha_inv_rest and
            self.Beta_rest is other.Beta_rest and
            self.area_rest is other.area_rest and
            self.node_mass == other.node_mass and
            self.thickness == other.thickness and
            self.youngs_modulus == other.youngs_modulus and
            self.possion_ratio == other.possion_ratio and
            self.ground_friction_coeff == other.ground_friction_coeff and
            self.ground_contact_stiffness == other.ground_contact_stiffness and
            self.gravity == other.gravity and
            self.grip_stiffness == other.grip_stiffness and
            self.gripper_radius == other.gripper_radius and
            self.num_floating_grippers == other.num_floating_grippers and
            self.contact_smoothing == other.contact_smoothing
        )


def _pack_state(x_node: jax.Array, x_grip: jax.Array) -> jax.Array:
    assert x_node.ndim == 2 and x_node.shape[-1] == 3
    assert x_grip.ndim == 2 and x_grip.shape[-1] == 3
    state = jnp.concatenate((x_node.ravel(), x_grip.ravel()))
    return state

@partial(jax.jit, static_argnums=1)
def _unpack_state(state: jax.Array, params: ClothEnvParams) -> tuple[jax.Array, jax.Array]:
    assert state.ndim == 1
    sizes = [
        params.x_node_rest.size,
        params.num_floating_grippers * 3,
    ]
    assert state.size == sum(sizes)
    splits = list(accumulate(sizes))[:-1]
    x_node, x_grip = jnp.split(state, splits)
    x_node = x_node.reshape(-1, 3)
    x_grip = x_grip.reshape(-1, 3)
    return x_node, x_grip

def _pack_control(v_grip: jax.Array, c_grip: jax.Array) -> jax.Array:
    assert v_grip.ndim == 2 and v_grip.shape[-1] == 3
    assert c_grip.ndim == 1
    control = jnp.concatenate((v_grip.ravel(), c_grip))
    return control

@partial(jax.jit, static_argnums=1)
def _unpack_control(control: jax.Array, params: ClothEnvParams) -> tuple[jax.Array, jax.Array]:
    assert control.ndim == 1
    sizes = [
        params.num_floating_grippers * 3,
        params.num_floating_grippers,
    ]
    assert control.size == sum(sizes)
    splits = list(accumulate(sizes))[:-1]
    v_grip, c_grip = jnp.split(control, splits)
    v_grip = v_grip.reshape(-1, 3)
    return v_grip, c_grip


def _deformation_Alpha(x_node: jax.Array, triangles: jax.Array) -> jax.Array:
    xi = x_node[triangles[:, 0]]  # (Nt, 3)
    xj = x_node[triangles[:, 1]]  # (Nt, 3)
    xk = x_node[triangles[:, 2]]  # (Nt, 3)
    e1e2 = jnp.stack((xj - xi, xk - xi), axis=-1)  # (Nt, 3, 2)
    Alpha = jnp.einsum("Nki,Nkj->Nij", e1e2, e1e2)    # (Nt, 2, 2)
    return Alpha

def _deformation_Beta(x_node: jax.Array, triangles: jax.Array, adjacency: jax.Array) -> jax.Array:
    Nt = triangles.shape[0]
    xi = x_node[triangles[:, 0]]   # (Nt, 3)
    xj = x_node[triangles[:, 1]]   # (Nt, 3)
    xk = x_node[triangles[:, 2]]   # (Nt, 3)
    e1 = xj - xi  # (Nt, 3)
    e2 = xk - xi  # (Nt, 3)

    n = jnp.cross(e1, e2)  # (Nt, 3)
    n = n / vector_norm(n, axis=1, keepdims=True)

    # For boundary edges (adjacency == -1), use own normal.
    adj_safe = jnp.where(adjacency < 0, jnp.arange(Nt)[:, None], adjacency)
    n_i = n[adj_safe[:, 0]]  # neighbor across edge (j,k)
    n_j = n[adj_safe[:, 1]]  # neighbor across edge (k,i)
    n_k = n[adj_safe[:, 2]]  # neighbor across edge (i,j)

    dn1 = n_j - n_i  # (Nt, 3)
    dn2 = n_k - n_i  # (Nt, 3)

    B11 = jnp.sum(dn1 * e1, axis=1)  # (Nt,)
    B12 = jnp.sum(dn1 * e2, axis=1)  # (Nt,)
    B21 = jnp.sum(dn2 * e1, axis=1)  # (Nt,)
    B22 = jnp.sum(dn2 * e2, axis=1)  # (Nt,)

    Beta = 2.0 * jnp.stack((
        jnp.stack((B11, B12), axis=-1),
        jnp.stack((B21, B22), axis=-1),
    ), axis=-2)  # (Nt, 2, 2)
    return Beta


@partial(jax.jit, static_argnums=1)
def _saint_venant_energy(Strain: jax.Array, params: ClothEnvParams) -> jax.Array:
    E, nu = params.youngs_modulus, params.possion_ratio
    lame_lambda = E * nu / (1 - nu**2)
    lame_mu = E / (2 * (1+ nu))
    return (
        0.5 * lame_lambda * jnp.linalg.trace(Strain)**2
        + lame_mu * jnp.linalg.trace(Strain @ Strain)
    )

@partial(jax.jit, static_argnums=1)
def _elastic_energy(x_node: jax.Array, params: ClothEnvParams):
    x_node = x_node.reshape(-1, 3)
    Alpha = _deformation_Alpha(x_node, params.triangles)
    Beta = _deformation_Beta(x_node, params.triangles, params.adjacency)

    Alpha_inv_rest = params.Alpha_inv_rest
    Beta_rest = params.Beta_rest
    area = params.area_rest
    h = params.thickness

    E_shear = _saint_venant_energy(Alpha_inv_rest @ Alpha - jnp.eye(2), params) * area * h / 4
    E_bend = _saint_venant_energy(Alpha_inv_rest @ (Beta - Beta_rest), params) * area * h**3 / 12

    return jnp.sum(E_shear) + jnp.sum(E_bend)

_elastic_energy_grad = jax.jit(jax.grad(_elastic_energy), static_argnums=1)

_elastic_energy_hess = jax.jit(jax.hessian(_elastic_energy), static_argnums=1)


@partial(jax.jit, static_argnums=2)
def _deformable_grip_jacobian_stiffness(
    state: jax.Array,
    control: jax.Array,
    params: ClothEnvParams
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x_node, x_grip = _unpack_state(state, params)
    v_grip, c_grip = _unpack_control(control, params)

    Js = []
    phis = []
    ks = []

    for i in range(params.num_floating_grippers):
        J, phi = TrianglesPointCollision(x_node, params.triangles, x_grip[i], closest_only=True)
        dist_squared = jnp.sum(phi**2, axis=-1)

        k = params.grip_stiffness
        if params.contact_smoothing == 0.0:
            k *= (dist_squared <= (params.thickness * 0.5 + params.gripper_radius)**2)
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
    params: ClothEnvParams,
) -> jax.Array:
    x_node, x_grip = _unpack_state(state, params)
    v_grip, c_grip = _unpack_control(control, params)

    x_node_next = x_node_next.ravel()
    x_node = x_node.ravel()

    m_over_dt2 = params.node_mass / params.dt**2
    mg = jnp.tile(jnp.array([0, 0, -params.node_mass * params.gravity]), x_node.size // 3)

    J, phi, K = _deformable_grip_jacobian_stiffness(state, control, params)

    x_grip_next = x_grip + v_grip * params.dt
    x_ref = x_grip_next.ravel()

    return m_over_dt2 * (x_node_next - x_node) \
        + _elastic_energy_grad(x_node_next, params) \
        + J.T @ K @ (J @ x_node_next + phi * 0.9 - x_ref) \
        - mg

@partial(jax.jit, static_argnums=3)
def _deformable_residual_grad(
    x_node_next: jax.Array,
    state: jax.Array,
    control: jax.Array,
    params: ClothEnvParams,
) -> jax.Array:
    x_node_next = x_node_next.ravel()

    M_over_dt2 = params.node_mass / params.dt**2 * jnp.eye(x_node_next.size)

    K_elastic = _elastic_energy_hess(x_node_next, params)
    K_elastic = jax.lax.stop_gradient(K_elastic)  # Assume ∂K/∂x=0 to avoid expensive 3rd order tensor.

    J, phi, K = _deformable_grip_jacobian_stiffness(state, control, params)

    return M_over_dt2 + K_elastic + J.T @ K @ J


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _step_deformable(
    state: jax.Array,
    control: jax.Array,
    params: ClothEnvParams
) -> jax.Array:
    max_iters, tol= 100, 1e-4
    max_ls_iters, ls_rho = 8, 0.5

    x_node, _ = _unpack_state(state, params)
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

def _step_deformable_fwd(state: jax.Array, control: jax.Array, params: ClothEnvParams):
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
    params: ClothEnvParams,
) -> jax.Array:
    J, phi = PolygonizeFrictionCone(
        *SpheresGroundCollision(x_node, params.thickness / 2),
        params.ground_friction_coeff
    )
    f = J.T @ smoothrelu(-params.ground_contact_stiffness * phi, params.contact_smoothing * 0.0)

    A = _deformable_residual_grad(x_node, state, control, params)
    x_node_next = x_node + jnp.linalg.solve(A, f).reshape(-1,3)

    return x_node_next

@partial(jax.jit, static_argnums=2)
def _step_all(state: jax.Array, control: jax.Array, params: ClothEnvParams) -> jax.Array:
    x_node_next = _step_deformable(state, control, params)
    x_node_next = _step_contact(x_node_next, state, control, params)

    _, x_grip = _unpack_state(state, params)
    v_grip, _ = _unpack_control(control, params)
    x_grip_next = x_grip + v_grip * params.dt

    state_next = _pack_state(x_node_next, x_grip_next)
    return state_next


@partial(jax.jit, static_argnums=2)
def _step(state: jax.Array, control: jax.Array, params: ClothEnvParams) -> jax.Array:
    if params.contact_smoothing == 0.0:
        return _step_all(state, control, params)

    state_smoothed = _step_all(state, control, params)
    state_true = _step_all(state, control, params._replace(contact_smoothing=0.0))
    return jax.lax.stop_gradient(state_true) + state_smoothed - jax.lax.stop_gradient(state_smoothed)


def _visualize(
        server,
        state: jax.Array,
        params: ClothEnvParams,
        cloth_color: tuple[int,int,int] | tuple[float,float,float],
        gripper_color: tuple[int,int,int] | tuple[float,float,float],
        gripper_opacity: float=0.7,
    ) -> None:
    def draw_shpere(pos, radius, **kwargs):
        server.scene.add_icosphere(radius=radius, position=pos, **kwargs)

    x_node, x_grip = _unpack_state(state, params)
    x_node, x_grip = np.array(x_node), np.array(x_grip)
    triangles = np.array(params.triangles)

    server.scene.add_mesh_simple(
        name="cloth/mesh",
        vertices=x_node,
        faces=triangles,
        color=cloth_color,
        side="double",
        flat_shading=False,
    )

    for i in range(x_grip.shape[0]):
        draw_shpere(x_grip[i], params.gripper_radius * 0.2,
                    name=f"gripper{i}", color=gripper_color, opacity=gripper_opacity)


def _triangles_adjacency(triangles: np.ndarray) -> np.ndarray:
    triangles = np.array(triangles)
    assert triangles.ndim == 2 and triangles.shape[-1] == 3
    Nt = triangles.shape[0]

    edges = np.stack((
        triangles[:, [1, 2]],
        triangles[:, [2, 0]],
        triangles[:, [0, 1]],
    ), axis=1)                       # (Nt, 3, 2)
    edges = np.sort(edges, axis=-1)  # (Nt, 3, 2)

    flat_edges = edges.reshape(-1, 2)      # (Nt*3, 2)
    tri_ids = np.repeat(np.arange(Nt), 3)  # (Nt*3,)
    edge_ids = np.tile(np.arange(3), Nt)   # (Nt*3,)

    # Lexicographic sort edges
    order = np.lexsort((flat_edges[:, 1], flat_edges[:, 0]))
    sorted_flat_edges = flat_edges[order]

    # Consecutive identical edges are matching pairs
    same_as_next = np.all(sorted_flat_edges[:-1] == sorted_flat_edges[1:], axis=1)
    idx = np.nonzero(same_as_next)[0]

    a_tri_id = tri_ids[order[idx]]
    b_tri_id = tri_ids[order[idx + 1]]

    a_edge_id = edge_ids[order[idx]]
    b_edge_id = edge_ids[order[idx + 1]]

    neighbors = np.full((Nt, 3), -1, dtype=np.int32)
    neighbors[a_tri_id, a_edge_id] = b_tri_id
    neighbors[b_tri_id, b_edge_id] = a_tri_id
    return neighbors


def _load_obj_mesh(path):
    vertices = []
    faces = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()
                vertices.append([float(x), float(y), float(z)])

            elif line.startswith("f "):
                parts = line.split()[1:]
                face = []
                for p in parts:
                    idx = p.split("/")[0]
                    face.append(int(idx) - 1)   # OBJ is 1-indexed
                faces.append(face)

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    return vertices, faces
