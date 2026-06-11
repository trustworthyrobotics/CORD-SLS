import jax
import jax.numpy as jnp

from .utils import vector_norm


def SpheresGroundCollision(sphere_positions: jax.Array, sphere_radius: float) -> tuple[jax.Array, jax.Array]:
    assert sphere_positions.ndim == 2 and sphere_positions.shape[1] == 3
    J = jnp.kron(
        jnp.eye(sphere_positions.shape[0]),
        jnp.array([[0, 0, 1], # Contact normal is the z-axis.
                   [1, 0, 0],
                   [0, 1, 0]])
    )
    phi = sphere_positions[:, 2] - sphere_radius
    return J, phi


def LinesegmentsPointCollision(
        node_positions: jax.Array,
        point: jax.Array,
        closest_only: bool=False
    ) -> tuple[jax.Array, jax.Array]:
    assert node_positions.ndim == 2 and node_positions.shape[-1] == 3
    assert point.ndim == 1 and point.shape[-1] == 3

    n = node_positions.shape[0]

    A = node_positions[:-1]
    B = node_positions[1:]
    P = point

    AB = B - A                                                 # (n-1,3)
    AP = P - A                                                 # (n-1,3)
    t = jnp.sum(AP * AB, axis=-1) / jnp.sum(AB * AB, axis=-1)  # (n-1,)
    t = jnp.clip(t, 0.0, 1.0)
    Q = A + t[:, None] * AB                                    # (n-1,3)
    QP = P - Q

    if not closest_only:
        w = jnp.zeros((n - 1, n))
        w = w.at[jnp.arange(n-1), jnp.arange(n-1)].set(1 - t)
        w = w.at[jnp.arange(n-1), jnp.arange(n-1) + 1].set(t)
        J = jnp.kron(w, jnp.eye(3))
        phi = QP.ravel()
    else:
        idx = jnp.argmin(jnp.sum(QP**2, axis=1))
        w = jnp.zeros((1, n))
        w = w.at[0, idx].set(1 - t[idx])
        w = w.at[0, idx + 1].set(t[idx])
        J = jnp.kron(w, jnp.eye(3))
        phi = QP[idx]

    return J, phi


def TrianglesPointCollision(
    node_positions: jax.Array,
    triangles: jax.Array,
    point: jax.Array,
    closest_only: bool=False
) -> tuple[jax.Array, jax.Array]:
    assert node_positions.ndim == 2 and node_positions.shape[-1] == 3
    assert triangles.ndim == 2 and triangles.shape[-1] == 3
    assert point.ndim == 1 and point.shape[-1] == 3

    Nn = node_positions.shape[0]
    Nt = triangles.shape[0]

    A = node_positions[triangles[:, 0]]  # (Nt, 3)
    B = node_positions[triangles[:, 1]]  # (Nt, 3)
    C = node_positions[triangles[:, 2]]  # (Nt, 3)
    P = point[None, :]                   # (1, 3)

    # Edge vectors
    e0 = B - A
    e1 = C - A
    e2 = P - A

    # Dot products
    d00 = jnp.sum(e0 * e0, axis=1)
    d01 = jnp.sum(e0 * e1, axis=1)
    d11 = jnp.sum(e1 * e1, axis=1)
    d20 = jnp.sum(e2 * e0, axis=1)
    d21 = jnp.sum(e2 * e1, axis=1)

    denom = d00 * d11 - d01 * d01
    inv_denom = 1.0 / jnp.maximum(denom, 1e-12)

    # Unconstrained barycentric coordinates
    v = (d11 * d20 - d01 * d21) * inv_denom
    w = (d00 * d21 - d01 * d20) * inv_denom

    # mask for negative barycentrics
    v = jnp.maximum(v, 0.0)
    w = jnp.maximum(w, 0.0)

    # check if we are outside v+w<=1
    s = jnp.maximum(v + w, 1.0)
    v = v / s
    w = w / s
    u = 1.0 - v - w
    Q = u[:, None] * A + v[:, None] * B + w[:, None] * C

    QP = P - Q
    if not closest_only:
        J = jnp.zeros((Nt, Nn))
        J = J.at[jnp.arange(Nt), triangles[:,0]].set(u)
        J = J.at[jnp.arange(Nt), triangles[:,1]].set(v)
        J = J.at[jnp.arange(Nt), triangles[:,2]].set(w)
        J = jnp.kron(J, jnp.eye(3))
        phi = QP.ravel()
    else:
        idx = jnp.argmin(jnp.sum(QP**2, axis=1))
        J = jnp.zeros((1, Nn))
        J = J.at[0, triangles[idx,0]].set(u[idx])
        J = J.at[0, triangles[idx,1]].set(v[idx])
        J = J.at[0, triangles[idx,2]].set(w[idx])
        J = jnp.kron(J, jnp.eye(3))
        phi = QP[idx]

    return J, phi


def PolygonizeFrictionCone(J: jax.Array, phi: jax.Array, friction_coefficient: float) -> tuple[jax.Array, jax.Array]:
    """
    Converts constraint
        J dq + [ϕ₁ 0 0 ϕ₂ 0 0 ⋯]ᵀ ∈ ℱ_μ × ℱ_μ ⋯
    to an approximately equivalent constraint
        J̃ dq + ϕ̃  ≥ 0
    """
    assert J.shape[0] == len(phi) * 3
    mu = friction_coefficient
    Jn, Jt1, Jt2 = J[0::3], J[1::3], J[2::3]
    J_tilde = jnp.concat(
        [
            Jn - mu * Jt1,
            Jn - mu * Jt2,
            Jn + mu * Jt1,
            Jn + mu * Jt2,
        ],
        axis=0,
    )
    phi_tilde = jnp.tile(phi, 4)
    return J_tilde, phi_tilde
