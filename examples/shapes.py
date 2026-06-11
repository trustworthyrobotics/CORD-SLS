import numpy as np
import viser


def draw_cone(
    server: viser.ViserServer,
    center_xy: tuple[float, float],
    radius: float,
    z_min: float,
    z_max: float,
    n_sides: int = 48,
    **kwargs,
):
    theta = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    base = np.stack([
        center_xy[0] + radius * np.cos(theta),
        center_xy[1] + radius * np.sin(theta),
        np.full_like(theta, z_min),
    ], axis=1)

    apex = np.array([[center_xy[0], center_xy[1], z_max]])

    vertices = np.vstack([base, apex])
    apex_idx = len(vertices) - 1

    faces = []
    for k in range(n_sides):
        k_next = (k + 1) % n_sides
        faces.append([k, k_next, apex_idx])
    faces = np.array(faces, dtype=np.int32)

    defaults = {
        "name": "cone",
        "color": (1.0, 0.4, 0.0),
        "side": "double",
    }
    for k, v in defaults.items():
        if k not in kwargs.keys():
            kwargs[k] = v

    server.scene.add_mesh_simple(
        vertices=vertices,
        faces=faces,
        **kwargs,
    )


def draw_half_ellipsoid(
    server: viser.ViserServer,
    center_xy: tuple[float, float],
    radius_x: float,
    radius_y: float,
    z_min: float,
    z_max: float,
    n_theta: int = 48,
    n_phi: int = 24,
    **kwargs,
):
    c = z_max - z_min

    phi = np.linspace(0.0, np.pi / 2.0, n_phi)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing="ij")

    x = center_xy[0] + radius_x * np.sin(phi_grid) * np.cos(theta_grid)
    y = center_xy[1] + radius_y * np.sin(phi_grid) * np.sin(theta_grid)
    z = z_min + c * np.cos(phi_grid)

    vertices = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)

    faces = []
    for i in range(n_phi - 1):
        for j in range(n_theta):
            j_next = (j + 1) % n_theta
            v00 = i * n_theta + j
            v01 = i * n_theta + j_next
            v10 = (i + 1) * n_theta + j
            v11 = (i + 1) * n_theta + j_next
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces = np.array(faces, dtype=np.int32)

    defaults = {
        "name": "half_ellipsoid",
        "color": (1.0, 0.4, 0.0),
        "side": "double",
    }
    for k, v in defaults.items():
        kwargs.setdefault(k, v)

    server.scene.add_mesh_simple(
        vertices=vertices,
        faces=faces,
        **kwargs,
    )
