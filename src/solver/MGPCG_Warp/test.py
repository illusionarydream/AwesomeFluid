import warp as wp
import numpy as np

from mgpcg import WarpMGPCG

wp.init()


# Dirichlet BC
# Poisson sin(kx x) * sin(ky y)
# [0, 1] x [0, 1]
@wp.kernel
def test_c0_init_kernel2D(
    cell_type: wp.array(dtype=wp.uint8, ndim=2),
    cell_diag: wp.array(dtype=wp.float32, ndim=2),
    mac_a_x: wp.array(dtype=wp.float32, ndim=2),
    mac_a_y: wp.array(dtype=wp.float32, ndim=2),
    rhs: wp.array(dtype=wp.float32, ndim=2),
    dx: float,
    kx: float,
    ky: float,
):
    i, j = wp.tid()

    cell_type[i, j] = wp.uint8(0)  # fluid cell
    cell_diag[i, j] = 4.0
    if i > 0:
        mac_a_x[i, j] = 1.0
    if j > 0:
        mac_a_y[i, j] = 1.0

    fact = dx * dx * (kx * kx + ky * ky)
    x = (wp.float32(i) + 1.0) * dx
    y = (wp.float32(j) + 1.0) * dx
    rhs[i, j] = fact * wp.sin(kx * x) * wp.sin(ky * y)


# Dirichlet BC
# Poisson sin(kx x) * sin(ky y) * sin(kz z)
# [0, 1]  x [0, 1] x [0, 1]
@wp.kernel
def test_c0_init_kernel3D(
    cell_type: wp.array(dtype=wp.uint8, ndim=3),
    cell_diag: wp.array(dtype=wp.float32, ndim=3),
    mac_a_x: wp.array(dtype=wp.float32, ndim=3),
    mac_a_y: wp.array(dtype=wp.float32, ndim=3),
    mac_a_z: wp.array(dtype=wp.float32, ndim=3),
    rhs: wp.array(dtype=wp.float32, ndim=3),
    dx: float,
    kx: float,
    ky: float,
    kz: float,
):
    i, j, k = wp.tid()
    cell_type[i, j, k] = wp.uint8(0)  # fluid cell
    cell_diag[i, j, k] = 6.0
    if i > 0:
        mac_a_x[i, j, k] = 1.0
    if j > 0:
        mac_a_y[i, j, k] = 1.0
    if k > 0:
        mac_a_z[i, j, k] = 1.0

    fact = dx * dx * (kx * kx + ky * ky + kz * kz)
    x = (wp.float32(i) + 1.0) * dx
    y = (wp.float32(j) + 1.0) * dx
    z = (wp.float32(k) + 1.0) * dx
    rhs[i, j, k] = fact * wp.sin(kx * x) * wp.sin(ky * y) * wp.sin(kz * z)


# Neumann BC
# Poisson cos(kx x) * cos(ky y)
# [0, 1] x [0, 1]
@wp.kernel
def test_c1_init_kernel2D(
    cell_type: wp.array(dtype=wp.uint8, ndim=2),
    cell_diag: wp.array(dtype=wp.float32, ndim=2),
    mac_a_x: wp.array(dtype=wp.float32, ndim=2),
    mac_a_y: wp.array(dtype=wp.float32, ndim=2),
    rhs: wp.array(dtype=wp.float32, ndim=2),
    dx: float,
    kx: float,
    ky: float,
):
    i, j = wp.tid()

    cell_type[i, j] = wp.uint8(0)  # fluid cell
    diag = 0.0
    mac_a_x[i, j] = 0.0
    mac_a_y[i, j] = 0.0

    if i != 0:
        diag += 1.0
        mac_a_x[i, j] = 1.0
    if i != cell_diag.shape[0] - 1:
        diag += 1.0
    if j != 0:
        diag += 1.0
        mac_a_y[i, j] = 1.0
    if j != cell_diag.shape[1] - 1:
        diag += 1.0

    cell_diag[i, j] = diag

    fact = dx * dx * (kx * kx + ky * ky)
    x = (wp.float32(i) + 0.5) * dx
    y = (wp.float32(j) + 0.5) * dx
    rhs[i, j] = fact * wp.cos(kx * x) * wp.cos(ky * y)


# Neumann BC
# Poisson cos(kx x) * cos(ky y) * cos(kz z)
# [0, 1] x [0, 1] x [0, 1]
@wp.kernel
def test_c1_init_kernel3D(
    cell_type: wp.array(dtype=wp.uint8, ndim=3),
    cell_diag: wp.array(dtype=wp.float32, ndim=3),
    mac_a_x: wp.array(dtype=wp.float32, ndim=3),
    mac_a_y: wp.array(dtype=wp.float32, ndim=3),
    mac_a_z: wp.array(dtype=wp.float32, ndim=3),
    rhs: wp.array(dtype=wp.float32, ndim=3),
    dx: float,
    kx: float,
    ky: float,
    kz: float,
):
    i, j, k = wp.tid()
    cell_type[i, j, k] = wp.uint8(0)  # fluid cell
    diag = 0.0
    mac_a_x[i, j, k] = 0.0
    mac_a_y[i, j, k] = 0.0
    mac_a_z[i, j, k] = 0.0

    if i != 0:
        diag += 1.0
        mac_a_x[i, j, k] = 1.0
    if i != cell_diag.shape[0] - 1:
        diag += 1.0
    if j != 0:
        diag += 1.0
        mac_a_y[i, j, k] = 1.0
    if j != cell_diag.shape[1] - 1:
        diag += 1.0
    if k != 0:
        diag += 1.0
        mac_a_z[i, j, k] = 1.0
    if k != cell_diag.shape[2] - 1:
        diag += 1.0

    cell_diag[i, j, k] = diag

    fact = dx * dx * (kx * kx + ky * ky + kz * kz)
    x = (wp.float32(i) + 0.5) * dx
    y = (wp.float32(j) + 0.5) * dx
    z = (wp.float32(k) + 0.5) * dx
    rhs[i, j, k] = fact * wp.cos(kx * x) * wp.cos(ky * y) * wp.cos(kz * z)


# Dirichlet BC
# u_xx + cy u_yy = f
# sin(kx x) * sin(ky y)
# [0, 1] x [0, 1]
@wp.kernel
def test_c2_init_kernel2D(
    cell_type: wp.array(dtype=wp.uint8, ndim=2),
    cell_diag: wp.array(dtype=wp.float32, ndim=2),
    mac_a_x: wp.array(dtype=wp.float32, ndim=2),
    mac_a_y: wp.array(dtype=wp.float32, ndim=2),
    rhs: wp.array(dtype=wp.float32, ndim=2),
    dx: float,
    kx: float,
    ky: float,
    cy: float,
):
    i, j = wp.tid()

    cell_type[i, j] = wp.uint8(0)  # fluid cell
    cell_diag[i, j] = 2.0 + 2.0 * cy
    if i > 0:
        mac_a_x[i, j] = 1.0
    if j > 0:
        mac_a_y[i, j] = cy

    fact = dx * dx * (kx * kx + cy * ky * ky)
    x = (wp.float32(i) + 1.0) * dx
    y = (wp.float32(j) + 1.0) * dx
    rhs[i, j] = fact * wp.sin(kx * x) * wp.sin(ky * y)


# Dirichlet BC
# uxx+cy uyy + cz uzz = f
# sin(kx x) * sin(ky y) * sin(kz z)
# [0, 1] x [0, 1] x [0, 1]
@wp.kernel
def test_c2_init_kernel3D(
    cell_type: wp.array(dtype=wp.uint8, ndim=3),
    cell_diag: wp.array(dtype=wp.float32, ndim=3),
    mac_a_x: wp.array(dtype=wp.float32, ndim=3),
    mac_a_y: wp.array(dtype=wp.float32, ndim=3),
    mac_a_z: wp.array(dtype=wp.float32, ndim=3),
    rhs: wp.array(dtype=wp.float32, ndim=3),
    dx: float,
    kx: float,
    ky: float,
    kz: float,
    cy: float,
    cz: float,
):
    i, j, k = wp.tid()
    cell_type[i, j, k] = wp.uint8(0)  # fluid cell
    cell_diag[i, j, k] = 2.0 + 2.0 * cy + 2.0 * cz
    if i > 0:
        mac_a_x[i, j, k] = 1.0
    if j > 0:
        mac_a_y[i, j, k] = cy
    if k > 0:
        mac_a_z[i, j, k] = cz

    fact = dx * dx * (kx * kx + cy * ky * ky + cz * kz * kz)
    x = (wp.float32(i) + 1.0) * dx
    y = (wp.float32(j) + 1.0) * dx
    z = (wp.float32(k) + 1.0) * dx
    rhs[i, j, k] = fact * wp.sin(kx * x) * wp.sin(ky * y) * wp.sin(kz * z)


def test2D():
    # basic settings
    dim = 2
    res = [256, 256]
    dx = 1.0 / res[0]

    # initialize warp
    solver = WarpMGPCG(
        dim=dim,
        N=res,
        device="cuda",
    )

    # construct multigrid hierarchy
    all_neumann = False

    def c0_config():
        wp.launch(
            test_c0_init_kernel2D,
            dim=tuple(res),
            inputs=[
                solver.cell_type[0],
                solver.cell_diag[0],
                *solver.mac_alpha[0],  # 2 components in 2D
                solver.r[0],
                dx,
                5.0 * np.pi,
                5.0 * np.pi,
            ],
            device="cuda",
        )

    def c1_config():
        nonlocal all_neumann
        all_neumann = True
        wp.launch(
            test_c1_init_kernel2D,
            dim=tuple(res),
            inputs=[
                solver.cell_type[0],
                solver.cell_diag[0],
                *solver.mac_alpha[0],
                solver.r[0],
                dx,
                5.0 * np.pi,
                5.0 * np.pi,
            ],
            device="cuda",
        )

    def c2_config():
        wp.launch(
            test_c2_init_kernel2D,
            dim=tuple(res),
            inputs=[
                solver.cell_type[0],
                solver.cell_diag[0],
                *solver.mac_alpha[0],
                solver.r[0],
                dx,
                5.0 * np.pi,
                5.0 * np.pi,
                0.01,
            ],
            device="cuda",
        )

    # choose configuration
    c2_config()

    solver.build_coarse_levels()

    # solve
    solver.solve(max_iters=-1, all_neumann=all_neumann)


def test3D():
    # basic settings
    dim = 3
    res = [256, 256, 256]
    dx = 1.0 / res[0]

    # initialize warp
    solver = WarpMGPCG(
        dim=dim,
        N=res,
        device="cuda",
    )

    # construct multigrid hierarchy
    all_neumann = False

    def c0_config():
        wp.launch(
            test_c0_init_kernel3D,
            dim=tuple(res),
            inputs=[
                solver.cell_type[0],
                solver.cell_diag[0],
                *solver.mac_alpha[0],
                solver.r[0],
                dx,
                5.0 * np.pi,
                5.0 * np.pi,
                5.0 * np.pi,
            ],
            device="cuda",
        )

    def c1_config():
        nonlocal all_neumann
        all_neumann = True
        wp.launch(
            test_c1_init_kernel3D,
            dim=tuple(res),
            inputs=[
                solver.cell_type[0],
                solver.cell_diag[0],
                *solver.mac_alpha[0],
                solver.r[0],
                dx,
                5.0 * np.pi,
                5.0 * np.pi,
                5.0 * np.pi,
            ],
            device="cuda",
        )

    def c2_config():
        wp.launch(
            test_c2_init_kernel3D,
            dim=tuple(res),
            inputs=[
                solver.cell_type[0],
                solver.cell_diag[0],
                *solver.mac_alpha[0],
                solver.r[0],
                dx,
                5.0 * np.pi,
                5.0 * np.pi,
                5.0 * np.pi,
                0.01,
                0.001,
            ],
            device="cuda",
        )

    c2_config()
    solver.build_coarse_levels()

    # solve
    solver.solve(all_neumann=all_neumann)


if __name__ == "__main__":
    test2D()
    test3D()
