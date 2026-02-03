import warp as wp
import numpy as np
import time

wp.init()


class WarpMGPCG:
    def __init__(self, dim, N, base_level=3, pre_and_post_smoothing=2, bottom_smoothing=50, device="cuda"):
        """
        Initialize Warp-based MGPCG solver.

        Args:
            dim: Problem dimension (2 or 3).
            N: Grid size in each dimension.
            base_level: Base level for multigrid, (e.g., 3 for 8x8 grid at coarsest level).
            pre_and_post_smoothing: Number of smoothing iterations before and after restriction/prolongation.
            bottom_smoothing: Number of smoothing iterations at the coarsest level.
            device: Device to run the computations on (e.g., "cuda" or "cpu").

        Variables:
            cell_type[level][i, j, k]: Cell types (fluid, solid, etc.).
            cell_diag[level][i, j, k]: Cell diagonal entries.
            mac_alpha[level][d][i, j, k]: Mac edge fluid permeability, ranges from 0 to 1.

        Cut-Cell Discretization Rules:
            cell_type: 0 - fluid, 1 - solid, 2 - empty

            mac_alpha (face permeability):
                fluid–fluid : 1.0
                fluid–solid : fraction in (0, 1]
                fluid–empty : 0.0
                NOTE: equal to -a_x/a_y/a_z

            cell_diag (Poisson diagonal contribution):
                fluid / solid cells:
                    diag += sum(mac_alpha over incident faces)
                empty cells:
                    diag += 1.0   # identity constraint to keep the system well-posed
        """
        self.dim = dim
        self.N = N
        self.n_mg_levels = int(np.log2(min(N))) - base_level + 1
        self.pre_and_post_smoothing = pre_and_post_smoothing
        self.bottom_smoothing = bottom_smoothing
        self.device = device

        # init field
        self.r = [wp.zeros([n // 2**l for n in N], dtype=float, device=device) for l in range(self.n_mg_levels)]  # residual
        self.z = [wp.zeros([n // 2**l for n in N], dtype=float, device=device) for l in range(self.n_mg_levels)]  # M^-1 self.r
        self.x = wp.zeros(N, dtype=float, device=device)  # solution
        self.p = wp.zeros(N, dtype=float, device=device)  # conjugate gradient
        self.Ap = wp.zeros(N, dtype=float, device=device)  # A * p
        self.sum = wp.zeros(1, dtype=float, device=device)  # reduction sum
        self.r_weight = wp.zeros(1, dtype=float, device=device)  # for all-neumann recentering
        self.r_mean = wp.zeros(1, dtype=float, device=device)

        # geometric multigrid data
        self.cell_type = [wp.zeros([n // 2**l for n in N], dtype=wp.uint8, device=device) for l in range(self.n_mg_levels)]
        self.cell_diag = [wp.zeros([n // 2**l for n in N], dtype=float, device=device) for l in range(self.n_mg_levels)]
        self.mac_alpha = []
        for l in range(self.n_mg_levels):
            curr_N = [max(1, n // (2**l)) for n in N]
            components = []
            for d in range(dim):
                shape = list(curr_N)
                shape[d] += 1
                components.append(wp.zeros(shape, dtype=float, device=device))
            self.mac_alpha.append(components)

        # print variables info
        print(
            f"[SOLVER] Warp MGPCG initialization:\n"
            f"  Dimension: {dim}D\n"
            f"  Grid Size: {N}\n"
            f"  Multigrid Levels: {self.n_mg_levels}\n"
            f"  Pre/Post Smoothing: {pre_and_post_smoothing}\n"
            f"  Bottom Smoothing: {bottom_smoothing}\n"
            f"  Device: {device}\n"
        )

        # 2D/3D function selection
        if dim == 2:
            self.build_coarse_cell_kernel = self.build_coarse_cell_kernel2D
            self.compute_Ap_kernel = self.compute_Ap_kernel2D
            self.restrict_kernel = self.restrict_kernel2D
            self.prolongate_kernel = self.prolongate_kernel2D
            self.smooth_kernel = self.smooth_kernel2D
            self.AXPBY_kernel = self.AXPBY_kernel2D
            self.AXPY_kernel = self.AXPY_kernel2D
            self.reduce_kernel = self.reduce_kernel2D
            self.mean_kernel = self.mean_kernel2D
            self.recenter_kernel = self.recenter_kernel2D
        elif dim == 3:
            self.build_coarse_cell_kernel = self.build_coarse_cell_kernel3D
            self.compute_Ap_kernel = self.compute_Ap_kernel3D
            self.restrict_kernel = self.restrict_kernel3D
            self.prolongate_kernel = self.prolongate_kernel3D
            self.smooth_kernel = self.smooth_kernel3D
            self.AXPBY_kernel = self.AXPBY_kernel3D
            self.AXPY_kernel = self.AXPY_kernel3D
            self.reduce_kernel = self.reduce_kernel3D
            self.mean_kernel = self.mean_kernel3D
            self.recenter_kernel = self.recenter_kernel3D

        # ? debug
        print("cell_type.shape:", [ct.shape for ct in self.cell_type])
        print("cell_diag.shape:", [cd.shape for cd in self.cell_diag])
        print("mac_alpha.shape:", [[ma[d].shape for d in range(dim)] for ma in self.mac_alpha])

    # * Geometric Multigrid Data Construction
    @wp.kernel
    def build_coarse_cell_kernel2D(
        cell_type_f: wp.array(dtype=wp.uint8, ndim=2),
        cell_diag_f: wp.array(dtype=float, ndim=2),
        mac_a_x_f: wp.array(dtype=float, ndim=2),
        mac_a_y_f: wp.array(dtype=float, ndim=2),
        cell_type_c: wp.array(dtype=wp.uint8, ndim=2),
        cell_diag_c: wp.array(dtype=float, ndim=2),
        mac_a_x_c: wp.array(dtype=float, ndim=2),
        mac_a_y_c: wp.array(dtype=float, ndim=2),
    ):
        i, j = wp.tid()

        sum_type = wp.uint8(0)
        sum_diag = 0.0

        for di in range(2):
            for dj in range(2):
                fi = 2 * i + di
                fj = 2 * j + dj
                sum_type += cell_type_f[fi, fj]
                sum_diag += cell_diag_f[fi, fj]

        cell_type_c[i, j] = wp.uint8(1) if sum_type == 4 else wp.uint8(0)
        cell_diag_c[i, j] = sum_diag * 0.25

        # MAC alpha
        ax, ay = 0.0, 0.0
        for dt in range(2):
            ax += 0.5 * mac_a_x_f[2 * i, 2 * j + dt]
            ay += 0.5 * mac_a_y_f[2 * i + dt, 2 * j]
        mac_a_x_c[i, j] = ax
        mac_a_y_c[i, j] = ay

        ax, ay = 0.0, 0.0
        if i == cell_type_c.shape[0] - 1:
            for dt in range(2):
                ax += 0.5 * mac_a_x_f[2 * i + 2, 2 * j + dt]
            mac_a_x_c[i + 1, j] = ax

        if j == cell_type_c.shape[1] - 1:
            for dt in range(2):
                ay += 0.5 * mac_a_y_f[2 * i + dt, 2 * j + 2]
            mac_a_y_c[i, j + 1] = ay

    @wp.kernel
    def build_coarse_cell_kernel3D(
        cell_type_f: wp.array(dtype=wp.uint8, ndim=3),
        cell_diag_f: wp.array(dtype=float, ndim=3),
        mac_a_x_f: wp.array(dtype=float, ndim=3),
        mac_a_y_f: wp.array(dtype=float, ndim=3),
        mac_a_z_f: wp.array(dtype=float, ndim=3),
        cell_type_c: wp.array(dtype=wp.uint8, ndim=3),
        cell_diag_c: wp.array(dtype=float, ndim=3),
        mac_a_x_c: wp.array(dtype=float, ndim=3),
        mac_a_y_c: wp.array(dtype=float, ndim=3),
        mac_a_z_c: wp.array(dtype=float, ndim=3),
    ):
        i, j, k = wp.tid()

        sum_type = wp.uint8(0)
        sum_diag = 0.0

        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    fi = 2 * i + di
                    fj = 2 * j + dj
                    fk = 2 * k + dk
                    sum_type += cell_type_f[fi, fj, fk]
                    sum_diag += cell_diag_f[fi, fj, fk]

        cell_type_c[i, j, k] = wp.uint8(1) if sum_type == 8 else wp.uint8(0)
        cell_diag_c[i, j, k] = sum_diag * 0.125

        # MAC alpha
        ax, ay, az = 0.0, 0.0, 0.0
        for dt in range(2):
            for ds in range(2):
                ax += 0.25 * mac_a_x_f[2 * i, 2 * j + ds, 2 * k + dt]
                ay += 0.25 * mac_a_y_f[2 * i + dt, 2 * j, 2 * k + ds]
                az += 0.25 * mac_a_z_f[2 * i + ds, 2 * j + dt, 2 * k]
        mac_a_x_c[i, j, k] = ax
        mac_a_y_c[i, j, k] = ay
        mac_a_z_c[i, j, k] = az

        ax, ay, az = 0.0, 0.0, 0.0
        if i == cell_type_c.shape[0] - 1:
            for ds in range(2):
                for dt in range(2):
                    ax += 0.25 * mac_a_x_f[2 * i + 2, 2 * j + ds, 2 * k + dt]
            mac_a_x_c[i + 1, j, k] = ax

        if j == cell_type_c.shape[1] - 1:
            for ds in range(2):
                for dt in range(2):
                    ay += 0.25 * mac_a_y_f[2 * i + dt, 2 * j + 2, 2 * k + ds]
            mac_a_y_c[i, j + 1, k] = ay

        if k == cell_type_c.shape[2] - 1:
            for ds in range(2):
                for dt in range(2):
                    az += 0.25 * mac_a_z_f[2 * i + ds, 2 * j + dt, 2 * k + 2]
            mac_a_z_c[i, j, k + 1] = az

    def build_coarse_levels(self):
        """
        Build coarse level grids from finest level using multigrid hierarchy.
        Constructs cell_type, cell_diag, and mac_alpha for all coarse levels.
        """
        for l in range(self.n_mg_levels - 1):
            coarse_shape = self.cell_type[l + 1].shape
            wp.launch(
                kernel=self.build_coarse_cell_kernel,
                dim=coarse_shape,
                inputs=[
                    self.cell_type[l],
                    self.cell_diag[l],
                    *self.mac_alpha[l],
                    self.cell_type[l + 1],
                    self.cell_diag[l + 1],
                    *self.mac_alpha[l + 1],
                ],
                device=self.device,
            )

    # * MGPCG Kernels
    @wp.kernel
    def compute_Ap_kernel2D(
        p: wp.array(dtype=float, ndim=2),
        Ap: wp.array(dtype=float, ndim=2),
        cell_type: wp.array(dtype=wp.uint8, ndim=2),
        cell_diag: wp.array(dtype=float, ndim=2),
        mac_a_x: wp.array(dtype=float, ndim=2),
        mac_a_y: wp.array(dtype=float, ndim=2),
    ):
        i, j = wp.tid()

        if cell_type[i, j] == 0:  # fluid cell
            Ap[i, j] = cell_diag[i, j] * p[i, j] - neighbor_sum_func2D(i, j, p, mac_a_x, mac_a_y)

    @wp.kernel
    def compute_Ap_kernel3D(
        p: wp.array(dtype=float, ndim=3),
        Ap: wp.array(dtype=float, ndim=3),
        cell_type: wp.array(dtype=wp.uint8, ndim=3),
        cell_diag: wp.array(dtype=float, ndim=3),
        mac_a_x: wp.array(dtype=float, ndim=3),
        mac_a_y: wp.array(dtype=float, ndim=3),
        mac_a_z: wp.array(dtype=float, ndim=3),
    ):
        i, j, k = wp.tid()

        if cell_type[i, j, k] == 0:  # fluid cell
            Ap[i, j, k] = cell_diag[i, j, k] * p[i, j, k] - neighbor_sum_func3D(i, j, k, p, mac_a_x, mac_a_y, mac_a_z)

    @wp.kernel
    def restrict_kernel2D(
        r_f: wp.array(dtype=float, ndim=2),
        z_f: wp.array(dtype=float, ndim=2),
        r_c: wp.array(dtype=float, ndim=2),
        cell_type_f: wp.array(dtype=wp.uint8, ndim=2),
        cell_diag_f: wp.array(dtype=float, ndim=2),
        mac_a_x_f: wp.array(dtype=float, ndim=2),
        mac_a_y_f: wp.array(dtype=float, ndim=2),
    ):
        i, j = wp.tid()

        sum_val = 0.0
        for di in range(2):
            for dj in range(2):
                fi = 2 * i + di
                fj = 2 * j + dj
                if cell_type_f[fi, fj] == 0:  # fluid cell
                    res = cell_diag_f[fi, fj] * z_f[fi, fj] - neighbor_sum_func2D(fi, fj, z_f, mac_a_x_f, mac_a_y_f)
                    sum_val += r_f[fi, fj] - res

        r_c[i, j] = sum_val / 1.0  # sum_val / (dim - 1)

    @wp.kernel
    def restrict_kernel3D(
        r_f: wp.array(dtype=float, ndim=3),
        z_f: wp.array(dtype=float, ndim=3),
        r_c: wp.array(dtype=float, ndim=3),
        cell_type_f: wp.array(dtype=wp.uint8, ndim=3),
        cell_diag_f: wp.array(dtype=float, ndim=3),
        mac_a_x_f: wp.array(dtype=float, ndim=3),
        mac_a_y_f: wp.array(dtype=float, ndim=3),
        mac_a_z_f: wp.array(dtype=float, ndim=3),
    ):
        i, j, k = wp.tid()

        sum_val = 0.0
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    fi = 2 * i + di
                    fj = 2 * j + dj
                    fk = 2 * k + dk
                    if cell_type_f[fi, fj, fk] == 0:  # fluid cell
                        res = cell_diag_f[fi, fj, fk] * z_f[fi, fj, fk] - neighbor_sum_func3D(fi, fj, fk, z_f, mac_a_x_f, mac_a_y_f, mac_a_z_f)
                        sum_val += r_f[fi, fj, fk] - res

        r_c[i, j, k] = sum_val / 2.0  # sum_val / (dim - 1)

    @wp.kernel
    def prolongate_kernel2D(
        z_c: wp.array(dtype=float, ndim=2),
        z_f: wp.array(dtype=float, ndim=2),
        cell_type_f: wp.array(dtype=wp.uint8, ndim=2),
    ):
        i, j = wp.tid()

        if cell_type_f[i, j] == 0:  # fluid cell
            ci = i // 2
            cj = j // 2
            z_f[i, j] += z_c[ci, cj]

    @wp.kernel
    def prolongate_kernel3D(
        z_c: wp.array(dtype=float, ndim=3),
        z_f: wp.array(dtype=float, ndim=3),
        cell_type_f: wp.array(dtype=wp.uint8, ndim=3),
    ):
        i, j, k = wp.tid()

        if cell_type_f[i, j, k] == 0:  # fluid cell
            ci = i // 2
            cj = j // 2
            ck = k // 2
            z_f[i, j, k] += z_c[ci, cj, ck]

    @wp.kernel
    def smooth_kernel2D(
        r: wp.array(dtype=float, ndim=2),
        z: wp.array(dtype=float, ndim=2),
        cell_type: wp.array(dtype=wp.uint8, ndim=2),
        cell_diag: wp.array(dtype=float, ndim=2),
        mac_a_x: wp.array(dtype=float, ndim=2),
        mac_a_y: wp.array(dtype=float, ndim=2),
        phase: int,
    ):
        """
        Red/Black Gauss-Seidel Smoothing for 2D grids.
        """
        i, j = wp.tid()
        if ((i + j) & 1) == phase and cell_type[i, j] == 0:  # fluid cell
            neighbor_sum = neighbor_sum_func2D(i, j, z, mac_a_x, mac_a_y)
            z[i, j] = (r[i, j] + neighbor_sum) / cell_diag[i, j]

    @wp.kernel
    def smooth_kernel3D(
        r: wp.array(dtype=float, ndim=3),
        z: wp.array(dtype=float, ndim=3),
        cell_type: wp.array(dtype=wp.uint8, ndim=3),
        cell_diag: wp.array(dtype=float, ndim=3),
        mac_a_x: wp.array(dtype=float, ndim=3),
        mac_a_y: wp.array(dtype=float, ndim=3),
        mac_a_z: wp.array(dtype=float, ndim=3),
        phase: int,
    ):
        """
        Red/Black Gauss-Seidel Smoothing for 3D grids.
        """
        i, j, k = wp.tid()
        if ((i + j + k) & 1) == phase and cell_type[i, j, k] == 0:  # fluid cell
            neighbor_sum = neighbor_sum_func3D(i, j, k, z, mac_a_x, mac_a_y, mac_a_z)
            z[i, j, k] = (r[i, j, k] + neighbor_sum) / cell_diag[i, j, k]

    @wp.kernel
    def AXPBY_kernel2D(
        x: wp.array(dtype=float, ndim=2),
        y: wp.array(dtype=float, ndim=2),
        z: wp.array(dtype=float, ndim=2),
        a: float,
        b: float,
        cell_type: wp.array(dtype=wp.uint8, ndim=2),
    ):
        i, j = wp.tid()
        if cell_type[i, j] == 0:  # fluid cell
            z[i, j] = a * x[i, j] + b * y[i, j]

    @wp.kernel
    def AXPBY_kernel3D(
        x: wp.array(dtype=float, ndim=3),
        y: wp.array(dtype=float, ndim=3),
        z: wp.array(dtype=float, ndim=3),
        a: float,
        b: float,
        cell_type: wp.array(dtype=wp.uint8, ndim=3),
    ):
        i, j, k = wp.tid()
        if cell_type[i, j, k] == 0:  # fluid cell
            z[i, j, k] = a * x[i, j, k] + b * y[i, j, k]

    @wp.kernel
    def AXPY_kernel2D(
        x: wp.array(dtype=float, ndim=2),
        y: wp.array(dtype=float, ndim=2),
        a: float,
        b: float,
        cell_type: wp.array(dtype=wp.uint8, ndim=2),
    ):
        i, j = wp.tid()
        if cell_type[i, j] == 0:  # fluid cell
            y[i, j] = a * x[i, j] + b * y[i, j]

    @wp.kernel
    def AXPY_kernel3D(
        x: wp.array(dtype=float, ndim=3),
        y: wp.array(dtype=float, ndim=3),
        a: float,
        b: float,
        cell_type: wp.array(dtype=wp.uint8, ndim=3),
    ):
        i, j, k = wp.tid()
        if cell_type[i, j, k] == 0:  # fluid cell
            y[i, j, k] = a * x[i, j, k] + b * y[i, j, k]

    @wp.kernel
    def reduce_kernel2D(
        p: wp.array(dtype=float, ndim=2),
        r: wp.array(dtype=float, ndim=2),
        sum: wp.array(dtype=float, ndim=1),
        cell_type: wp.array(dtype=wp.uint8, ndim=2),
    ):
        i, j = wp.tid()
        if cell_type[i, j] == 0:  # fluid cell
            sum[0] += p[i, j] * r[i, j]

    @wp.kernel
    def reduce_kernel3D(
        p: wp.array(dtype=float, ndim=3),
        r: wp.array(dtype=float, ndim=3),
        sum: wp.array(dtype=float, ndim=1),
        cell_type: wp.array(dtype=wp.uint8, ndim=3),
    ):
        i, j, k = wp.tid()
        if cell_type[i, j, k] == 0:  # fluid cell
            sum[0] += p[i, j, k] * r[i, j, k]

    @wp.kernel
    def mean_kernel2D(
        r: wp.array(dtype=float, ndim=2),
        r_mean: wp.array(dtype=float, ndim=1),
        r_weight: wp.array(dtype=float, ndim=1),
        cell_type: wp.array(dtype=wp.uint8, ndim=2),
    ):
        i, j = wp.tid()
        if cell_type[i, j] == 0:  # fluid cell
            r_mean[0] += r[i, j]
            r_weight[0] += 1.0

    @wp.kernel
    def mean_kernel3D(
        r: wp.array(dtype=float, ndim=3),
        r_mean: wp.array(dtype=float, ndim=1),
        r_weight: wp.array(dtype=float, ndim=1),
        cell_type: wp.array(dtype=wp.uint8, ndim=3),
    ):
        i, j, k = wp.tid()
        if cell_type[i, j, k] == 0:  # fluid cell
            r_mean[0] += r[i, j, k]
            r_weight[0] += 1.0

    @wp.kernel
    def recenter_kernel2D(
        r: wp.array(dtype=float, ndim=2),
        r_mean: wp.array(dtype=float, ndim=1),
        r_weight: wp.array(dtype=float, ndim=1),
        cell_type: wp.array(dtype=wp.uint8, ndim=2),
    ):
        i, j = wp.tid()
        if cell_type[i, j] == 0:  # fluid cell
            r[i, j] -= r_mean[0] / r_weight[0]

    @wp.kernel
    def recenter_kernel3D(
        r: wp.array(dtype=float, ndim=3),
        r_mean: wp.array(dtype=float, ndim=1),
        r_weight: wp.array(dtype=float, ndim=1),
        cell_type: wp.array(dtype=wp.uint8, ndim=3),
    ):
        i, j, k = wp.tid()
        if cell_type[i, j, k] == 0:  # fluid cell
            r[i, j, k] -= r_mean[0] / r_weight[0]

    # * MGPCG Solver Functions
    def setup_system(self, rhs, cell_type=None, cell_diag=None, mac_alpha=None, build_coarse=True):
        """
        Setup the linear system Ax = b for MGPCG solver.

        :parameter rhs: Right-hand side vector b.
        :parameter cell_type: Cell type grid.
        :parameter cell_diag: Cell diagonal entries grid.
        :parameter mac_alpha: Mac edge fluid permeability grids.

        NOTE: Geometric multigrid data can be modified externally or passed in.

        """
        # set rhs
        wp.copy(self.r[0], rhs)

        # set initial guess
        self.x.zero_()

        # set geometric multigrid data
        if cell_type is not None:
            wp.copy(self.cell_type[0], cell_type)
        if cell_diag is not None:
            wp.copy(self.cell_diag[0], cell_diag)
        if mac_alpha is not None:
            for d in range(self.dim):
                wp.copy(self.mac_alpha[0][d], mac_alpha[d])

        # build coarse levels
        if build_coarse:
            self.build_coarse_levels()

    def get_solution(self, x):
        wp.copy(x, self.x)

    def apply_preconditioner(self):
        # * No preconditioning
        # wp.copy(self.z[0], self.r[0])

        # * MG V-Cycle
        self.z[0].zero_()
        for l in range(self.n_mg_levels - 1):
            # Pre-smoothing
            for _ in range(2):
                wp.launch(
                    self.smooth_kernel,
                    dim=self.z[l].shape,
                    inputs=[self.r[l], self.z[l], self.cell_type[l], self.cell_diag[l], *self.mac_alpha[l], 0],
                    device=self.device,
                )
                wp.launch(
                    self.smooth_kernel,
                    dim=self.z[l].shape,
                    inputs=[self.r[l], self.z[l], self.cell_type[l], self.cell_diag[l], *self.mac_alpha[l], 1],
                    device=self.device,
                )
            # Restriction
            self.z[l + 1].zero_()
            self.r[l + 1].zero_()
            wp.launch(
                self.restrict_kernel,
                dim=self.r[l + 1].shape,
                inputs=[self.r[l], self.z[l], self.r[l + 1], self.cell_type[l], self.cell_diag[l], *self.mac_alpha[l]],
                device=self.device,
            )

        # Bottom Smoothing
        for _ in range(10):
            wp.launch(
                self.smooth_kernel,
                dim=self.z[-1].shape,
                inputs=[self.r[-1], self.z[-1], self.cell_type[-1], self.cell_diag[-1], *self.mac_alpha[-1], 0],
                device=self.device,
            )
            wp.launch(
                self.smooth_kernel,
                dim=self.z[-1].shape,
                inputs=[self.r[-1], self.z[-1], self.cell_type[-1], self.cell_diag[-1], *self.mac_alpha[-1], 1],
                device=self.device,
            )

        for l in reversed(range(self.n_mg_levels - 1)):
            # Prolongation
            wp.launch(
                self.prolongate_kernel,
                dim=self.z[l].shape,
                inputs=[self.z[l + 1], self.z[l], self.cell_type[l]],
                device=self.device,
            )
            # Post-smoothing
            for _ in range(2):
                wp.launch(
                    self.smooth_kernel,
                    dim=self.z[l].shape,
                    inputs=[self.r[l], self.z[l], self.cell_type[l], self.cell_diag[l], *self.mac_alpha[l], 0],
                    device=self.device,
                )
                wp.launch(
                    self.smooth_kernel,
                    dim=self.z[l].shape,
                    inputs=[self.r[l], self.z[l], self.cell_type[l], self.cell_diag[l], *self.mac_alpha[l], 1],
                    device=self.device,
                )

    def solve(self, max_iters=400, eps=1e-12, abs_tol=1e-6, rel_tol=1e-6, all_neumann=False, verbose=True):
        """
        Solve the linear system using MGPCG.

        :parameter max_iters: Specify the maximal iterations. -1 for no limit.
        :parameter eps: Specify a non-zero value to prevent ZeroDivisionError.
        :parameter abs_tol: Specify the absolute tolerance of loss.
        :parameter rel_tol: Specify the tolerance of loss relative to initial loss.
        :parameter all_neumann: Whether the system has all-Neumann boundary conditions.
        """

        # * Initialization
        # init alpha, beta
        alpha, beta = 0.0, 0.0

        # recenter for all-neumann system
        if all_neumann:
            self.r_weight.zero_()
            self.r_mean.zero_()
            wp.launch(
                self.mean_kernel,
                dim=self.r[0].shape,
                inputs=[self.r[0], self.r_mean, self.r_weight, self.cell_type[0]],
                device=self.device,
            )
            wp.launch(
                self.recenter_kernel,
                dim=self.r[0].shape,
                inputs=[self.r[0], self.r_mean, self.r_weight, self.cell_type[0]],
                device=self.device,
            )

        # p = z[0] = P^-1 r[0]
        self.apply_preconditioner()
        wp.copy(self.p, self.z[0])

        # old_zTr = z[0]^T r[0]
        self.sum.zero_()
        wp.launch(
            self.reduce_kernel,
            dim=self.z[0].shape,
            inputs=[self.z[0], self.r[0], self.sum, self.cell_type[0]],
            device=self.device,
        )
        old_zTr = self.sum.numpy()[0]

        # set tolerance
        self.sum.zero_()
        wp.launch(
            self.reduce_kernel,
            dim=self.r[0].shape,
            inputs=[self.r[0], self.r[0], self.sum, self.cell_type[0]],
            device=self.device,
        )
        rTr_sqr0 = wp.sqrt(self.sum.numpy()[0])
        tol = max(abs_tol, rel_tol * rTr_sqr0)

        # * PCG Iteration
        it = 0
        start_t = time.time()
        while max_iters == -1 or it < max_iters:
            # Step1: alpha = z[0]^T r[0] / p^T Ap
            wp.launch(
                self.compute_Ap_kernel,
                dim=self.p.shape,
                inputs=[self.p, self.Ap, self.cell_type[0], self.cell_diag[0], *self.mac_alpha[0]],
                device=self.device,
            )  # Ap = A * p
            self.sum.zero_()
            wp.launch(
                self.reduce_kernel,
                dim=self.p.shape,
                inputs=[self.p, self.Ap, self.sum, self.cell_type[0]],
                device=self.device,
            )  # pAp = p^T Ap
            pAp = self.sum.numpy()[0]
            alpha = old_zTr / (pAp + eps)

            # Step2: x = x + alpha p
            wp.launch(
                self.AXPY_kernel,
                dim=self.x.shape,
                inputs=[self.p, self.x, alpha, 1.0, self.cell_type[0]],
                device=self.device,
            )

            # Step3: r = r - alpha Ap
            wp.launch(
                self.AXPY_kernel,
                dim=self.r[0].shape,
                inputs=[self.Ap, self.r[0], -alpha, 1.0, self.cell_type[0]],
                device=self.device,
            )

            # Step4: check for convergence
            self.sum.zero_()
            wp.launch(
                self.reduce_kernel,
                dim=self.r[0].shape,
                inputs=[self.r[0], self.r[0], self.sum, self.cell_type[0]],
                device=self.device,
            )
            rTr_sqr = wp.sqrt(self.sum.numpy()[0])

            abs_err = rTr_sqr
            rel_err = rTr_sqr / (rTr_sqr0 + eps)
            if verbose:
                print(f"iter {it}, " f"abs_err={abs_err:.3e}, " f"rel_err={rel_err:.3e}")

            if abs_err < abs_tol or rel_err < rel_tol:
                wp.synchronize()
                end_t = time.time()

                if verbose:
                    reason = "abs_tol" if abs_err < abs_tol else "rel_tol"
                    print(
                        f"[MGPCG] Converged at iter {it} "
                        f"(by {reason}), "
                        f"abs_err={abs_err:.3e}, "
                        f"rel_err={rel_err:.3e}, "
                        f"time={end_t - start_t:.3f}s"
                    )
                return

            # recenter for all-neumann system
            if all_neumann:
                self.r_weight.zero_()
                self.r_mean.zero_()
                wp.launch(
                    self.mean_kernel,
                    dim=self.r[0].shape,
                    inputs=[self.r[0], self.r_mean, self.r_weight, self.cell_type[0]],
                    device=self.device,
                )
                wp.launch(
                    self.recenter_kernel,
                    dim=self.r[0].shape,
                    inputs=[self.r[0], self.r_mean, self.r_weight, self.cell_type[0]],
                    device=self.device,
                )

            # Step5: z[0] = P^-1 r[0]
            self.apply_preconditioner()

            # Step6: self.beta = new_rTr / old_rTr
            self.sum.zero_()
            wp.launch(
                self.reduce_kernel,
                dim=self.z[0].shape,
                inputs=[self.z[0], self.r[0], self.sum, self.cell_type[0]],
                device=self.device,
            )
            new_zTr = self.sum.numpy()[0]
            beta = new_zTr / (old_zTr + eps)

            # Step7: p = beta * p + z[0]
            wp.launch(
                self.AXPY_kernel,
                dim=self.p.shape,
                inputs=[self.z[0], self.p, 1.0, beta, self.cell_type[0]],
                device=self.device,
            )

            old_zTr = new_zTr
            it += 1

        wp.synchronize()
        end_t = time.time()
        if verbose:
            print(
                "[MGPCG] Return without converging at iter: ",
                it,
                f"abs_err={abs_err:.3e}, ",
                f"rel_err={rel_err:.3e}, ",
                f"time={end_t - start_t:.3f}s",
            )


@wp.func
def neighbor_sum_func2D(
    i: int,
    j: int,
    p: wp.array(dtype=float, ndim=2),
    mac_a_x: wp.array(dtype=float, ndim=2),
    mac_a_y: wp.array(dtype=float, ndim=2),
) -> float:
    sum_val = 0.0

    if i > 0:
        sum_val += mac_a_x[i, j] * p[i - 1, j]
    if i < p.shape[0] - 1:
        sum_val += mac_a_x[i + 1, j] * p[i + 1, j]

    if j > 0:
        sum_val += mac_a_y[i, j] * p[i, j - 1]
    if j < p.shape[1] - 1:
        sum_val += mac_a_y[i, j + 1] * p[i, j + 1]

    return sum_val


@wp.func
def neighbor_sum_func3D(
    i: int,
    j: int,
    k: int,
    p: wp.array(dtype=float, ndim=3),
    mac_a_x: wp.array(dtype=float, ndim=3),
    mac_a_y: wp.array(dtype=float, ndim=3),
    mac_a_z: wp.array(dtype=float, ndim=3),
) -> float:
    sum_val = 0.0

    if i > 0:
        sum_val += mac_a_x[i, j, k] * p[i - 1, j, k]
    if i < p.shape[0] - 1:
        sum_val += mac_a_x[i + 1, j, k] * p[i + 1, j, k]

    if j > 0:
        sum_val += mac_a_y[i, j, k] * p[i, j - 1, k]
    if j < p.shape[1] - 1:
        sum_val += mac_a_y[i, j + 1, k] * p[i, j + 1, k]

    if k > 0:
        sum_val += mac_a_z[i, j, k] * p[i, j, k - 1]
    if k < p.shape[2] - 1:
        sum_val += mac_a_z[i, j, k + 1] * p[i, j, k + 1]

    return sum_val
