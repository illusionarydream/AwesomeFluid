import numpy as np
import trimesh
import mesh_to_sdf


def load_levelset_from_obj(phi, dx, file_path, save_path=None):
    # Load mesh
    scene = trimesh.load(file_path)
    if isinstance(scene, trimesh.Scene):
        mesh = list(scene.geometry.values())[0]
    else:
        mesh = scene

    # Create grid points
    nx, ny, nz = phi.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    z = np.arange(nz) * dx
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Compute SDF
    sdf = mesh_to_sdf.mesh_to_sdf(
        mesh,
        points,
        surface_point_method="scan",
        sign_method="depth",
    )
    sdf = sdf.reshape(phi.shape)

    # Store in phi
    phi[:] = sdf

    # Save
    if save_path:
        np.save(save_path, sdf)

        # Optional: write VTK if needed
        # write_vtk.write_grid_vtk(
        #     sdf.shape,
        #     dx,
        #     {"phi": phi},
        #     save_path.replace(".npy", ""),
        # )


if __name__ == "__main__":
    N = 256
    dx = 1.0 / N
    res = (N, N, N)
    name = "bunny"  # Change to desired model name
    phi = np.zeros(res, dtype=np.float32)
    load_levelset_from_obj(phi, dx, f"{name}.obj", f"{name}.npy")
    print(f"Saved {name}.npy")
