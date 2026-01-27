import taichi as ti
import numpy as np


def write_scalar_field_vtk(scalar_field, dx, filename):
    try:
        from pyevtk.hl import gridToVTK  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise RuntimeError(
            "Failed to import pyevtk. Please install it via /\
        `pip install pyevtk` first. "
        )

    scalar_field_np = scalar_field.to_numpy()
    field_shape = scalar_field_np.shape
    dimension = len(field_shape)

    if dimension not in (2, 3):
        raise ValueError("The input field must be a 2D or 3D scalar field.")

    if dimension == 2:
        scalar_field_np = scalar_field_np[:, :, np.newaxis]
        zcoords = np.array([0]) * dx
    elif dimension == 3:
        zcoords = np.arange(0, field_shape[2]) * dx

    x = np.arange(0, field_shape[0]) * dx
    y = np.arange(0, field_shape[1]) * dx
    z = zcoords
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    gridToVTK(
        filename,
        x=xx,
        y=yy,
        z=zz,
        # cellData={filename: scalar_field_np},
        pointData={"scalar": scalar_field_np},
    )


def write_grid_vtk(res, dx, grid_dict, filename, verbose=False):
    try:
        from pyevtk.hl import gridToVTK  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise RuntimeError(
            "Failed to import pyevtk. Please install it via /\
        `pip install pyevtk` first. "
        )

    dimension = len(res)

    if dimension not in (2, 3):
        raise ValueError("The input field must be a 2D or 3D scalar field.")

    data = {}
    for k, v in grid_dict.items():
        v_np = v
        if not isinstance(v_np, np.ndarray):
            v_np = v_np.to_numpy()

        if dimension == 2:
            if v_np.ndim == dimension:
                v_np = v_np[:, :, np.newaxis]
            elif v_np.ndim == dimension + 1:
                v_np = v_np[:, :, np.newaxis, :]
            elif v_np.ndim == dimension + 2:
                v_np = v_np[:, :, np.newaxis, :, :]

        result = v_np
        # write 3d vector
        if v_np.ndim == 3 + 1 and v_np.shape[-1] == 3:
            result = (np.ascontiguousarray(v_np[..., 0]), np.ascontiguousarray(v_np[..., 1]), np.ascontiguousarray(v_np[..., 2]))
            data[k] = result
            if verbose:
                print('write_grid_vtk: parse vector feature, key="', k, '", value shape=', v_np.shape)
        # write 2d vector
        if v_np.ndim == 3 + 1 and v_np.shape[-1] == 2:
            result = (np.ascontiguousarray(v_np[..., 0]), np.ascontiguousarray(v_np[..., 1]), np.zeros_like(v_np[..., 0]))
            data[k] = result
            if verbose:
                print('write_grid_vtk: parse vector feature, key="', k, '", value shape=', v_np.shape)
        # write 2x2 tensor
        elif v_np.ndim == 3 + 2 and v_np.shape[-2] == 2 and v_np.shape[-1] == 2:
            result = (np.ascontiguousarray(v_np[..., 0, 0]), np.ascontiguousarray(v_np[..., 0, 1]), np.zeros_like(v_np[..., 0, 0]))
            data[k + "x"] = result
            result = (np.ascontiguousarray(v_np[..., 1, 0]), np.ascontiguousarray(v_np[..., 1, 1]), np.zeros_like(v_np[..., 0, 0]))
            data[k + "y"] = result
            result = (np.zeros_like(v_np[..., 0, 0]), np.zeros_like(v_np[..., 0, 0]), np.ones_like(v_np[..., 0, 0]))
            data[k + "z"] = result
            if verbose:
                print('write_grid_vtk: parse tensor feature, key="', k, '", value shape=', v_np.shape)
        # write 3x3 tensor
        elif v_np.ndim == 3 + 2 and v_np.shape[-2] == 3 and v_np.shape[-1] == 3:
            result = (np.ascontiguousarray(v_np[..., 0, 0]), np.ascontiguousarray(v_np[..., 1, 0]), np.ascontiguousarray(v_np[..., 2, 0]))
            data[k + "x"] = result
            result = (np.ascontiguousarray(v_np[..., 0, 1]), np.ascontiguousarray(v_np[..., 1, 1]), np.ascontiguousarray(v_np[..., 2, 1]))
            data[k + "y"] = result
            result = (np.ascontiguousarray(v_np[..., 0, 2]), np.ascontiguousarray(v_np[..., 1, 2]), np.ascontiguousarray(v_np[..., 2, 2]))
            data[k + "z"] = result
            if verbose:
                print('write_grid_vtk: parse tensor feature, key="', k, '", value shape=', v_np.shape)
        else:
            if verbose:
                print('write_particle_vtk: parse scalar feature, key="', k, '", value shape=', v_np.shape)

        if v_np.ndim != 3 + 2:
            data[k] = result

    # write grid x
    if dimension == 2:
        zcoords = np.array([0]) * dx
    elif dimension == 3:
        zcoords = np.arange(0, res[2]) * dx + 0.5 * dx

    x = np.arange(0, res[0]) * dx + 0.5 * dx
    y = np.arange(0, res[1]) * dx + 0.5 * dx
    z = zcoords
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    gridToVTK(
        filename,
        x=xx,
        y=yy,
        z=zz,
        # cellData={filename: scalar_field_np},
        pointData=data,
    )


def write_mac_grid_vtk(res, dx, grid_dicts, filenames, verbose=False):
    try:
        from pyevtk.hl import gridToVTK  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise RuntimeError(
            "Failed to import pyevtk. Please install it via /\
        `pip install pyevtk` first. "
        )

    dimension = len(res)

    if dimension not in (2, 3):
        raise ValueError("The input field must be a 2D or 3D scalar field.")

    data = [{}, {}, {}]
    for d in np.arange(len(grid_dicts)):
        grid_dict = grid_dicts[d]
        for k, v in grid_dict.items():
            v_np = v
            if not isinstance(v_np, np.ndarray):
                v_np = v_np.to_numpy()

            if dimension == 2:
                if v_np.ndim == dimension:
                    v_np = v_np[:, :, np.newaxis]
                elif v_np.ndim == dimension + 1:
                    v_np = v_np[:, :, np.newaxis, :]
                elif v_np.ndim == dimension + 2:
                    v_np = v_np[:, :, np.newaxis, :, :]

            result = v_np
            # write 3d vector
            if v_np.ndim == 3 + 1 and v_np.shape[-1] == 3:
                result = (np.ascontiguousarray(v_np[..., 0]), np.ascontiguousarray(v_np[..., 1]), np.ascontiguousarray(v_np[..., 2]))
                data[d][k] = result
                if verbose:
                    print('write_grid_vtk: parse vector feature, key="', k, '", value shape=', v_np.shape)
            # write 2d vector
            if v_np.ndim == 3 + 1 and v_np.shape[-1] == 2:
                result = (np.ascontiguousarray(v_np[..., 0]), np.ascontiguousarray(v_np[..., 1]), np.zeros_like(v_np[..., 0]))
                data[d][k] = result
                if verbose:
                    print('write_grid_vtk: parse vector feature, key="', k, '", value shape=', v_np.shape)
            # write 2x2 tensor
            elif v_np.ndim == 3 + 2 and v_np.shape[-2] == 2 and v_np.shape[-1] == 2:
                result = (np.ascontiguousarray(v_np[..., 0, 0]), np.ascontiguousarray(v_np[..., 0, 1]), np.zeros_like(v_np[..., 0, 0]))
                data[d][k + "x"] = result
                result = (np.ascontiguousarray(v_np[..., 1, 0]), np.ascontiguousarray(v_np[..., 1, 1]), np.zeros_like(v_np[..., 0, 0]))
                data[d][k + "y"] = result
                result = (np.zeros_like(v_np[..., 0, 0]), np.zeros_like(v_np[..., 0, 0]), np.ones_like(v_np[..., 0, 0]))
                data[d][k + "z"] = result
                if verbose:
                    print('write_grid_vtk: parse tensor feature, key="', k, '", value shape=', v_np.shape)
            # write 3x3 tensor
            elif v_np.ndim == 3 + 2 and v_np.shape[-2] == 3 and v_np.shape[-1] == 3:
                result = (np.ascontiguousarray(v_np[..., 0, 0]), np.ascontiguousarray(v_np[..., 1, 0]), np.ascontiguousarray(v_np[..., 2, 0]))
                data[d][k + "x"] = result
                result = (np.ascontiguousarray(v_np[..., 0, 1]), np.ascontiguousarray(v_np[..., 1, 1]), np.ascontiguousarray(v_np[..., 2, 1]))
                data[d][k + "y"] = result
                result = (np.ascontiguousarray(v_np[..., 0, 2]), np.ascontiguousarray(v_np[..., 1, 2]), np.ascontiguousarray(v_np[..., 2, 2]))
                data[d][k + "z"] = result
                if verbose:
                    print('write_grid_vtk: parse tensor feature, key="', k, '", value shape=', v_np.shape)
            else:
                if verbose:
                    print('write_particle_vtk: parse scalar feature, key="', k, '", value shape=', v_np.shape)

            if v_np.ndim != 3 + 2:
                data[d][k] = result

        # write grid x
        if dimension == 2:
            zcoords = np.array([0]) * dx
        elif dimension == 3:
            zcoords = np.arange(0, res[2] + (1 if d == 2 else 0)) * dx + (0.5 * dx if d != 2 else 0)

        x = np.arange(0, res[0] + (1 if d == 0 else 0)) * dx + (0.5 * dx if d != 0 else 0)
        y = np.arange(0, res[1] + (1 if d == 1 else 0)) * dx + (0.5 * dx if d != 1 else 0)
        z = zcoords
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        gridToVTK(
            filenames[d],
            x=xx,
            y=yy,
            z=zz,
            # cellData={filename: scalar_field_np},
            pointData=data[d],
        )


def write_particle_vtk(points, feature_dict, filename, verbose=False):
    try:
        from pyevtk.hl import pointsToVTK  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise RuntimeError(
            "Failed to import pyevtk. Please install it via /\
        `pip install pyevtk` first. "
        )

    # check dim
    dim = 3
    p = np.zeros((dim))
    if isinstance(points, ti.Field):
        p = points.to_numpy()
        dim = p.shape[-1]
        if verbose:
            print("write_particle_vtk: parse points from taichi, dim=", dim, ",size=", p.shape)

    if isinstance(points, np.ndarray):
        p = points
        dim = points.shape[-1]
        if verbose:
            print("write_particle_vtk: parse points from numpy, dim=", dim, ",size=", p.shape)

    # parse positions to numpy array
    x = np.ascontiguousarray(p[:, 0])
    y = np.ascontiguousarray(p[:, 1])
    z = np.zeros_like(y)
    if dim == 3:
        z = np.ascontiguousarray(p[:, 2])

    num = x.shape[0]
    if "num" in feature_dict.keys():
        num = int(feature_dict["num"])

    x = x[0:num]
    y = y[0:num]
    z = z[0:num]

    # parse features to numpy array
    data = {}
    for k, v in feature_dict.items():
        if k == "num":
            continue
        v_np = v
        if not isinstance(v_np, np.ndarray):
            v_np = v_np.to_numpy()
        v_np = v_np[0:num]
        result = v_np
        # write 3d vector
        if v_np.ndim == 2 and v_np.shape[-1] == 3:
            result = (np.ascontiguousarray(v_np[..., 0]), np.ascontiguousarray(v_np[..., 1]), np.ascontiguousarray(v_np[..., 2]))
            if verbose:
                print('write_particle_vtk: parse vector feature, key="', k, '", value shape=', v_np.shape)
        # write 2d vector
        elif v_np.ndim == 2 and v_np.shape[-1] == 2:
            result = (np.ascontiguousarray(v_np[..., 0]), np.ascontiguousarray(v_np[..., 1]), np.zeros_like(v_np[..., 0]))
            if verbose:
                print('write_particle_vtk: parse vector feature, key="', k, '", value shape=', v_np.shape)
        # write 2x2 tensor
        elif v_np.ndim == 3 and v_np.shape[-2] == 2 and v_np.shape[-1] == 2:
            result = (np.ascontiguousarray(v_np[..., 0, 0]), np.ascontiguousarray(v_np[..., 0, 1]), np.zeros_like(v_np[..., 0, 0]))
            data[k + "x"] = result
            result = (np.ascontiguousarray(v_np[..., 1, 0]), np.ascontiguousarray(v_np[..., 1, 1]), np.zeros_like(v_np[..., 0, 0]))
            data[k + "y"] = result
            result = (np.zeros_like(v_np[..., 0, 0]), np.zeros_like(v_np[..., 0, 0]), np.ones_like(v_np[..., 0, 0]))
            data[k + "z"] = result
            if verbose:
                print('write_particle_vtk: parse tensor feature, key="', k, '", value shape=', v_np.shape)
        # write 3x3 tensor
        elif v_np.ndim == 3 and v_np.shape[-2] == 3 and v_np.shape[-1] == 3:
            result = (np.ascontiguousarray(v_np[..., 0, 0]), np.ascontiguousarray(v_np[..., 1, 0]), np.ascontiguousarray(v_np[..., 2, 0]))
            data[k + "x"] = result
            result = (np.ascontiguousarray(v_np[..., 0, 1]), np.ascontiguousarray(v_np[..., 1, 1]), np.ascontiguousarray(v_np[..., 2, 1]))
            data[k + "y"] = result
            result = (np.ascontiguousarray(v_np[..., 0, 2]), np.ascontiguousarray(v_np[..., 1, 2]), np.ascontiguousarray(v_np[..., 2, 2]))
            data[k + "z"] = result
            if verbose:
                print('write_particle_vtk: parse tensor feature, key="', k, '", value shape=', v_np.shape)
        else:
            if verbose:
                print('write_particle_vtk: parse scalar feature, key="', k, '", value shape=', v_np.shape)
        if v_np.ndim != 3:
            data[k] = result

    # write out
    pointsToVTK(filename, x, y, z, data=data)


__all__ = ["write_vtk"]
