import argparse

import numpy as np

from ptycho_fm.utils.cli import nonnegative_float, positive_float, positive_int


def generate_raster_positions(nx: int, ny: int, dx: float, dy: float, jitter: float = 0.0) -> np.ndarray:
    """Generate raster scan positions centered at the origin.

    The scan starts from the top-left corner, moves left-to-right along each row,
    and proceeds row-by-row from top to bottom.

    Args:
        nx: Number of points in the x direction.
        ny: Number of points in the y direction.
        dx: Step size in meters in the x direction.
        dy: Step size in meters in the y direction.
        jitter: Maximum random jitter in meters applied to each position in both x and y.
            A uniform random offset in [-jitter, +jitter] is added independently to each
            coordinate. Use a small fraction of dx/dy (e.g. 10%) to break periodicity
            and suppress grid artifacts without significantly altering the scan pattern.
            Default is 0.0 (no jitter).

    Returns:
        Array of shape (nx * ny, 2) containing (y, x) positions in meters.
        The origin (0, 0) is at the center of the scan grid.
    """
    # Calculate the extent of the scan
    width = (nx - 1) * dx
    height = (ny - 1) * dy

    # Generate x and y coordinates centered at origin
    x_coords = np.linspace(-width / 2, width / 2, nx)
    y_coords = np.linspace(height / 2, -height / 2, ny)  # Top to bottom

    # Create the raster pattern: for each row (y), sweep through all x values
    positions = []
    for y in y_coords:
        for x in x_coords:
            positions.append((y, x)) # Follows python ij indexing

    positions = np.array(positions)
    if jitter > 0.0:
        positions += np.random.uniform(-jitter, jitter, size=positions.shape)
    return positions


def generate_zigzag_positions(nx: int, ny: int, dx: float, dy: float, jitter: float = 0.0) -> np.ndarray:
    """Generate zigzag (serpentine) scan positions centered at the origin.

    The scan starts from the top-left corner. Even rows (0, 2, 4, ...) go left-to-right,
    odd rows (1, 3, 5, ...) go right-to-left.

    Args:
        nx: Number of points in the x direction.
        ny: Number of points in the y direction.
        dx: Step size in meters in the x direction.
        dy: Step size in meters in the y direction.
        jitter: Maximum random jitter in meters applied to each position in both x and y.
            A uniform random offset in [-jitter, +jitter] is added independently to each
            coordinate. Use a small fraction of dx/dy (e.g. 10%) to break periodicity
            and suppress grid artifacts without significantly altering the scan pattern.
            Default is 0.0 (no jitter).

    Returns:
        Array of shape (nx * ny, 2) containing (y, x) positions in meters.
        The origin (0, 0) is at the center of the scan grid.
    """
    # Calculate the extent of the scan
    width = (nx - 1) * dx
    height = (ny - 1) * dy

    # Generate x and y coordinates centered at origin
    x_coords = np.linspace(-width / 2, width / 2, nx)
    y_coords = np.linspace(height / 2, -height / 2, ny)  # Top to bottom

    # Create the zigzag pattern
    positions = []
    for row_idx, y in enumerate(y_coords):
        if row_idx % 2 == 0:
            # Even rows (0, 2, 4, ...): left to right
            for x in x_coords:
                positions.append((y, x))  # Follows python ij indexing
        else:
            # Odd rows (1, 3, 5, ...): right to left
            for x in reversed(x_coords):
                positions.append((y, x))  # Follows python ij indexing

    positions = np.array(positions)
    if jitter > 0.0:
        positions += np.random.uniform(-jitter, jitter, size=positions.shape)
    return positions


def generate_zigzag_pos_whole_pix(nx: int, ny: int, step_multiple: int, pixel_size: float, jitter: float = 0.0) -> np.ndarray:
    """Generate zigzag flyscan positions centered at the origin where the step size is defined as a whole
    number multiple of the pixel size.

    The scan starts from the top-left corner. Odd rows (0, 2, 4, ...) go left-to-right,
    even rows (1, 3, 5, ...) go right-to-left.

    Args:
        nx: Number of points in the x direction.
        ny: Number of points in the y direction.
        step_multiple: Step size as a whole number multiple of pixel_size.
        pixel_size: Pixel size in meters.
        jitter: Maximum random jitter in meters applied to each position in both x and y.
            See generate_zigzag_positions for details. Default is 0.0 (no jitter).

    Returns:
        Array of shape (nx * ny, 2) containing (y, x) positions in meters.
        The origin (0, 0) is at the center of the scan grid.
    """
    step = step_multiple * pixel_size
    return generate_zigzag_positions(nx, ny, step, step, jitter=jitter)


def offset_spiralsquares(base_pos: str, grid_x: int, grid_y: int, offset: float) -> np.ndarray:
    """Generate positions for a grid of multiple spiral square scans from a position file for one scan.
    
    Order of scans is fixed in this function. Columns (x): left to right. Rows (y): bottom to top.

    Args:
        base_pos: Path to .csv file of the probe positions for one spiral square scan in meters
        grid_x: Number of scans in the horizontal direction
        grid_y: Number of scans in the vertical direction
        offset: Step in meters between each scan (in both x and y, this only works for even steps)

    Returns:
        Array of shape (grid_x * grid_y * base_pos_array.shape[0], 2) containing (y, x) positions in meters
        The origin (0, 0) is at the center of the entire position array
    """
    base_pos_array = np.genfromtxt(base_pos, delimiter=",")
    base_pos_array[:, 1] = -base_pos_array[:, 1] # Spiral scans need flipped x positions
    pos_array = np.zeros((grid_x * grid_y * base_pos_array.shape[0], 2), dtype=np.float32)
    range_x = (grid_x - 1) * offset
    range_y = (grid_y - 1) * offset
    x_offsets = np.linspace(-range_x / 2, range_x / 2, grid_x)
    y_offsets = np.linspace(-range_y / 2, range_y / 2, grid_y)
    print("X offsets:", x_offsets, "| Y offsets:", y_offsets)
    scan_num = 0
    for i in range(grid_y):
        for j in range(grid_x): 
            tmp = np.zeros(base_pos_array.shape)
            tmp[:, 0] = base_pos_array[:, 0] - y_offsets[i]
            tmp[:, 1] = base_pos_array[:, 1] + x_offsets[j]
            pos_array[scan_num*base_pos_array.shape[0]:(scan_num+1)*base_pos_array.shape[0]] = tmp
            scan_num += 1

    return pos_array


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate scan positions as a headerless CSV of (y, x) coordinates in meters.",
        epilog=(
            "Example: python scripts/generate_positions.py raster --nx 100 --ny 101 "
            "--dx 100e-9 --dy 100e-9 --jitter 10e-9 --output positions.csv"
        ),
    )
    subparsers = parser.add_subparsers(dest="pattern", required=True)
    patterns = (
        ("raster", "Raster scan, left-to-right on every row", generate_raster_positions),
        ("zigzag", "Serpentine scan with alternating row directions", generate_zigzag_positions),
        ("zigzag-pixels", "Serpentine scan with steps in whole pixels", generate_zigzag_pos_whole_pix),
        ("spiral-squares", "Grid of spiral square scans from a base CSV", offset_spiralsquares),
    )
    for name, description, generator in patterns:
        command = subparsers.add_parser(name, help=description, description=description)
        command.set_defaults(generator=generator)
        command.add_argument("-o", "--output", required=True, help="Output CSV path")
        if name == "spiral-squares":
            command.add_argument("--base-pos", required=True, help="Base scan CSV path (coordinates in meters)")
            command.add_argument("--grid-x", type=positive_int, required=True, help="Number of scans horizontally")
            command.add_argument("--grid-y", type=positive_int, required=True, help="Number of scans vertically")
            command.add_argument("--offset", type=positive_float, required=True, help="Spacing between scans in meters")
        else:
            command.add_argument("--nx", type=positive_int, required=True, help="Number of points in x")
            command.add_argument("--ny", type=positive_int, required=True, help="Number of points in y")
            command.add_argument("--jitter", type=nonnegative_float, default=0.0, help="Maximum random offset in meters (default: 0)")
            if name == "zigzag-pixels":
                command.add_argument("--step-multiple", type=positive_int, required=True, help="Step size in whole pixels")
                command.add_argument("--pixel-size", type=positive_float, required=True, help="Pixel size in meters")
            else:
                command.add_argument("--dx", type=positive_float, required=True, help="Step size in x in meters")
                command.add_argument("--dy", type=positive_float, required=True, help="Step size in y in meters")

    args = vars(parser.parse_args(argv))
    generator = args.pop("generator")
    output = args.pop("output")
    args.pop("pattern")
    positions = generator(**args)
    np.savetxt(output, positions, delimiter=",")
    print(f"Saved {len(positions)} positions to {output}")


if __name__ == '__main__':
    main()
