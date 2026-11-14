from pathlib import Path
import argparse
import math
import re

import numpy as np
import matplotlib.pyplot as plt

ENVI_DTYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8",
               12: "u2", 13: "u4", 14: "i8", 15: "u8"}
ENVI_LAYOUT = {"bsq": "CHW", "bil": "HCW", "bip": "HWC"}


def read_header(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    pairs = re.findall(r"(?mi)^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)", text)
    return {k.strip().lower(): v.strip().strip("{}").strip() for k, v in pairs}


def find_header(path):
    for candidate in (path.with_suffix(".hdr"), path.with_suffix(".HDR"), Path(str(path) + ".hdr")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No ENVI header for {path}")


def load_cube(path, layout, key, nodata, scale):
    path = Path(path)
    suffix = path.suffix.lower()
    wavelengths = None
    good = None

    if suffix == ".npy":
        array, order = np.load(path, mmap_mode="r"), layout
    elif suffix == ".npz":
        archive = np.load(path)
        key = key or list(archive.keys())[0]
        array, order = archive[key], layout
    elif suffix == ".mat":
        try:
            from scipy.io import loadmat
            content = loadmat(path)
            names = [k for k, v in content.items() if not k.startswith("__") and getattr(v, "ndim", 0) == 3]
            key = key or names[0]
            array = content[key]
        except NotImplementedError:
            import h5py
            with h5py.File(path, "r") as handle:
                key = key or [k for k in handle.keys() if handle[k].ndim == 3][0]
                array = np.array(handle[key])
        order = layout
    elif suffix in {".tif", ".tiff"}:
        import tifffile
        array, order = tifffile.imread(path), layout
    else:
        meta = read_header(find_header(path))
        samples, lines, bands = (int(meta[k]) for k in ("samples", "lines", "bands"))
        dtype = np.dtype(("<" if int(meta["byte order"]) == 0 else ">") + ENVI_DTYPES[int(meta["data type"])])
        order = ENVI_LAYOUT[meta["interleave"].strip().lower()]
        sizes = {"C": bands, "H": lines, "W": samples}
        array = np.memmap(path, mode="r", dtype=dtype, offset=int(meta.get("header offset", 0)),
                          shape=tuple(sizes[a] for a in order))
        if "wavelength" in meta:
            wavelengths = np.asarray([float(v) for v in meta["wavelength"].split(",") if v.strip()])
            if "micro" in meta.get("wavelength units", "").lower():
                wavelengths = wavelengths * 1000
        if "data ignore value" in meta:
            nodata = list(nodata) + [float(meta["data ignore value"])]
        if "bbl" in meta:
            good = np.asarray([float(v) for v in meta["bbl"].split(",") if v.strip()]) != 0

    order = order.upper()
    cube = np.transpose(array, [order.index(a) for a in "CHW"])
    return cube, wavelengths, good, list(nodata), scale


def crop_cube(cube, rows, cols, nodata, scale, good):
    r0, r1 = rows if rows else (0, cube.shape[1])
    c0, c1 = cols if cols else (0, cube.shape[2])
    data = np.array(cube[:, r0:r1, c0:c1], dtype=np.float32)
    valid = np.isfinite(data)
    for value in nodata:
        valid &= data != value
    if good is not None:
        valid &= good[:, None, None]
    return data / scale, valid


def stretch(image, valid, low=2, high=98):
    values = image[valid & np.isfinite(image)]
    if values.size == 0:
        return np.zeros_like(image), 0.0, 1.0
    lo, hi = np.percentile(values, [low, high])
    hi = max(hi, lo + 1e-12)
    return np.clip((image - lo) / (hi - lo), 0, 1), lo, hi


def nearest_band(wavelengths, target, bands):
    if wavelengths is None:
        return min(bands - 1, max(0, int(round(target / 1000 * bands))))
    return int(np.argmin(np.abs(wavelengths - target)))


def composite(data, valid, indices):
    channels = []
    for index in indices:
        scaled, _, _ = stretch(data[index], valid[index])
        channels.append(scaled)
    rgb = np.dstack(channels)
    mask = ~valid[list(indices)].all(axis=0)
    rgb[mask] = 0
    return rgb


def pick_bands(valid, count, explicit):
    if explicit:
        return explicit
    usable = np.flatnonzero(valid.reshape(valid.shape[0], -1).any(axis=1))
    return sorted({int(usable[i]) for i in np.linspace(0, len(usable) - 1, count).round().astype(int)})


def plot_cube(name, data, valid, wavelengths, bands, rgb_bands, points, out):
    count = len(bands)
    fig = plt.figure(figsize=(4 * max(count, 3), 14))
    grid = fig.add_gridspec(3, max(count, 3), height_ratios=[1, 1, 1])

    for column, band in enumerate(bands):
        ax = fig.add_subplot(grid[0, column])
        scaled, _, _ = stretch(data[band], valid[band])
        ax.imshow(np.ma.array(scaled, mask=~valid[band]), cmap="gray", interpolation="nearest", aspect="equal")
        label = f"band {band}" + (f"\n{wavelengths[band]:.0f} nm" if wavelengths is not None else "")
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(grid[1, 0])
    ax.imshow(composite(data, valid, rgb_bands), interpolation="nearest", aspect="equal")
    ax.set_title(f"Composite bands {rgb_bands}", fontsize=9)
    ax.axis("off")

    x_axis = wavelengths if wavelengths is not None else np.arange(data.shape[0])
    x_label = "Wavelength (nm)" if wavelengths is not None else "Band index"

    ax = fig.add_subplot(grid[1, 1:])
    masked = np.where(valid, data, np.nan)
    mean = np.nanmean(masked.reshape(data.shape[0], -1), axis=1)
    low = np.nanpercentile(masked.reshape(data.shape[0], -1), 10, axis=1)
    high = np.nanpercentile(masked.reshape(data.shape[0], -1), 90, axis=1)
    ax.fill_between(x_axis, low, high, color="lightsteelblue", alpha=0.5, label="10-90 percentile")
    ax.plot(x_axis, mean, color="navy", label="Mean spectrum")
    for row, col in points:
        ax.plot(x_axis, np.where(valid[:, row, col], data[:, row, col], np.nan), linewidth=0.8,
                label=f"pixel ({row}, {col})")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[2, 0])
    coverage = valid.mean(axis=0)
    handle = ax.imshow(coverage, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_title("Fraction of valid bands", fontsize=9)
    ax.axis("off")
    fig.colorbar(handle, ax=ax, fraction=0.046)

    ax = fig.add_subplot(grid[2, 1:])
    mid = bands[len(bands) // 2]
    profile = np.nanmean(np.where(valid[mid], data[mid], np.nan), axis=0)
    ax.plot(profile, color="black")
    ax.set_title(f"Column profile, band {mid} (stripes appear as spikes)", fontsize=9)
    ax.set_xlabel("Column")
    ax.grid(alpha=0.3)

    fig.suptitle(f"{name}  |  {data.shape[0]} bands x {data.shape[1]} rows x {data.shape[2]} cols", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_comparison(data_a, valid_a, data_b, valid_b, wavelengths, bands, points, out):
    valid = valid_a & valid_b
    difference = data_b - data_a
    fig, axes = plt.subplots(3, len(bands), figsize=(4 * len(bands), 11), squeeze=False)
    for column, band in enumerate(bands):
        _, lo, hi = stretch(data_a[band], valid[band])
        for row, (image, title) in enumerate([(data_a[band], "A"), (data_b[band], "B")]):
            axes[row, column].imshow(np.ma.array(image, mask=~valid[band]), cmap="gray",
                                     vmin=lo, vmax=hi, interpolation="nearest", aspect="equal")
            axes[row, column].set_title(f"{title} | band {band}", fontsize=9)
            axes[row, column].axis("off")
        amplitude = max(float(np.percentile(np.abs(difference[band][valid[band]]), 99)) if valid[band].any() else 0, 1e-12)
        handle = axes[2, column].imshow(np.ma.array(difference[band], mask=~valid[band]), cmap="coolwarm",
                                        vmin=-amplitude, vmax=amplitude, interpolation="nearest", aspect="equal")
        axes[2, column].set_title("B - A", fontsize=9)
        axes[2, column].axis("off")
        fig.colorbar(handle, ax=axes[2, column], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    x_axis = wavelengths if wavelengths is not None else np.arange(data_a.shape[0])
    fig, axes = plt.subplots(1, len(points), figsize=(5 * len(points), 4), squeeze=False)
    for ax, (row, col) in zip(axes[0], points):
        ax.plot(x_axis, np.where(valid_a[:, row, col], data_a[:, row, col], np.nan), color="gray", label="A")
        ax.plot(x_axis, np.where(valid_b[:, row, col], data_b[:, row, col], np.nan), color="royalblue", label="B")
        ax.set_title(f"pixel ({row}, {col})", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_name(out.stem + "_spectra.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cube")
    parser.add_argument("--compare")
    parser.add_argument("--layout", default="CHW")
    parser.add_argument("--key")
    parser.add_argument("--rows", nargs=2, type=int)
    parser.add_argument("--cols", nargs=2, type=int)
    parser.add_argument("--nodata", nargs="*", type=float, default=[])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--bands", nargs="*", type=int)
    parser.add_argument("--n-bands", type=int, default=4)
    parser.add_argument("--rgb", nargs=3, type=int)
    parser.add_argument("--points", nargs="*", type=int, help="row col row col ...")
    parser.add_argument("--out", default="hsi_view")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cube, wavelengths, good, nodata, scale = load_cube(args.cube, args.layout, args.key, args.nodata, args.scale)
    data, valid = crop_cube(cube, args.rows, args.cols, nodata, scale, good)
    bands = pick_bands(valid, args.n_bands, args.bands)

    bands_count = data.shape[0]
    rgb_bands = args.rgb or [nearest_band(wavelengths, w, bands_count) for w in (650, 550, 450)]
    if args.rgb is None and wavelengths is None:
        rgb_bands = [int(bands_count * f) for f in (0.75, 0.5, 0.25)]

    height, width = data.shape[1:]
    if args.points:
        points = list(zip(args.points[0::2], args.points[1::2]))
    else:
        points = [(height // 4, width // 4), (3 * height // 4, 3 * width // 4)]

    plot_cube(Path(args.cube).stem, data, valid, wavelengths, bands, rgb_bands, points,
              out_dir / f"{Path(args.cube).stem}_overview.png")

    if args.compare:
        other, wave_b, good_b, nodata_b, scale_b = load_cube(args.compare, args.layout, args.key, args.nodata, args.scale)
        data_b, valid_b = crop_cube(other, args.rows, args.cols, nodata_b, scale_b, good_b)
        if data_b.shape != data.shape:
            raise RuntimeError(f"Shape mismatch: {data.shape} vs {data_b.shape}")
        plot_comparison(data, valid, data_b, valid_b, wavelengths, bands, points,
                        out_dir / "comparison.png")


if __name__ == "__main__":
    main()
