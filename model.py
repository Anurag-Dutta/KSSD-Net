from pathlib import Path
import copy
import csv
import json
import math
import random
import re
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

CFG = {
    "name": "m3_north_pole",
    "data_root": r"C:\Users\Dell\Desktop\HSI Denoise\Lunar North Pole M3 Data\l2 data",
    "output_dir": r"C:\Users\Dell\Desktop\HSI Denoise\anurag\output_ch_1",
    "file_glob": "**/*.img",
    "file_format": "auto",
    "array_layout": "HWC",
    "array_key": None,
    "data_scale": 1.0,
    "nodata_values": [-999.0],
    "bad_bands": [],
    "wavelengths": None,
    "wavelength_units": "nm",
    "header_overrides": {"*": {"header offset": 0, "wavelength units": "nm"}},
    "split_mode": "acquisition",
    "splits": {
        "train": ["M3G20090622T042035_V01_RFL", "M3G20090622T124051_V01_RFL"],
        "validation": ["M3G20090623T012541_V01_RFL"],
        "test": ["M3G20090623T182551_V01_RFL"],
    },
    "auto_discover_train": True,
    "discover_regex": "^M3.*_RFL$",
    "regions": {
        "train": {"centers": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
                  "height": 400, "width": 200},
        "validation": {"centers": [0.18, 0.38, 0.58, 0.78], "height": 200, "width": 200},
        "test": {"centers": [0.18, 0.38, 0.58, 0.78], "height": 200, "width": 200},
    },
    "spatial_split": {
        "product": None,
        "fractions": {"train": [0.0, 0.6], "validation": [0.6, 0.8], "test": [0.8, 1.0]},
        "gap": 48,
    },
    "min_valid_fraction": 0.95,
    "display_bands": [10, 30, 50, 70],
    "n_display_bands": 4,
    "show_region_previews": True,
    "regions_per_preview_page": 4,
    "seed": 1234,
    "patch": 32,
    "batch_size": 2,
    "epochs": 50,
    "steps_per_epoch": 2000,
    "patience": 10,
    "lr": 2e-4,
    "width": 16,
    "spatial_blocks": 4,
    "state_size": 12,
    "spectral_chunk": 512,
    "tile": 48,
    "stride": 32,
    "mask_every": 4,
    "val_patches_per_region": 2,
    "run_final_test": True,
    "run_raw_transfer": True,
    "raw_transfer_region_index": 0,
    "noise": {
        "sigma_min": 0.01,
        "sigma_max": 0.12,
        "impulse_max": 0.005,
        "column_probability": 0.06,
        "stripe_std": 0.18,
    },
}


def deep_update(base, new):
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


if len(sys.argv) > 1 and sys.argv[-1].lower().endswith(".json"):
    with open(sys.argv[-1], "r", encoding="utf-8") as file:
        deep_update(CFG, json.load(file))

SEED = CFG["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = Path(CFG["data_root"])
OUT = Path(CFG["output_dir"])
OUT.mkdir(parents=True, exist_ok=True)

SPLIT_MODE = CFG["split_mode"]
SPLIT_PRODUCTS = {k: list(v) for k, v in CFG["splits"].items()}
REGION_SPECS = CFG["regions"]
MIN_VALID_FRACTION = CFG["min_valid_fraction"]
NODATA_VALUES = list(CFG["nodata_values"])
DATA_SCALE = float(CFG["data_scale"])
SHOW_REGION_PREVIEWS = CFG["show_region_previews"]
REGIONS_PER_PREVIEW_PAGE = CFG["regions_per_preview_page"]

PATCH = CFG["patch"]
BATCH_SIZE = CFG["batch_size"]
EPOCHS = CFG["epochs"]
STEPS_PER_EPOCH = CFG["steps_per_epoch"]
PATIENCE = CFG["patience"]
LR = CFG["lr"]
WIDTH = CFG["width"]
SPATIAL_BLOCKS = CFG["spatial_blocks"]
STATE_SIZE = CFG["state_size"]
SPECTRAL_CHUNK = CFG["spectral_chunk"]
TILE = CFG["tile"]
STRIDE = CFG["stride"]
MASK_EVERY = CFG["mask_every"]
VAL_PATCHES_PER_REGION = CFG["val_patches_per_region"]
RUN_FINAL_TEST = CFG["run_final_test"]
RUN_RAW_TRANSFER = CFG["run_raw_transfer"]
RAW_TRANSFER_REGION_INDEX = CFG["raw_transfer_region_index"]
NOISE = CFG["noise"]

assert REGIONS_PER_PREVIEW_PAGE > 0
assert SPLIT_MODE in ("acquisition", "spatial")

ENVI_DTYPES = {
    1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8",
    12: "u2", 13: "u4", 14: "i8", 15: "u8",
}
ENVI_LAYOUT = {"bsq": "CHW", "bil": "HCW", "bip": "HWC"}
ENVI_SUFFIXES = {".img", ".raw", ".dat", ".bsq", ".bil", ".bip", ".hsi", ""}
UM_UNITS = {"micrometers", "micrometer", "micrometres", "micrometre", "um", "µm"}
NM_UNITS = {"nanometers", "nanometer", "nanometres", "nanometre", "nm"}


def parse_envi_header(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    entries = re.findall(r"(?mi)^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)", text)
    return {
        key.strip().lower(): value.strip().strip("{}").strip()
        for key, value in entries
    }


def find_header(path):
    candidates = [
        path.with_suffix(".hdr"),
        path.with_suffix(".HDR"),
        Path(str(path) + ".hdr"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No ENVI header found for:\n{path}")


def resolve_format(path):
    fmt = str(CFG["file_format"]).lower()
    if fmt != "auto":
        return fmt
    suffix = path.suffix.lower()
    if suffix in {".npy"}:
        return "npy"
    if suffix == ".npz":
        return "npz"
    if suffix == ".mat":
        return "mat"
    if suffix in {".tif", ".tiff"}:
        return "tiff"
    if suffix in ENVI_SUFFIXES:
        return "envi"
    raise ValueError(f"Cannot infer format for {path}; set file_format.")


def load_array_file(path, fmt):
    key = CFG["array_key"]
    if fmt == "npy":
        return np.load(path, mmap_mode="r")
    if fmt == "npz":
        archive = np.load(path)
        names = list(archive.keys())
        key = key or (names[0] if len(names) == 1 else None)
        if key is None or key not in archive:
            raise KeyError(f"Set array_key; available: {names}")
        return archive[key]
    if fmt == "mat":
        try:
            from scipy.io import loadmat
            content = loadmat(path)
            names = [k for k, v in content.items()
                     if not k.startswith("__") and getattr(v, "ndim", 0) == 3]
            key = key or (names[0] if len(names) == 1 else None)
            if key is None or key not in content:
                raise KeyError(f"Set array_key; 3-D arrays: {names}")
            return content[key]
        except NotImplementedError:
            import h5py
            with h5py.File(path, "r") as handle:
                names = [k for k in handle.keys() if handle[k].ndim == 3]
                key = key or (names[0] if len(names) == 1 else None)
                if key is None or key not in handle:
                    raise KeyError(f"Set array_key; 3-D datasets: {names}")
                return np.array(handle[key])
    if fmt == "tiff":
        import tifffile
        return tifffile.imread(path)
    raise ValueError(f"Unsupported format: {fmt}")


class HSICube:
    def __init__(self, product, path):
        self.product = product
        self.path = path
        self.format = resolve_format(path)
        self.metadata = {}
        self.nodata = list(NODATA_VALUES)

        if self.format == "envi":
            self._open_envi()
        else:
            self._open_array()

        dims = dict(zip(self.layout, self.map.shape))
        self.bands, self.lines, self.samples = dims["C"], dims["H"], dims["W"]

        if not hasattr(self, "good_bands"):
            self.good_bands = np.ones(self.bands, dtype=bool)
        for index in CFG["bad_bands"]:
            if not 0 <= index < self.bands:
                raise ValueError(f"bad_bands index out of range: {index}")
            self.good_bands[index] = False
        if not self.good_bands.any():
            raise RuntimeError(f"No usable bands in {path}")

    def _open_envi(self):
        self.header_path = find_header(self.path)
        metadata = parse_envi_header(self.header_path)
        overrides = CFG["header_overrides"]
        metadata.update(overrides.get("*", {}))
        metadata.update(overrides.get(self.product, {}))

        required = ["samples", "lines", "bands", "data type",
                    "byte order", "interleave", "header offset"]
        missing = [key for key in required if key not in metadata]
        if missing:
            raise RuntimeError(
                f"Header: {self.header_path}\nMissing fields: {missing}\n"
                "Add documented values under header_overrides. "
                "The reader will not guess the binary layout."
            )

        samples = int(metadata["samples"])
        lines = int(metadata["lines"])
        bands = int(metadata["bands"])
        offset = int(metadata["header offset"])
        interleave = str(metadata["interleave"]).strip().lower()
        dtype_code = int(metadata["data type"])
        byte_order = int(metadata["byte order"])

        if "bbl" in metadata:
            bbl = np.asarray(
                [float(v.strip()) for v in metadata["bbl"].split(",") if v.strip()],
                dtype=np.float64,
            )
            if bbl.shape != (bands,) or not np.isfinite(bbl).all():
                raise RuntimeError(f"Invalid BBL in {self.header_path}")
            self.good_bands = bbl != 0

        if dtype_code not in ENVI_DTYPES:
            raise ValueError(f"Unsupported ENVI datatype: {dtype_code}")
        if byte_order not in [0, 1]:
            raise ValueError("Invalid byte order.")
        if interleave not in ENVI_LAYOUT:
            raise ValueError(f"Unknown interleave: {interleave}")
        if min(samples, lines, bands) <= 0 or offset < 0:
            raise ValueError("Invalid cube dimensions/offset.")

        dtype = np.dtype(("<" if byte_order == 0 else ">") + ENVI_DTYPES[dtype_code])
        layout = ENVI_LAYOUT[interleave]
        sizes = {"C": bands, "H": lines, "W": samples}
        shape = tuple(sizes[a] for a in layout)

        expected = offset + bands * lines * samples * dtype.itemsize
        actual = self.path.stat().st_size
        if actual != expected:
            raise RuntimeError(
                f"File-size mismatch: {self.path.name}\n"
                f"Expected: {expected:,}\nActual:   {actual:,}\n"
                "Verify the header before continuing."
            )

        if "data ignore value" in metadata:
            self.nodata.append(float(metadata["data ignore value"]))

        self.layout = layout
        self.metadata = metadata
        self.map = np.memmap(self.path, dtype=dtype, mode="r", offset=offset, shape=shape)

    def _open_array(self):
        array = load_array_file(self.path, self.format)
        layout = str(CFG["array_layout"]).upper()
        if array.ndim != 3 or sorted(layout) != ["C", "H", "W"]:
            raise ValueError(
                f"{self.path.name}: need a 3-D array and array_layout "
                f"as a permutation of C,H,W; got shape {array.shape}, layout {layout}"
            )
        self.layout = layout
        self.map = array

    def wavelengths_nm(self):
        raw = self.metadata.get("wavelength")
        if raw is not None:
            values = [float(v.strip()) for v in raw.split(",") if v.strip()]
            units = self.metadata.get("wavelength units", CFG["wavelength_units"])
        elif CFG["wavelengths"] is not None:
            values = list(CFG["wavelengths"])
            units = CFG["wavelength_units"]
        else:
            return None
        values = np.asarray(values, dtype=np.float64)
        units = str(units).strip().strip('"').lower()
        if units in UM_UNITS:
            values = values * 1000.0
        elif units not in NM_UNITS:
            raise RuntimeError(f"Unrecognized wavelength units: {units!r}")
        if values.shape != (self.bands,) or not np.isfinite(values).all():
            raise RuntimeError(f"Expected {self.bands} finite wavelengths.")
        return values

    def crop(self, first, last, left, right):
        if not (0 <= first < last <= self.lines and 0 <= left < right <= self.samples):
            raise ValueError("Crop exceeds cube dimensions.")

        index = {"H": slice(first, last), "W": slice(left, right), "C": slice(None)}
        sub = self.map[tuple(index[a] for a in self.layout)]
        view = np.transpose(sub, [self.layout.index(a) for a in "CHW"])

        array = np.array(view, dtype=np.float32, copy=True)
        valid = np.isfinite(array)
        for value in self.nodata:
            valid &= array != value
        valid &= self.good_bands[:, None, None]

        if DATA_SCALE != 1.0:
            array = array / DATA_SCALE
        return array, valid

    def close(self):
        mapping = getattr(self, "map", None)
        if isinstance(mapping, np.memmap):
            mapping._mmap.close()
        self.map = None


available_files = {}
for candidate in DATA_ROOT.glob(CFG["file_glob"]):
    if candidate.is_file():
        available_files.setdefault(candidate.stem.lower(), []).append(candidate)


def locate(product):
    matches = available_files.get(product.lower(), [])
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one file for '{product}' under {DATA_ROOT} "
            f"matching {CFG['file_glob']}; found {matches}"
        )
    return matches[0]


if SPLIT_MODE == "acquisition":
    flat_products = [p for products in SPLIT_PRODUCTS.values() for p in products]
    if len(flat_products) != len(set(flat_products)):
        raise RuntimeError("An acquisition appears in multiple partitions.")

    held_out_products = set(SPLIT_PRODUCTS.get("validation", []) + SPLIT_PRODUCTS.get("test", []))
    discovered = []
    if CFG["auto_discover_train"] and CFG["discover_regex"]:
        pattern = re.compile(CFG["discover_regex"], re.IGNORECASE)
        discovered = sorted(
            paths[0].stem for paths in available_files.values()
            if pattern.match(paths[0].stem) and paths[0].stem not in held_out_products
        )
    TRAIN_SAMPLING_PRODUCTS = sorted(set(SPLIT_PRODUCTS["train"]) | set(discovered))
    flat_products += [p for p in TRAIN_SAMPLING_PRODUCTS if p not in flat_products]
    assert not (set(TRAIN_SAMPLING_PRODUCTS) & held_out_products)
else:
    SPATIAL_PRODUCT = CFG["spatial_split"]["product"]
    if SPATIAL_PRODUCT is None:
        if len(available_files) != 1:
            raise RuntimeError("Set spatial_split.product; files found: "
                               f"{[v[0].name for v in available_files.values()]}")
        SPATIAL_PRODUCT = next(iter(available_files.values()))[0].stem
    flat_products = [SPATIAL_PRODUCT]
    TRAIN_SAMPLING_PRODUCTS = [SPATIAL_PRODUCT]

cubes = {product: HSICube(product, locate(product)) for product in flat_products}

band_counts = {cube.bands for cube in cubes.values()}
if len(band_counts) != 1:
    raise RuntimeError("Different band counts require explicit harmonization.")
BANDS = band_counts.pop()

reference_cube = next(iter(cubes.values()))
if CFG["display_bands"]:
    DISPLAY_BANDS = [b for b in CFG["display_bands"] if 0 <= b < BANDS]
else:
    good_indices = np.flatnonzero(reference_cube.good_bands)
    picks = np.linspace(0, len(good_indices) - 1, CFG["n_display_bands"]).round().astype(int)
    DISPLAY_BANDS = sorted({int(good_indices[i]) for i in picks})
if not DISPLAY_BANDS:
    raise RuntimeError("None of the requested display bands exists.")


def build_manifest():
    manifest = []

    if SPLIT_MODE == "spatial":
        cube = cubes[SPATIAL_PRODUCT]
        fractions = CFG["spatial_split"]["fractions"]
        gap = int(CFG["spatial_split"]["gap"])
        for split in ("train", "validation", "test"):
            start_fraction, end_fraction = fractions[split]
            first = int(round(start_fraction * cube.lines))
            last = int(round(end_fraction * cube.lines))
            if first > 0:
                first += gap
            if last - first < 8:
                raise RuntimeError(f"Spatial zone for {split} is too small.")
            manifest.append({
                "id": f"{split}_{SPATIAL_PRODUCT}_00",
                "split": split,
                "product": SPATIAL_PRODUCT,
                "first": first,
                "last": last,
                "left": 0,
                "right": cube.samples,
            })
        return manifest

    for split, products in SPLIT_PRODUCTS.items():
        spec = REGION_SPECS[split]
        for product in products:
            cube = cubes[product]
            height = min(int(spec["height"]), cube.lines)
            region_width = min(int(spec["width"]), cube.samples)
            centers = spec["centers"] if height < cube.lines else [0.5]

            left = (cube.samples - region_width) // 2
            previous = []

            for index, fraction in enumerate(centers):
                if not 0 <= fraction <= 1:
                    raise ValueError("Center fractions must be in [0,1].")

                center = int(round(fraction * (cube.lines - 1)))
                first = int(np.clip(center - height // 2, 0, cube.lines - height))
                last = first + height

                for old_first, old_last in previous:
                    if max(first, old_first) < min(last, old_last):
                        raise RuntimeError(
                            f"Overlapping regions in {product}; "
                            "adjust centers or height."
                        )
                previous.append((first, last))

                manifest.append({
                    "id": f"{split}_{product}_{index:02d}",
                    "split": split,
                    "product": product,
                    "first": first,
                    "last": last,
                    "left": left,
                    "right": left + region_width,
                })
    return manifest


manifest = build_manifest()

min_height = min(e["last"] - e["first"] for e in manifest)
min_width = min(e["right"] - e["left"] for e in manifest)
eval_entries = [e for e in manifest if e["split"] != "train"]
eval_height = min(e["last"] - e["first"] for e in eval_entries)
eval_width = min(e["right"] - e["left"] for e in eval_entries)

PATCH = min(PATCH, min_height, min_width)
TILE = min(TILE, eval_height, eval_width)
STRIDE = min(STRIDE, TILE)
assert PATCH >= 5, "Scenes/regions are too small for the requested patch size."

with open(OUT / "region_manifest.json", "w") as file:
    json.dump(manifest, file, indent=2)


def load_manifest_regions(split):
    regions = []
    for entry in manifest:
        if entry["split"] != split:
            continue

        cube = cubes[entry["product"]]
        array, valid = cube.crop(
            entry["first"], entry["last"], entry["left"], entry["right"]
        )

        if valid.any(axis=0).mean() < MIN_VALID_FRACTION:
            raise RuntimeError(
                f"{entry['id']} has <{MIN_VALID_FRACTION:.0%} valid values. "
                "Inspect nodata_values, bad_bands and region selection."
            )

        assert array.shape[0] == BANDS
        regions.append({**entry, "raw": array, "valid": valid})

    if not regions:
        raise RuntimeError(f"No regions for split {split}")
    return regions


train_regions = load_manifest_regions("train")
validation_regions = load_manifest_regions("validation")

offset = np.zeros((BANDS, 1, 1), dtype=np.float32)
scale = np.zeros_like(offset)

for band in tqdm(range(BANDS), desc="Training-only scaling"):
    values = np.concatenate([
        item["raw"][band][item["valid"][band]] for item in train_regions
    ])

    if values.size == 0:
        offset[band, 0, 0] = 0.0
        scale[band, 0, 0] = 1.0
        continue

    low, high = np.percentile(values, [10, 90])
    median = float(np.median(values))

    offset[band, 0, 0] = median
    scale[band, 0, 0] = max(
        float(high - low), float(values.std()), abs(median) * 1e-4, 1e-6
    )


def prepare_region(item):
    filled = np.where(item["valid"], item["raw"], offset).astype(np.float32)
    normalized = ((filled - offset) / scale).astype(np.float32)

    if not np.isfinite(normalized).all():
        raise RuntimeError(f"Non-finite normalized values: {item['id']}")

    return {
        **item,
        "raw": filled,
        "x": torch.from_numpy(normalized),
        "v": torch.from_numpy(item["valid"]),
    }


train_regions = [prepare_region(item) for item in train_regions]
validation_regions = [prepare_region(item) for item in validation_regions]

log_scale = np.log(scale[:, 0, 0])
log_scale = (
    (log_scale - log_scale.mean()) / max(float(log_scale.std()), 1e-6)
).astype(np.float32)

band_coordinate = np.linspace(-1, 1, BANDS, dtype=np.float32)


def inverse_scale(array):
    return array * scale + offset


scale_t = torch.from_numpy(scale).unsqueeze(0).to(DEVICE)
offset_t = torch.from_numpy(offset).unsqueeze(0).to(DEVICE)


def limits(image, valid):
    values = image[valid & np.isfinite(image)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(values, [2, 98])
    return float(low), float(max(high, low + 1e-6))


def preview_regions(regions, filename_prefix):
    if not SHOW_REGION_PREVIEWS:
        return

    bands = DISPLAY_BANDS

    for page_start in range(0, len(regions), REGIONS_PER_PREVIEW_PAGE):
        subset = regions[page_start:page_start + REGIONS_PER_PREVIEW_PAGE]

        fig, axes = plt.subplots(
            len(subset), len(bands),
            figsize=(4.1 * len(bands), 5.0 * len(subset)),
            squeeze=False,
        )

        for row, item in enumerate(subset):
            for column, band in enumerate(bands):
                low, high = limits(item["raw"][band], item["valid"][band])
                display = np.ma.array(item["raw"][band], mask=~item["valid"][band])

                axes[row, column].imshow(
                    display, cmap="gray", vmin=low, vmax=high,
                    interpolation="nearest", aspect="equal",
                )
                axes[row, column].set_title(
                    f"{item['split']} | band index {band}\n"
                    f"{item['product']}\n"
                    f"rows {item['first']}:{item['last']}, "
                    f"cols {item['left']}:{item['right']}",
                    fontsize=8,
                )
                axes[row, column].axis("off")

        page_number = page_start // REGIONS_PER_PREVIEW_PAGE + 1

        fig.suptitle(
            f"Region preview\n\nALL {BANDS} bands are loaded; "
            f"only {bands} are displayed",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(
            OUT / f"{filename_prefix}_page_{page_number:02d}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.show()
        plt.close(fig)


preview_regions(train_regions[:1], "training_multiband_preview")
preview_regions(validation_regions[:1], "validation_multiband_preview")


def corrupt(
    target,
    generator=None,
    fixed=False,
    mode="mixed",
    gaussian_sigma=None,
    impulse_probability=0.002,
):
    batch, channels, height, width = target.shape

    def rand(shape):
        return torch.rand(
            shape, device=target.device, dtype=target.dtype, generator=generator
        )

    def normal(shape):
        return torch.randn(
            shape, device=target.device, dtype=target.dtype, generator=generator
        )

    if mode not in ["mixed", "gaussian", "impulse", "stripe", "identity"]:
        raise ValueError(mode)

    if gaussian_sigma is not None:
        sigma = target.new_full((batch, 1, 1, 1), gaussian_sigma)
    elif fixed:
        sigma = target.new_full((batch, 1, 1, 1), 0.07)
    else:
        sigma = (
            NOISE["sigma_min"]
            + rand((batch, 1, 1, 1)) * (NOISE["sigma_max"] - NOISE["sigma_min"])
        )

    gaussian = normal(target.shape) * sigma * (0.8 + 0.4 * rand((batch, channels, 1, 1)))

    probability = (
        target.new_full((batch, 1, 1, 1), impulse_probability)
        if fixed
        else rand((batch, 1, 1, 1)) * NOISE["impulse_max"]
    )

    impulse_mask = rand(target.shape) < probability
    impulse_sign = torch.where(rand(target.shape) < 0.5, -1.0, 1.0)
    impulses = impulse_mask * impulse_sign * (0.5 + 1.5 * rand(target.shape))

    columns = (rand((batch, 1, 1, width)) < NOISE["column_probability"]).to(target.dtype)

    if not fixed and random.random() < 0.5:
        columns = F.max_pool2d(columns, kernel_size=(1, 3), stride=1, padding=(0, 1))

    activity = (rand((batch, channels, 1, 1)) < 0.6).to(target.dtype)
    common = normal((batch, 1, 1, width))
    independent = normal((batch, channels, 1, width))

    additive = (0.7 * common + 0.3 * independent) * NOISE["stripe_std"]
    gain = normal((batch, channels, 1, width)) * 0.03
    stripes = columns * activity * (additive + gain * target)

    if mode not in ["mixed", "gaussian"]:
        gaussian = torch.zeros_like(target)
    if mode not in ["mixed", "impulse"]:
        impulses = torch.zeros_like(target)
    if mode not in ["mixed", "stripe"]:
        stripes = torch.zeros_like(target)

    return target + gaussian + impulses + stripes, stripes


class ChebyKAN(nn.Module):
    def __init__(self, inputs, outputs, degree=3):
        super().__init__()

        if degree < 1:
            raise ValueError("degree must be >=1")

        self.degree = degree
        self.base = nn.Linear(inputs, outputs)
        self.coefficients = nn.Parameter(
            torch.randn(outputs, inputs, degree + 1) * (0.02 / math.sqrt(inputs))
        )

    def forward(self, x):
        z = torch.tanh(x)
        terms = [torch.ones_like(z), z]

        for _ in range(2, self.degree + 1):
            terms.append(2 * z * terms[-1] - terms[-2])

        basis = torch.stack(terms[:self.degree + 1], dim=-1)

        return self.base(F.silu(x)) + torch.einsum(
            "...ik,oik->...o", basis, self.coefficients
        )


def zero_kan(layer):
    nn.init.zeros_(layer.base.weight)
    nn.init.zeros_(layer.base.bias)
    nn.init.zeros_(layer.coefficients)


class SpatialBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1, padding_mode="reflect")
        self.kan = ChebyKAN(width, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, padding_mode="reflect")

    def forward(self, x):
        y = self.conv1(x)
        y = self.kan(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + 0.1 * self.conv2(y)


class SpatialBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(2, WIDTH, 3, padding=1, padding_mode="reflect")
        self.blocks = nn.Sequential(*[SpatialBlock(WIDTH) for _ in range(SPATIAL_BLOCKS)])
        self.head = nn.Conv2d(WIDTH, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, observed):
        batch, channels, height, width = x.shape

        images = x.reshape(batch * channels, 1, height, width)
        flags = observed.expand_as(x).reshape(batch * channels, 1, height, width)

        features = self.blocks(self.stem(torch.cat([images, flags], dim=1)))

        return self.head(features).reshape(batch, channels, height, width)


class SpectralBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter_kan = ChebyKAN(4, 3 * STATE_SIZE)
        self.log_rates = nn.Parameter(torch.linspace(-2, 1, STATE_SIZE))
        self.head = ChebyKAN(2 * STATE_SIZE + 1, 1)
        zero_kan(self.head)

        self.register_buffer("coordinate", torch.from_numpy(band_coordinate))
        self.register_buffer("scale_metadata", torch.from_numpy(log_scale))

    def scan(self, drive, steps, reverse=False):
        number, length, state_size = drive.shape
        state = drive.new_zeros(number, state_size)
        outputs = [None] * length
        rates = F.softplus(self.log_rates) + 1e-4

        indices = range(length - 1, -1, -1) if reverse else range(length)

        for index in indices:
            decay = torch.exp(-steps[:, index] * rates)
            state = decay * state + (1 - decay) * drive[:, index]
            outputs[index] = state

        return torch.stack(outputs, dim=1)

    def sequences(self, values, observed):
        number, channels = values.shape

        tokens = torch.stack([
            values,
            observed,
            self.coordinate.expand(number, channels),
            self.scale_metadata.expand(number, channels),
        ], dim=-1)

        drive, steps, gates = self.parameter_kan(tokens).chunk(3, dim=-1)

        drive = torch.tanh(drive)
        steps = F.softplus(steps) + 1e-3
        gates = torch.sigmoid(gates)

        forward = self.scan(drive, steps)
        backward = self.scan(drive, steps, reverse=True)

        features = torch.cat([
            gates * forward, gates * backward, values.unsqueeze(-1)
        ], dim=-1)

        return self.head(features).squeeze(-1)

    def forward(self, x, observed):
        batch, channels, height, width = x.shape

        values = x.permute(0, 2, 3, 1).reshape(-1, channels)
        flags = observed.expand_as(x).permute(0, 2, 3, 1).reshape(-1, channels)

        outputs = []
        for start in range(0, len(values), SPECTRAL_CHUNK):
            stop = start + SPECTRAL_CHUNK
            outputs.append(self.sequences(values[start:stop], flags[start:stop]))

        return torch.cat(outputs, dim=0).reshape(
            batch, height, width, channels
        ).permute(0, 3, 1, 2)


class StripeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 24, 7, padding=3, padding_mode="replicate"),
            nn.SiLU(),
            nn.Conv1d(24, 24, 5, padding=2, padding_mode="replicate"),
            nn.SiLU(),
            nn.Conv1d(24, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        batch, channels, height, width = x.shape

        mean_profile = x.mean(dim=2)
        median_profile = x.median(dim=2).values

        profiles = torch.stack([mean_profile, median_profile], dim=2).reshape(
            batch * channels, 2, width
        )

        offsets = self.net(profiles).reshape(batch, channels, 1, width)

        return offsets.expand(batch, channels, height, width)


class DualDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = SpatialBranch()
        self.spectral = SpectralBranch()
        self.stripe = StripeHead()
        self.gate = nn.Conv2d(3, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x, observed=None):
        if x.ndim != 4:
            raise ValueError("Expected [batch,bands,height,width].")

        if x.shape[1] != BANDS:
            raise RuntimeError(f"Expected ALL {BANDS} bands; received {x.shape[1]}.")

        if observed is None:
            observed = torch.ones_like(x)

        stripe = self.stripe(x)
        base = x - stripe

        spatial_delta = self.spatial(base, observed)
        spectral_delta = self.spectral(base, observed)

        batch, channels, height, width = x.shape

        features = torch.stack([base, spatial_delta, spectral_delta], dim=2).reshape(
            batch * channels, 3, height, width
        )

        gate = torch.sigmoid(self.gate(features)).reshape(batch, channels, height, width)

        return {
            "output": base + gate * spatial_delta + (1 - gate) * spectral_delta,
            "spatial": base + spatial_delta,
            "spectral": base + spectral_delta,
            "stripe": stripe,
            "gate": gate,
        }


def mask_input(x, valid):
    batch, channels, height, width = x.shape

    hidden = torch.rand((batch, 1, height, width), device=x.device) < 0.04

    if random.random() < 0.5:
        column = random.randrange(1, width - 1)
        hidden[:, :, :, column] = True

    observed = (~hidden).expand_as(x) & valid
    source = torch.where(observed, x, torch.zeros_like(x))

    kernel = x.new_ones((1, 1, 5, 5))
    kernel[:, :, 2, 2] = 0

    numerator = F.conv2d(
        source.reshape(batch * channels, 1, height, width), kernel, padding=2
    )
    denominator = F.conv2d(
        observed.float().reshape(batch * channels, 1, height, width),
        kernel, padding=2,
    )

    replacement = (numerator / denominator.clamp_min(1)).reshape_as(x)

    masked = torch.where(observed, x, replacement)
    selected = hidden.expand_as(x) & valid

    return masked, observed.float(), selected


def mse_valid(prediction, target, valid):
    values = (prediction - target).square()[valid]
    if values.numel() == 0:
        raise RuntimeError("No valid loss measurements.")
    return values.mean()


def spectral_cosine_loss(prediction, target, valid):
    prediction_original = prediction * scale_t + offset_t
    target_original = target * scale_t + offset_t

    active = valid.flatten(2).any(dim=2)[:, :, None, None]
    usable = (valid | ~active).all(dim=1) & valid.any(dim=1)
    prediction_original = torch.where(active, prediction_original, 0.0)
    target_original = torch.where(active, target_original, 0.0)

    denominator = prediction_original.norm(dim=1) * target_original.norm(dim=1)
    usable &= denominator > 1e-10

    if not usable.any():
        return prediction.sum() * 0

    cosine = (prediction_original * target_original).sum(dim=1) / denominator.clamp_min(1e-10)

    return (1 - cosine.clamp(-1, 1))[usable].mean()


training_sources = []
for product in TRAIN_SAMPLING_PRODUCTS:
    cube = cubes[product]
    if SPLIT_MODE == "spatial":
        train_entry = next(e for e in manifest if e["split"] == "train")
        row_start, row_end = train_entry["first"], train_entry["last"]
    else:
        row_start, row_end = 0, cube.lines
    if row_end - row_start < PATCH or cube.samples < PATCH:
        raise RuntimeError(f"Training area too small for PATCH: {product}")
    count = (row_end - row_start - PATCH + 1) * (cube.samples - PATCH + 1)
    training_sources.append((cube, row_start, row_end, count))

training_patch_counts = [s[3] for s in training_sources]


def sample_batch():
    targets, masks = [], []

    for _ in range(BATCH_SIZE):
        for _ in range(40):
            cube, row_start, row_end, _ = random.choices(
                training_sources, weights=training_patch_counts, k=1
            )[0]
            top = row_start + random.randrange(row_end - row_start - PATCH + 1)
            left = random.randrange(cube.samples - PATCH + 1)
            raw, valid = cube.crop(top, top + PATCH, left, left + PATCH)
            if valid.any(axis=0).mean() > MIN_VALID_FRACTION:
                break
        else:
            raise RuntimeError(
                "Repeated invalid training patches. "
                "Inspect nodata_values, bad_bands and coverage."
            )

        filled = np.where(valid, raw, offset).astype(np.float32)
        normalized = ((filled - offset) / scale).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise RuntimeError(
                f"Non-finite normalized training patch: {cube.product}, "
                f"rows {top}:{top + PATCH}, columns {left}:{left + PATCH}"
            )
        targets.append(torch.from_numpy(normalized))
        masks.append(torch.from_numpy(valid))

    target = torch.stack(targets).to(DEVICE)
    mask = torch.stack(masks).to(DEVICE)
    assert target.shape == (BATCH_SIZE, BANDS, PATCH, PATCH)
    return target, mask


def gradient_norm(module):
    values = [
        parameter.grad.detach().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt().item())


validation_examples = []
validation_generator = torch.Generator(device=DEVICE).manual_seed(9001)

for item in validation_regions:
    _, height, width = item["x"].shape

    for index in range(VAL_PATCHES_PER_REGION):
        fraction = (index + 1) / (VAL_PATCHES_PER_REGION + 1)
        top = int(fraction * (height - PATCH))
        left = int(fraction * (width - PATCH))

        target = item["x"][:, top:top + PATCH, left:left + PATCH].unsqueeze(0).to(DEVICE)
        valid = item["v"][:, top:top + PATCH, left:left + PATCH].unsqueeze(0).to(DEVICE)

        noisy, _ = corrupt(target, generator=validation_generator, fixed=True)

        validation_examples.append({
            "region": item["id"], "target": target, "valid": valid, "noisy": noisy,
        })


@torch.no_grad()
def validate(model):
    model.eval()

    errors = {key: [] for key in ["input", "output", "spatial", "spectral"]}

    for example in validation_examples:
        target = example["target"]
        valid = example["valid"]
        noisy = example["noisy"]

        result = model(noisy)

        errors["input"].append(mse_valid(noisy, target, valid).item())
        for key in ["output", "spatial", "spectral"]:
            errors[key].append(mse_valid(result[key], target, valid).item())

    errors = {key: float(np.mean(values)) for key, values in errors.items()}

    gains = {
        key: 10 * math.log10(max(errors["input"], 1e-20) / max(errors[key], 1e-20))
        for key in ["output", "spatial", "spectral"]
    }

    return errors, gains


model = DualDenoiser().to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=LR / 20
)

configuration = {
    "resolved_config": copy.deepcopy(CFG),
    "training_sampling_products": TRAIN_SAMPLING_PRODUCTS,
    "effective_patch": PATCH,
    "effective_tile": TILE,
    "effective_stride": STRIDE,
    "bands_processed": BANDS,
    "display_bands": DISPLAY_BANDS,
    "torch_version": str(torch.__version__),
    "numpy_version": np.__version__,
    "task": "added_synthetic_noise_restoration",
}

with open(OUT / "configuration.json", "w") as file:
    json.dump(configuration, file, indent=2)

history = []
best_error = float("inf")
best_state = None
bad_epochs = 0

for epoch in range(EPOCHS):
    model.train()

    losses = []
    gradients = [0.0, 0.0, 0.0]
    started = time.time()

    progress = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch + 1:02d}/{EPOCHS}")

    for step in progress:
        target, valid = sample_batch()

        noisy, synthetic_stripes = corrupt(target)

        optimizer.zero_grad(set_to_none=True)

        result = model(noisy)

        fused_loss = mse_valid(result["output"], target, valid)
        spatial_loss = mse_valid(result["spatial"], target, valid)
        spectral_loss = mse_valid(result["spectral"], target, valid)

        stripe_target = synthetic_stripes.mean(dim=2, keepdim=True).expand_as(target)
        stripe_loss = mse_valid(result["stripe"], stripe_target, valid)

        shape_loss = spectral_cosine_loss(result["output"], target, valid)

        loss = (
            fused_loss
            + 0.2 * spatial_loss
            + 0.2 * spectral_loss
            + 0.1 * stripe_loss
            + 0.02 * shape_loss
        )

        if step % MASK_EVERY == 0:
            masked, observed, selected = mask_input(target, valid)

            if selected.any():
                masked_result = model(masked, observed)

                masked_loss = (
                    mse_valid(masked_result["output"], target, selected)
                    + 0.1 * mse_valid(masked_result["spatial"], target, selected)
                    + 0.1 * mse_valid(masked_result["spectral"], target, selected)
                )

                loss = loss + 0.05 * masked_loss

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss. Inspect data and scaling.")

        loss.backward()

        if step == STEPS_PER_EPOCH - 1:
            gradients = [
                gradient_norm(model.spatial),
                gradient_norm(model.spectral),
                gradient_norm(model.stripe),
            ]

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        progress.set_postfix(loss=f"{np.mean(losses):.6f}")

    errors, gains = validate(model)
    scheduler.step()

    record = {
        "epoch": epoch + 1,
        "train_loss": float(np.mean(losses)),
        "validation_output_mse": errors["output"],
        "validation_fused_gain_db": gains["output"],
        "validation_spatial_gain_db": gains["spatial"],
        "validation_spectral_gain_db": gains["spectral"],
        "gradient_spatial": gradients[0],
        "gradient_spectral": gradients[1],
        "gradient_stripe": gradients[2],
        "seconds": time.time() - started,
    }

    history.append(record)
    print(record)

    if errors["output"] < best_error:
        best_error = errors["output"]
        bad_epochs = 0

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        torch.save({
            "state_dict": best_state,
            "offset": offset,
            "scale": scale,
            "log_scale": log_scale,
            "configuration": configuration,
            "manifest": manifest,
            "best_epoch": epoch + 1,
            "best_validation_mse": best_error,
        }, OUT / "best_model.pth")
    else:
        bad_epochs += 1

    torch.save({
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch + 1,
        "history": history,
        "offset": offset,
        "scale": scale,
        "configuration": configuration,
        "manifest": manifest,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, OUT / "latest_training_state.pth")

    with open(OUT / "history.json", "w") as file:
        json.dump(history, file, indent=2)

    if bad_epochs >= PATIENCE:
        break

if best_state is None:
    raise RuntimeError("No valid checkpoint produced.")

model.load_state_dict(best_state)
model.eval()

best_record = min(history, key=lambda item: item["validation_output_mse"])


@torch.no_grad()
def robustness_checks():
    model.eval()

    generator = torch.Generator(device=DEVICE).manual_seed(10101)

    cases = [
        ("identity", None), ("gaussian", 0.01), ("gaussian", 0.03),
        ("gaussian", 0.06), ("gaussian", 0.10), ("impulse", None),
        ("stripe", None), ("mixed", None),
    ]

    records = []

    for mode, sigma in cases:
        input_errors = []
        output_errors = []

        for example in validation_examples:
            target = example["target"]
            valid = example["valid"]

            noisy, _ = corrupt(
                target, generator=generator, fixed=True,
                mode=mode, gaussian_sigma=sigma,
            )

            result = model(noisy)

            input_errors.append(mse_valid(noisy, target, valid).item())
            output_errors.append(mse_valid(result["output"], target, valid).item())

        input_error = float(np.mean(input_errors))
        output_error = float(np.mean(output_errors))

        gain = (
            10 * math.log10(input_error / max(output_error, 1e-20))
            if input_error > 0 else None
        )

        record = {
            "mode": mode, "sigma": sigma,
            "input_mse": input_error, "output_mse": output_error, "gain_db": gain,
        }

        records.append(record)
        print(record)

    with open(OUT / "validation_robustness.json", "w") as file:
        json.dump(records, file, indent=2)

    return records


robustness_records = robustness_checks()


def starts(length, tile, stride):
    if tile > length or stride > tile or stride <= 0:
        raise ValueError("Invalid inference tiling.")

    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


@torch.no_grad()
def infer(cube):
    model.eval()

    channels, height, width = cube.shape

    if channels != BANDS:
        raise RuntimeError(f"Inference requires ALL {BANDS} bands.")

    keys = ["output", "spatial", "spectral", "stripe"]

    accumulators = {key: np.zeros_like(cube, dtype=np.float32) for key in keys}
    weights = np.zeros((height, width), dtype=np.float32)

    hann = np.hanning(TILE).astype(np.float32)
    window = np.maximum(np.outer(hann, hann), 0.05)

    coordinates = [
        (top, left)
        for top in starts(height, TILE, STRIDE)
        for left in starts(width, TILE, STRIDE)
    ]

    for top, left in tqdm(coordinates, desc="tiled inference"):
        tensor = torch.from_numpy(
            np.ascontiguousarray(cube[:, top:top + TILE, left:left + TILE])
        ).unsqueeze(0).to(DEVICE)

        result = model(tensor)

        for key in keys:
            accumulators[key][:, top:top + TILE, left:left + TILE] += (
                result[key][0].cpu().numpy() * window
            )

        weights[top:top + TILE, left:left + TILE] += window

    return {key: array / weights[None] for key, array in accumulators.items()}


def sam_degrees(reference, prediction, valid):
    active = valid.reshape(valid.shape[0], -1).any(axis=1)
    if not active.any():
        return float("nan")

    reference = reference[active]
    prediction = prediction[active]
    valid = valid[active]
    usable = valid.all(axis=0)

    x = reference[:, usable].astype(np.float64)
    y = prediction[:, usable].astype(np.float64)

    denominator = np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0)
    good = denominator > 1e-12

    if not good.any():
        return float("nan")

    cosine = np.sum(x[:, good] * y[:, good], axis=0) / denominator[good]

    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).mean())


def save_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def retain_invalid(array, valid):
    result = array.astype(np.float32).copy()
    result[~valid] = np.nan
    return result


def plot_comparison(item, arrays, labels, filename):
    fig, axes = plt.subplots(
        len(arrays), len(DISPLAY_BANDS),
        figsize=(4.25 * len(DISPLAY_BANDS), 3.5 * len(arrays)),
        squeeze=False,
    )

    for column, band in enumerate(DISPLAY_BANDS):
        low, high = limits(item["raw"][band], item["valid"][band])

        for row, (array, label) in enumerate(zip(arrays, labels)):
            axes[row, column].imshow(
                np.ma.array(array[band], mask=~item["valid"][band]),
                cmap="gray", vmin=low, vmax=high,
                interpolation="nearest", aspect="equal",
            )
            axes[row, column].set_title(f"{label}\nBand index {band}")
            axes[row, column].axis("off")

    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)


test_regions = []
test_metric_rows = []
test_band_rows = []

if RUN_FINAL_TEST or RUN_RAW_TRANSFER:
    test_regions = [prepare_region(item) for item in load_manifest_regions("test")]

if RUN_FINAL_TEST:
    for region_index, item in enumerate(test_regions):
        reference_normalized = item["x"].numpy()
        usable = item["valid"]

        generator = torch.Generator(device="cpu").manual_seed(SEED * 1000 + region_index)

        noisy_tensor, _ = corrupt(item["x"].unsqueeze(0), generator=generator, fixed=True)

        noisy = noisy_tensor[0].numpy()
        result = infer(noisy)

        input_error = float(np.mean(
            (noisy[usable] - reference_normalized[usable]).astype(np.float64) ** 2
        ))

        noisy_original = inverse_scale(noisy)
        input_sam = sam_degrees(item["raw"], noisy_original, usable)

        for key in ["spatial", "spectral", "output"]:
            output_error = float(np.mean(
                (result[key][usable] - reference_normalized[usable]).astype(np.float64) ** 2
            ))

            gain = 10 * math.log10(max(input_error, 1e-20) / max(output_error, 1e-20))

            output_original = inverse_scale(result[key])

            record = {
                "region": item["id"],
                "branch": key,
                "input_mse": input_error,
                "output_mse": output_error,
                "gain_db": gain,
                "input_sam_deg": input_sam,
                "output_sam_deg": sam_degrees(item["raw"], output_original, usable),
            }

            test_metric_rows.append(record)
            print(record)

        for band in range(BANDS):
            valid_band = usable[band]
            if not valid_band.any():
                continue

            input_band_error = float(np.mean(
                (noisy[band][valid_band] - reference_normalized[band][valid_band])
                .astype(np.float64) ** 2
            ))

            output_band_error = float(np.mean(
                (result["output"][band][valid_band]
                 - reference_normalized[band][valid_band]).astype(np.float64) ** 2
            ))

            test_band_rows.append({
                "region": item["id"],
                "band": band,
                "input_mse": input_band_error,
                "output_mse": output_band_error,
                "gain_db": 10 * math.log10(
                    max(input_band_error, 1e-20) / max(output_band_error, 1e-20)
                ),
            })

        restored_original = inverse_scale(result["output"])

        plot_comparison(
            item,
            arrays=[item["raw"], noisy_original, restored_original],
            labels=["Measured reference", "Added mixed corruption", "Restored"],
            filename=f"test_region_{region_index:02d}.png",
        )

        np.savez_compressed(
            OUT / f"test_region_{region_index:02d}.npz",
            reference=retain_invalid(item["raw"], usable),
            noisy=retain_invalid(noisy_original, usable),
            restored=retain_invalid(restored_original, usable),
            valid=usable,
            offset=offset,
            scale=scale,
        )

    save_csv(OUT / "test_region_metrics.csv", test_metric_rows)
    save_csv(OUT / "test_per_band_metrics.csv", test_band_rows)

    fused_rows = [row for row in test_metric_rows if row["branch"] == "output"]
    pooled_input = np.mean([row["input_mse"] for row in fused_rows])
    pooled_output = np.mean([row["output_mse"] for row in fused_rows])

    print(
        "Equal-region aggregated gain:",
        10 * math.log10(max(pooled_input, 1e-20) / max(pooled_output, 1e-20)),
        "dB",
    )

if RUN_RAW_TRANSFER:
    if not 0 <= RAW_TRANSFER_REGION_INDEX < len(test_regions):
        raise ValueError("Invalid raw_transfer_region_index.")

    item = test_regions[RAW_TRANSFER_REGION_INDEX]

    result = infer(item["x"].numpy())
    restored = inverse_scale(result["output"])

    plot_comparison(
        item,
        arrays=[
            item["raw"],
            inverse_scale(result["spatial"]),
            inverse_scale(result["spectral"]),
            restored,
        ],
        labels=[
            "Measured raw HSI",
            "Spatial output on raw",
            "Spectral output on raw",
            "Fused output on raw",
        ],
        filename="raw_transfer.png",
    )

    stripe_original = result["stripe"] * scale
    removed = item["raw"] - restored

    fig, axes = plt.subplots(
        3, len(DISPLAY_BANDS), figsize=(4.25 * len(DISPLAY_BANDS), 11), squeeze=False
    )

    for column, band in enumerate(DISPLAY_BANDS):
        valid_band = item["valid"][band]

        for row, (array, label) in enumerate([
            (stripe_original, "Predicted stripe correction"),
            (removed, "Total removed component"),
        ]):
            amplitude = max(
                float(np.percentile(np.abs(array[band][valid_band]), 99)), 1e-8
            )

            handle = axes[row, column].imshow(
                np.ma.array(array[band], mask=~valid_band),
                cmap="coolwarm", vmin=-amplitude, vmax=amplitude, aspect="equal",
            )
            axes[row, column].set_title(f"{label}\nBand {band}")
            axes[row, column].axis("off")
            fig.colorbar(handle, ax=axes[row, column], fraction=0.046)

        for array, label, color in [
            (item["raw"], "Measured", "gray"),
            (restored, "Restored", "royalblue"),
        ]:
            profile = np.nanmean(np.where(valid_band, array[band], np.nan), axis=0)
            axes[2, column].plot(profile, label=label, color=color)

        axes[2, column].set_title(f"Column profile - band {band}")
        axes[2, column].set_xlabel("ROI column")
        axes[2, column].grid(alpha=0.3)
        axes[2, column].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "raw_transfer_residuals.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    wavelengths = cubes[item["product"]].wavelengths_nm()
    x_axis = wavelengths if wavelengths is not None else np.arange(BANDS)
    x_label = "Wavelength (nm)" if wavelengths is not None else "Band index"

    _, height, width = item["raw"].shape

    points = [
        (height // 4, width // 4),
        (height // 4, 3 * width // 4),
        (3 * height // 4, width // 4),
        (3 * height // 4, 3 * width // 4),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    for ax, (row, column) in zip(axes.flat, points):
        usable = item["valid"][:, row, column]

        for array, label, color in [
            (item["raw"], "Measured", "gray"),
            (restored, "Restored", "royalblue"),
        ]:
            spectrum = np.where(usable, array[:, row, column], np.nan)
            ax.plot(x_axis, spectrum, label=label, color=color)

        ax.set_title(f"ROI pixel ({row}, {column})")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Reflectance / signal")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(OUT / "raw_transfer_spectra.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    np.savez_compressed(
        OUT / "raw_transfer_arrays.npz",
        raw=retain_invalid(item["raw"], item["valid"]),
        restored=retain_invalid(restored, item["valid"]),
        stripe_estimate=retain_invalid(stripe_original, item["valid"]),
        removed=retain_invalid(removed, item["valid"]),
        valid=item["valid"],
        offset=offset,
        scale=scale,
    )

fig, ax = plt.subplots(figsize=(9, 4))

for key, label in [
    ("validation_fused_gain_db", "Fused"),
    ("validation_spatial_gain_db", "Spatial intermediate"),
    ("validation_spectral_gain_db", "Spectral intermediate"),
]:
    ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=label)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("Fixed-validation added-noise gain (dB)")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(OUT / "validation_branch_gains.png", dpi=180, bbox_inches="tight")
plt.show()
plt.close(fig)


def benchmark_metrics(reference, prediction, valid):
    from skimage.metrics import structural_similarity
    from scipy.ndimage import minimum_filter

    psnr_values, ssim_values, relative_errors = [], [], []
    skipped_range = 0
    skipped_mean = 0

    for band in range(reference.shape[0]):
        mask = valid[band] & np.isfinite(reference[band])
        if not mask.any():
            continue
        if not np.isfinite(prediction[band][mask]).all():
            raise RuntimeError("Non-finite benchmark predictions on valid pixels.")

        ref = reference[band].astype(np.float64)
        pred = prediction[band].astype(np.float64)
        mse = float(np.mean((ref[mask] - pred[mask]) ** 2))
        mean = float(np.mean(ref[mask]))

        if abs(mean) > 1e-12:
            relative_errors.append(mse / mean ** 2)
        else:
            skipped_mean += 1

        data_range = float(np.max(ref[mask]) - np.min(ref[mask]))
        if data_range <= 1e-12:
            skipped_range += 1
            continue

        psnr_values.append(
            float("inf") if mse == 0 else 10.0 * math.log10(data_range ** 2 / mse)
        )

        window = min(7, ref.shape[0], ref.shape[1])
        if window % 2 == 0:
            window -= 1
        if window < 3:
            continue

        safe_ref = np.where(mask, ref, 0.0)
        safe_pred = np.where(mask, pred, 0.0)
        _, ssim_map = structural_similarity(
            safe_ref, safe_pred, data_range=data_range, win_size=window,
            channel_axis=None, full=True,
            gaussian_weights=False, use_sample_covariance=True,
        )
        centers = minimum_filter(
            mask.astype(np.uint8), size=window, mode="constant", cval=0
        ).astype(bool)
        if centers.any():
            ssim_values.append(float(np.mean(ssim_map[centers])))

    return {
        "mean_psnr_db": float(np.mean(psnr_values)) if psnr_values else float("nan"),
        "mean_ssim": float(np.mean(ssim_values)) if ssim_values else float("nan"),
        "sam_deg": sam_degrees(reference, prediction, valid),
        "ergas_ratio_1": (
            100.0 * math.sqrt(float(np.mean(relative_errors)))
            if relative_errors else float("nan")
        ),
        "psnr_bands": len(psnr_values),
        "ssim_bands": len(ssim_values),
        "ergas_bands": len(relative_errors),
        "constant_range_bands_excluded": skipped_range,
        "zero_mean_bands_excluded": skipped_mean,
    }


if RUN_FINAL_TEST:
    benchmark_cases = [
        ("identity", "identity", None, 0.002),
        ("gaussian_sigma_0.01", "gaussian", 0.01, 0.002),
        ("gaussian_sigma_0.10", "gaussian", 0.10, 0.002),
        ("impulse_p_0.002", "impulse", None, 0.002),
        ("impulse_p_0.02", "impulse", None, 0.02),
        ("stripes", "stripe", None, 0.002),
        ("mixed_sigma_0.01_impulse_0.002_stripes", "mixed", 0.01, 0.002),
        ("mixed_sigma_0.10_impulse_0.02_stripes", "mixed", 0.10, 0.02),
    ]
    benchmark_rows = []
    benchmark_keys = ["mean_psnr_db", "mean_ssim", "sam_deg", "ergas_ratio_1"]

    for region_index, item in enumerate(test_regions):
        reference = item["raw"]
        valid = item["valid"]

        for case_name, mode, sigma, probability in benchmark_cases:
            seed = SEED * 1000 + region_index
            generator = torch.Generator(device="cpu").manual_seed(seed)

            noisy_tensor, _ = corrupt(
                item["x"].unsqueeze(0), generator=generator, fixed=True,
                mode=mode, gaussian_sigma=sigma, impulse_probability=probability,
            )
            noisy_normalized = noisy_tensor[0].numpy()
            result_benchmark = infer(noisy_normalized)
            noisy_radiance = inverse_scale(noisy_normalized)

            for stage, prediction in [
                ("input", noisy_radiance),
                ("fused_output", inverse_scale(result_benchmark["output"])),
            ]:
                if mode == "identity" and stage == "input":
                    prediction = reference

                benchmark_rows.append({
                    "region": item["id"],
                    "noise_case": case_name,
                    "stage": stage,
                    "seed": seed,
                    "gaussian_sigma": sigma if sigma is not None else "",
                    "impulse_probability": (
                        probability if mode in {"impulse", "mixed"} else 0.0
                    ),
                    **benchmark_metrics(reference, prediction, valid),
                })

            del result_benchmark, noisy_tensor, noisy_normalized, noisy_radiance

    save_csv(OUT / "noise_benchmark_per_region.csv", benchmark_rows)

    summary_rows = []
    for case_name, _, _, _ in benchmark_cases:
        for stage in ["input", "fused_output"]:
            selected = [
                row for row in benchmark_rows
                if row["noise_case"] == case_name and row["stage"] == stage
            ]
            summary = {"noise_case": case_name, "stage": stage, "regions": len(selected)}
            for key in benchmark_keys:
                values = np.asarray([row[key] for row in selected], dtype=np.float64)
                usable = ~np.isnan(values)
                summary[key] = float(np.mean(values[usable])) if usable.any() else float("nan")
                summary[key + "_regions"] = int(usable.sum())
            summary_rows.append(summary)

    save_csv(OUT / "noise_benchmark_summary.csv", summary_rows)

    with open(OUT / "noise_benchmark_protocol.json", "w") as file:
        json.dump({
            "dataset": CFG["name"],
            "reference": "measured data (not noise-free ground truth)",
            "metric_domain": "original units; no clipping or independent rescaling",
            "psnr_ssim_range": "valid reference maximum minus minimum, separately per band and region",
            "mean_psnr_ssim": "arithmetic mean over eligible bands",
            "ssim": "uniform 7x7 windows, sample covariance; only fully valid interior windows",
            "ergas_resolution_ratio": 1,
            "degenerate_bands": "constant reference bands excluded from PSNR/SSIM; near-zero mean bands excluded from ERGAS; counts reported",
            "summary": "equal-region arithmetic means; not pooled cube metrics",
            "noise_domain": "saved training normalization",
            "gaussian": "specified sigma times independent band factor U(0.8,1.2)",
            "impulse": "independent probability per value, equal random sign, magnitude U(0.5,2.0)",
            "stripes": {
                "column_probability": NOISE["column_probability"],
                "band_probability": 0.6,
                "offset_std_multiplier": NOISE["stripe_std"],
                "common_weight": 0.7,
                "independent_weight": 0.3,
                "gain_std": 0.03,
                "column_widening": False,
            },
            "seed": "CPU generator reset to seed*1000 + region_index for every case",
            "cases": benchmark_cases,
        }, file, indent=2)

    for row in summary_rows:
        print(row)

for cube in cubes.values():
    cube.close()
