from pathlib import Path
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

CFG = {
    "name": "ch2_iirs_north_pole",
    "checkpoint": r"C:\Users\Dell\Desktop\HSI Denoise\anurag\output_ch_2\best_model.pth",
    "trust_checkpoint": True,
    "data_root": r"C:\Users\Dell\Desktop\HSI Denoise\Lunar North Pole IIRS Data",
    "export_root": r"C:\Users\Dell\Desktop\HSI Denoise\anurag\exports_ch_2",
    "output_nodata": -999.0,
    "copy_block_lines": 128,
    "expected_bands": 256,
    "tile": None,
    "stride": None,
    "source": {
        "file_glob": "**/*.qub",
        "file_format": None,
        "array_layout": None,
        "array_key": None,
        "data_scale": None,
        "nodata_values": None,
        "bad_bands": None,
        "header_overrides": None,
        "xml_sidecar": True,
        "xml_nodata_tags": ["missing_constant", "invalid_constant", "not_a_number_constant"],
        "xml_wavelength_tag": "center_wavelength",
        "wavelengths": None,
        "wavelength_units": "nm",
    },
    "metadata": {
        "product_dir_is_name": True,
        "copy_sidecar_suffixes": [".hdr", ".xml"],
        "copy_dirs": ["geometry"],
        "copy_files": ["miscellaneous/readme.txt"],
        "quantity_description": "Source calibrated radiance, source units unchanged; consult copied sidecar metadata for exact units",
    },
    "requests": [
        {"product": "ch2_iir_nci_20210112T0149187898_d_img_d32",
         "lines": [47188, 55270], "samples": [1, 250], "role": "training"},
        {"product": "ch2_iir_nci_20210112T0743064200_d_img_d32",
         "lines": [47188, 55270], "samples": [1, 250], "role": "training"},
        {"product": "ch2_iir_nci_20210611T0437452422_d_img_d32",
         "lines": [1, 8083], "samples": [1, 250], "role": "validation"},
        {"product": "ch2_iir_nci_20210611T0833509470_d_img_d32",
         "lines": [1, 8083], "samples": [1, 250], "role": "test"},
    ],
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = Path(CFG["checkpoint"])
DATA_ROOT = Path(CFG["data_root"])
EXPORT_ROOT = Path(CFG["export_root"])
OUTPUT_NODATA = float(CFG["output_nodata"])
COPY_BLOCK_LINES = int(CFG["copy_block_lines"])
SRC = CFG["source"]

ENVI_DTYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8",
               12: "u2", 13: "u4", 14: "i8", 15: "u8"}
ENVI_LAYOUT = {"bsq": "CHW", "bil": "HCW", "bip": "HWC"}
ENVI_SUFFIXES = {".img", ".raw", ".dat", ".bsq", ".bil", ".bip", ".hsi", ".qub", ""}
UM_UNITS = {"micrometers", "micrometer", "micrometres", "micrometre", "um", "µm"}
NM_UNITS = {"nanometers", "nanometer", "nanometres", "nanometre", "nm"}

RESOLVED = {}


def setting(key, default):
    value = SRC.get(key)
    if value is not None:
        return value
    return RESOLVED.get(key, default)


def read_header(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    pairs = re.findall(r"(?mi)^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)", text)
    return {k.strip().lower(): v.strip().strip("{}").strip() for k, v in pairs}


def find_header(path):
    for candidate in (path.with_suffix(".hdr"), path.with_suffix(".HDR"),
                      Path(str(path) + ".hdr")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No ENVI header found for {path}")


def find_sidecar(path, suffix):
    matches = [p for p in path.parent.iterdir()
               if p.is_file() and p.stem.lower() == path.stem.lower()
               and p.suffix.lower() == suffix]
    return matches[0] if len(matches) == 1 else None


def resolve_format(path):
    fmt = setting("file_format", "auto")
    if str(fmt).lower() != "auto":
        return str(fmt).lower()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return "npy"
    if suffix == ".npz":
        return "npz"
    if suffix == ".mat":
        return "mat"
    if suffix in {".tif", ".tiff"}:
        return "tiff"
    if suffix in ENVI_SUFFIXES:
        return "envi"
    raise ValueError(f"Cannot infer format for {path}; set source.file_format.")


def load_array_file(path, fmt):
    key = setting("array_key", None)
    if fmt == "npy":
        return np.load(path, mmap_mode="r")
    if fmt == "npz":
        archive = np.load(path)
        names = list(archive.keys())
        key = key or (names[0] if len(names) == 1 else None)
        if key is None or key not in archive:
            raise KeyError(f"Set source.array_key; available: {names}")
        return archive[key]
    if fmt == "mat":
        try:
            from scipy.io import loadmat
            content = loadmat(path)
            names = [k for k, v in content.items()
                     if not k.startswith("__") and getattr(v, "ndim", 0) == 3]
            key = key or (names[0] if len(names) == 1 else None)
            if key is None or key not in content:
                raise KeyError(f"Set source.array_key; 3-D arrays: {names}")
            return content[key]
        except NotImplementedError:
            import h5py
            with h5py.File(path, "r") as handle:
                names = [k for k in handle.keys() if handle[k].ndim == 3]
                key = key or (names[0] if len(names) == 1 else None)
                if key is None or key not in handle:
                    raise KeyError(f"Set source.array_key; 3-D datasets: {names}")
                return np.array(handle[key])
    if fmt == "tiff":
        import tifffile
        return tifffile.imread(path)
    raise ValueError(f"Unsupported format: {fmt}")


def locate(product):
    matches = sorted(
        p for p in DATA_ROOT.glob(SRC["file_glob"])
        if p.is_file() and p.stem.lower() == product.lower()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one file for '{product}' under {DATA_ROOT} "
            f"matching {SRC['file_glob']}; found {list(map(str, matches))}"
        )
    return matches[0]


class SourceCube:
    def __init__(self, product):
        self.product = product
        self.path = locate(product)
        self.format = resolve_format(self.path)
        self.header = None
        self.metadata = {}
        self.xml = find_sidecar(self.path, ".xml") if SRC["xml_sidecar"] else None
        self.xml_root = ET.parse(self.xml).getroot() if self.xml else None
        self.nodata = [float(v) for v in setting("nodata_values", [])]
        good_override = None

        if self.format == "envi":
            self.header = find_header(self.path)
            metadata = read_header(self.header)
            overrides = setting("header_overrides", {}) or {}
            metadata.update(overrides.get("*", {}))
            metadata.update(overrides.get(product, {}))
            self.metadata = metadata
            required = ["samples", "lines", "bands", "data type",
                        "byte order", "interleave", "header offset"]
            missing = [k for k in required if k not in metadata]
            if missing:
                raise RuntimeError(
                    f"Missing {missing} in {self.header}; add them under "
                    "source.header_overrides instead of guessing the layout."
                )
            samples, lines, bands = (int(metadata[k]) for k in ("samples", "lines", "bands"))
            offset = int(metadata["header offset"])
            order = int(metadata["byte order"])
            code = int(metadata["data type"])
            interleave = metadata["interleave"].strip().lower()
            if order not in (0, 1) or code not in ENVI_DTYPES or interleave not in ENVI_LAYOUT:
                raise RuntimeError(f"Unsupported ENVI settings in {self.header}")
            dtype = np.dtype(("<" if order == 0 else ">") + ENVI_DTYPES[code])
            sizes = {"C": bands, "H": lines, "W": samples}
            layout = ENVI_LAYOUT[interleave]
            shape = tuple(sizes[a] for a in layout)
            if offset < 0 or min(shape) <= 0:
                raise RuntimeError("Invalid source dimensions or offset")
            if self.path.stat().st_size != offset + math.prod(shape) * dtype.itemsize:
                raise RuntimeError(f"Binary file size mismatch: {self.path}")
            if "data ignore value" in metadata:
                self.nodata.append(float(metadata["data ignore value"]))
            if "bbl" in metadata:
                bbl = np.asarray([float(v) for v in metadata["bbl"].split(",") if v.strip()])
                if bbl.shape != (bands,) or not np.isfinite(bbl).all():
                    raise RuntimeError(f"Invalid BBL in {self.header}")
                good_override = bbl != 0
            self.layout = layout
            self.array = np.memmap(self.path, mode="r", dtype=dtype, offset=offset, shape=shape)
        else:
            array = load_array_file(self.path, self.format)
            layout = str(setting("array_layout", "HWC")).upper()
            if array.ndim != 3 or sorted(layout) != ["C", "H", "W"]:
                raise RuntimeError(
                    f"{self.path.name}: need a 3-D array and a C/H/W layout; "
                    f"got shape {array.shape}, layout {layout}"
                )
            self.layout = layout
            self.array = array

        dims = dict(zip(self.layout, self.array.shape))
        self.bands, self.lines, self.samples = dims["C"], dims["H"], dims["W"]

        if self.xml_root is not None:
            tags = {t.lower() for t in SRC["xml_nodata_tags"]}
            for element in self.xml_root.iter():
                if element.tag.rsplit("}", 1)[-1].lower() in tags:
                    try:
                        self.nodata.append(float(element.text.strip()))
                    except (ValueError, AttributeError):
                        pass

        self.good_bands = good_override if good_override is not None else np.ones(self.bands, bool)
        for index in setting("bad_bands", []):
            if not 0 <= index < self.bands:
                raise RuntimeError(f"bad_bands index out of range: {index}")
            self.good_bands[index] = False

        self.scale_factor = float(setting("data_scale", 1.0))

    def read(self, top, stop, left, right):
        if not (0 <= top < stop <= self.lines and 0 <= left < right <= self.samples):
            raise RuntimeError(f"Requested crop exceeds source bounds: {self.product}")
        index = {"H": slice(top, stop), "W": slice(left, right), "C": slice(None)}
        sub = self.array[tuple(index[a] for a in self.layout)]
        x = np.array(np.transpose(sub, [self.layout.index(a) for a in "CHW"]),
                     dtype=np.float32, copy=True)
        valid = np.isfinite(x)
        for value in self.nodata:
            valid &= x != value
        valid &= self.good_bands[:, None, None]
        if self.scale_factor != 1.0:
            x = x / self.scale_factor
        return x, valid

    def wavelengths(self):
        values, unit = None, SRC["wavelength_units"]
        if self.xml_root is not None and SRC["xml_wavelength_tag"]:
            tag = SRC["xml_wavelength_tag"].lower()
            found = []
            for element in self.xml_root.iter():
                if element.tag.rsplit("}", 1)[-1].lower() == tag:
                    value = float(element.text.strip())
                    element_unit = element.attrib.get("unit", "").lower().strip() or unit
                    found.append(value * 1000 if element_unit in UM_UNITS else value)
                    if element_unit not in UM_UNITS | NM_UNITS:
                        raise RuntimeError(f"Unrecognized XML wavelength unit: {element_unit!r}")
            if found:
                values = np.asarray(found, dtype=np.float64)
        if values is None and "wavelength" in self.metadata:
            values = np.asarray([float(v) for v in self.metadata["wavelength"].split(",") if v.strip()])
            header_unit = self.metadata.get("wavelength units", unit).strip().strip('"').lower()
            if header_unit in UM_UNITS:
                values = values * 1000
            elif header_unit not in NM_UNITS:
                raise RuntimeError(f"Unrecognized wavelength units: {header_unit!r}")
        if values is None and SRC["wavelengths"] is not None:
            values = np.asarray(SRC["wavelengths"], dtype=np.float64)
            if str(unit).lower() in UM_UNITS:
                values = values * 1000
        if values is None:
            return None
        if values.shape != (self.bands,) or not np.isfinite(values).all():
            raise RuntimeError(f"Expected {self.bands} valid wavelengths; found {values.size}")
        return values

    def close(self):
        if isinstance(self.array, np.memmap):
            self.array._mmap.close()


def starts(length, tile, stride):
    if not (0 < stride <= tile <= length):
        raise RuntimeError("Invalid tiling settings")
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def write_envi_header(path, height, width, bands, wavelengths, mask=False):
    text = ("ENVI\n"
            "description = {Crop on original pixel grid; no map projection assigned. "
            "See processing_metadata.json.}\n"
            f"samples = {width}\nlines = {height}\nbands = {bands}\n"
            f"header offset = 0\nfile type = ENVI Standard\ndata type = {1 if mask else 4}\n"
            "interleave = bil\nbyte order = 0\n")
    if not mask:
        text += f"data ignore value = {OUTPUT_NODATA}\n"
    if wavelengths is not None:
        text += ("wavelength units = Nanometers\nwavelength = {"
                 + ", ".join(f"{v:.10g}" for v in wavelengths) + "}\n")
    path.write_text(text, encoding="utf-8")


def product_root(cube):
    candidate = DATA_ROOT / cube.product
    if CFG["metadata"]["product_dir_is_name"] and candidate.is_dir():
        return candidate
    return cube.path.parent


def copy_source_metadata(cube, dest):
    target = dest / "source_metadata"
    target.mkdir()
    root = product_root(cube)
    for suffix in CFG["metadata"]["copy_sidecar_suffixes"]:
        sidecar = find_sidecar(cube.path, suffix.lower())
        if sidecar is not None:
            shutil.copy2(sidecar, target / sidecar.name)
    copied_dirs = []
    for name in CFG["metadata"]["copy_dirs"]:
        source_dir = root / name
        if source_dir.is_dir():
            shutil.copytree(source_dir, target / f"full_source_{name}")
            copied_dirs.append(str(target / f"full_source_{name}"))
    for pattern in CFG["metadata"]["copy_files"]:
        for path in root.glob(pattern):
            if path.is_file():
                shutil.copy2(path, target / path.name)
    return copied_dirs


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
        return self.base(F.silu(x)) + torch.einsum("...ik,oik->...o", basis, self.coefficients)


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
    def __init__(self, width, blocks):
        super().__init__()
        self.stem = nn.Conv2d(2, width, 3, padding=1, padding_mode="reflect")
        self.blocks = nn.Sequential(*[SpatialBlock(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, observed):
        batch, channels, height, width = x.shape
        images = x.reshape(batch * channels, 1, height, width)
        flags = observed.expand_as(x).reshape(batch * channels, 1, height, width)
        features = self.blocks(self.stem(torch.cat([images, flags], dim=1)))
        return self.head(features).reshape(batch, channels, height, width)


class SpectralBranch(nn.Module):
    def __init__(self, state_size, chunk, band_coordinate, log_scale):
        super().__init__()
        self.state_size = state_size
        self.chunk = chunk
        self.parameter_kan = ChebyKAN(4, 3 * state_size)
        self.log_rates = nn.Parameter(torch.linspace(-2, 1, state_size))
        self.head = ChebyKAN(2 * state_size + 1, 1)
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
            values, observed,
            self.coordinate.expand(number, channels),
            self.scale_metadata.expand(number, channels),
        ], dim=-1)
        drive, steps, gates = self.parameter_kan(tokens).chunk(3, dim=-1)
        drive = torch.tanh(drive)
        steps = F.softplus(steps) + 1e-3
        gates = torch.sigmoid(gates)
        forward = self.scan(drive, steps)
        backward = self.scan(drive, steps, reverse=True)
        features = torch.cat([gates * forward, gates * backward, values.unsqueeze(-1)], dim=-1)
        return self.head(features).squeeze(-1)

    def forward(self, x, observed):
        batch, channels, height, width = x.shape
        values = x.permute(0, 2, 3, 1).reshape(-1, channels)
        flags = observed.expand_as(x).permute(0, 2, 3, 1).reshape(-1, channels)
        outputs = [
            self.sequences(values[s:s + self.chunk], flags[s:s + self.chunk])
            for s in range(0, len(values), self.chunk)
        ]
        return torch.cat(outputs, dim=0).reshape(batch, height, width, channels).permute(0, 3, 1, 2)


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
        profiles = torch.stack([x.mean(dim=2), x.median(dim=2).values], dim=2).reshape(
            batch * channels, 2, width
        )
        offsets = self.net(profiles).reshape(batch, channels, 1, width)
        return offsets.expand(batch, channels, height, width)


class DualDenoiser(nn.Module):
    def __init__(self, bands, width, blocks, state_size, chunk, band_coordinate, log_scale):
        super().__init__()
        self.bands = bands
        self.spatial = SpatialBranch(width, blocks)
        self.spectral = SpectralBranch(state_size, chunk, band_coordinate, log_scale)
        self.stripe = StripeHead()
        self.gate = nn.Conv2d(3, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x, observed=None):
        if x.ndim != 4:
            raise ValueError("Expected [batch,bands,height,width].")
        if x.shape[1] != self.bands:
            raise RuntimeError(f"Expected {self.bands} bands; received {x.shape[1]}.")
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


def export_product(model, checkpoint, cube, request, offset, scale, tile, stride, bands):
    product = request["product"]
    role = request.get("role", "unspecified")
    line_first, line_last = request.get("lines") or [1, cube.lines]
    sample_first, sample_last = request.get("samples") or [1, cube.samples]
    height, width = line_last - line_first + 1, sample_last - sample_first + 1
    top0, left0 = line_first - 1, sample_first - 1
    tile = min(tile, height, width)
    stride = min(stride, tile)
    waves = cube.wavelengths()
    dest = EXPORT_ROOT / product
    if dest.exists():
        raise FileExistsError(
            f"Refusing to overwrite {dest}. Rename an old/incomplete export before rerunning."
        )
    dest.mkdir()
    (dest / "INCOMPLETE.txt").write_text("Export in progress. Do not share until COMPLETE.txt exists.\n")
    shape = (height, bands, width)
    arrays = []
    try:
        original = np.memmap(dest / "original_radiance.img", mode="w+", dtype="<f4", shape=shape)
        arrays.append(original)
        restored = np.memmap(dest / "denoised_radiance.img", mode="w+", dtype="<f4", shape=shape)
        arrays.append(restored)
        valid_map = np.memmap(dest / "valid_mask.img", mode="w+", dtype="u1", shape=shape)
        arrays.append(valid_map)
        weights = np.memmap(dest / "temporary_weights.dat", mode="w+", dtype="<f4", shape=(height, width))
        arrays.append(weights)
        restored[:] = 0
        weights[:] = 0

        for y in tqdm(range(0, height, COPY_BLOCK_LINES), desc=f"Copy original {product}"):
            end = min(y + COPY_BLOCK_LINES, height)
            raw, valid = cube.read(top0 + y, top0 + end, left0, sample_last)
            if (valid & (raw == OUTPUT_NODATA)).any():
                raise RuntimeError("output_nodata conflicts with a valid source value; choose another.")
            original[y:end] = np.where(valid, raw, OUTPUT_NODATA).transpose(1, 0, 2)
            valid_map[y:end] = valid.transpose(1, 0, 2)
        original.flush()
        valid_map.flush()

        hann = np.hanning(tile).astype(np.float32)
        window = np.maximum(np.outer(hann, hann), 0.05).astype(np.float32)
        coordinates = [(y, x) for y in starts(height, tile, stride) for x in starts(width, tile, stride)]

        with torch.inference_mode():
            for y, x in tqdm(coordinates, desc=f"Denoise {product}"):
                raw = np.array(original[y:y + tile, :, x:x + tile].transpose(1, 0, 2), copy=True)
                valid = np.asarray(valid_map[y:y + tile, :, x:x + tile].transpose(1, 0, 2), dtype=bool)
                normalized = ((np.where(valid, raw, offset) - offset) / scale).astype(np.float32)
                if not np.isfinite(normalized).all():
                    raise RuntimeError("Nonfinite normalized input")
                tensor = torch.from_numpy(np.ascontiguousarray(normalized)).unsqueeze(0).to(DEVICE)
                fused = model(tensor)["output"][0].cpu().numpy()
                if not np.isfinite(fused).all():
                    raise RuntimeError("Nonfinite model output")
                restored[y:y + tile, :, x:x + tile] += (fused * window).transpose(1, 0, 2)
                weights[y:y + tile, x:x + tile] += window

        for y in tqdm(range(0, height, COPY_BLOCK_LINES), desc="Restore physical units"):
            end = min(y + COPY_BLOCK_LINES, height)
            weight = np.asarray(weights[y:end])
            if not (weight > 0).all():
                raise RuntimeError("Uncovered output pixels")
            normalized = np.array(restored[y:end].transpose(1, 0, 2), copy=True) / weight[None]
            physical = normalized * scale + offset
            valid = np.asarray(valid_map[y:end].transpose(1, 0, 2), dtype=bool)
            if not np.isfinite(physical[valid]).all():
                raise RuntimeError("Nonfinite physical output")
            restored[y:end] = np.where(valid, physical, OUTPUT_NODATA).transpose(1, 0, 2)
        for arr in arrays:
            arr.flush()
    finally:
        for arr in arrays:
            arr._mmap.close()

    (dest / "temporary_weights.dat").unlink()
    for name in ("original_radiance", "denoised_radiance", "valid_mask"):
        write_envi_header(dest / (name + ".hdr"), height, width, bands, waves, mask=name == "valid_mask")

    wave_column = waves if waves is not None else np.full(bands, np.nan)
    np.savetxt(dest / "wavelengths.csv", np.column_stack((np.arange(bands), wave_column)),
               delimiter=",", header="band_index_zero_based,wavelength_nm", comments="",
               fmt=["%d", "%.10g"])
    np.savez(dest / "normalization.npz", offset=offset, scale=scale)
    geometry_locations = copy_source_metadata(cube, dest)

    metadata = {
        "dataset": CFG["name"],
        "source_product": product,
        "source_file": str(cube.path),
        "source_role_in_model_development": role,
        "source_line_range_1_based_inclusive": [line_first, line_last],
        "source_sample_range_1_based_inclusive": [sample_first, sample_last],
        "crop_row_mapping": "source line = crop zero-based row + first source line",
        "crop_column_mapping": "source sample = crop zero-based column + first source sample",
        "lines": height, "samples": width, "bands": bands,
        "data_type": "little-endian float32", "interleave": "bil",
        "physical_quantity": CFG["metadata"]["quantity_description"],
        "source_data_scale_divisor": cube.scale_factor,
        "wavelength_units": "nm" if waves is not None else None,
        "checkpoint": str(CHECKPOINT),
        "best_epoch": checkpoint.get("best_epoch"),
        "normalization": "saved checkpoint offset and scale; no refit",
        "tile": tile, "stride": stride,
        "blending": "Hann outer product with minimum weight 0.05",
        "synthetic_noise_added": False,
        "radiometric_or_spectral_clipping": False,
        "spatial_resampling": False,
        "output_nodata": OUTPUT_NODATA,
        "source_declared_nodata": cube.nodata,
        "bad_bands_marked_invalid": np.flatnonzero(~cube.good_bands).tolist(),
        "valid_mask": "1 valid, 0 invalid, separately for each band",
        "projection": "Not assigned. Outputs retain source pixel indices, not a geographic map grid.",
        "geometry": "Copied full-source geometry, if any, is not cropped, interpolated, or applied.",
        "full_source_geometry_directories": geometry_locations,
        "scientific_warning": "Model reconstruction is not noise-free ground truth. No reflectance, thermal or photometric correction is performed. Compare original and denoised spectra.",
        "configuration": checkpoint.get("configuration"),
    }
    (dest / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (dest / "README.txt").write_text(
        "PAIRED ORIGINAL AND DENOISED CROP\n\n"
        "Open each IMG using its matching ENVI HDR. All bands remain in source order.\n"
        "original_radiance: measured source values. No synthetic corruption.\n"
        "denoised_radiance: fused model output restored using saved training normalization.\n"
        f"Invalid values are {OUTPUT_NODATA:g}; valid_mask has 1=valid, 0=invalid per band.\n"
        "Both cubes have identical dimensions and source pixel alignment.\n"
        "No map projection, resampling, or radiometric/photometric correction was performed.\n"
        "Copied source metadata describes the FULL SOURCE, not the exported crop.\n"
        "Use the recorded crop indices to map back to the source.\n"
        "Denoising can alter real spectral features. Inspect both versions in scientific use.\n"
        "Only share or use a package after COMPLETE.txt exists.\n", encoding="utf-8")
    (dest / "INCOMPLETE.txt").unlink()
    (dest / "COMPLETE.txt").write_text("Paired export completed successfully.\n", encoding="utf-8")
    return metadata


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(DATA_ROOT)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu",
                            weights_only=not CFG["trust_checkpoint"])
    cfg = checkpoint["configuration"]
    RESOLVED.update(cfg.get("resolved_config", {}))

    def stored(key, default=None):
        return cfg.get(key, RESOLVED.get(key, default))

    bands = int(stored("bands_processed"))
    if CFG["expected_bands"] is not None and bands != int(CFG["expected_bands"]):
        raise RuntimeError(
            f"Checkpoint has {bands} bands; expected_bands is {CFG['expected_bands']}."
        )

    tile = int(CFG["tile"] or stored("effective_tile", stored("tile")))
    stride = int(CFG["stride"] or stored("effective_stride", stored("stride")))

    offset = np.asarray(checkpoint["offset"], dtype=np.float32).reshape(bands, 1, 1)
    scale = np.asarray(checkpoint["scale"], dtype=np.float32).reshape(bands, 1, 1)
    if not np.isfinite(offset).all() or not np.isfinite(scale).all() or not (scale > 0).all():
        raise RuntimeError("Invalid saved normalization")

    log_scale = np.asarray(checkpoint["log_scale"], dtype=np.float32)
    band_coordinate = np.linspace(-1, 1, bands, dtype=np.float32)

    model = DualDenoiser(
        bands=bands,
        width=int(stored("width")),
        blocks=int(stored("spatial_blocks")),
        state_size=int(stored("state_size")),
        chunk=int(stored("spectral_chunk", 512)),
        band_coordinate=band_coordinate,
        log_scale=log_scale,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    required = 0
    for request in CFG["requests"]:
        product = request["product"]
        if (EXPORT_ROOT / product).exists():
            raise FileExistsError(f"Export folder already exists: {EXPORT_ROOT / product}. Nothing overwritten.")
        cube = SourceCube(product)
        try:
            if cube.bands != bands:
                raise RuntimeError(f"Source/checkpoint band mismatch for {product}: {cube.bands} vs {bands}")
            first, last = request.get("lines") or [1, cube.lines]
            left, right = request.get("samples") or [1, cube.samples]
            if not (1 <= first <= last <= cube.lines and 1 <= left <= right <= cube.samples):
                raise RuntimeError(
                    f"Requested bounds invalid for {product}: source is "
                    f"{cube.lines} lines by {cube.samples} samples"
                )
            cube.wavelengths()
            h, w = last - first + 1, right - left + 1
            local_tile = min(tile, h, w)
            starts(h, local_tile, min(stride, local_tile))
            starts(w, local_tile, min(stride, local_tile))
            required += h * w * bands * 9
        finally:
            cube.close()

    required += 1_000_000_000
    if shutil.disk_usage(EXPORT_ROOT).free < required:
        raise RuntimeError(
            f"Insufficient free disk space: need about {required / 1e9:.2f} GB."
        )

    manifest = []
    for request in CFG["requests"]:
        cube = SourceCube(request["product"])
        try:
            manifest.append(export_product(
                model, checkpoint, cube, request, offset, scale, tile, stride, bands))
        finally:
            cube.close()
        (EXPORT_ROOT / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
