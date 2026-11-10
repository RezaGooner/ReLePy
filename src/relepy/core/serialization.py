"""Pickle-free model files.

A ``.relepy`` file is a ZIP archive with

* ``metadata.json`` -- format version, ReLePy version, algorithm, config, progress counters,
* ``state.json`` -- the structure of the learned state (nested dicts / lists / scalars) in
  which every array is replaced by a reference ``{"__array__": "<name>"}``,
* ``arrays/<name>.npy`` -- the arrays themselves, read with ``allow_pickle=False``.

Loading therefore never executes code from the file. Tensors (PyTorch) are stored as NumPy
arrays and restored as tensors; the structure supports the nested ``state_dict`` of modules
and optimizers, including integer keys and tuples.

Files written by ReLePy 0.1 (plain ``pickle``) are refused unless ``allow_legacy_pickle=True``.
"""

from __future__ import annotations

import io
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np

FORMAT_VERSION = 1
PathLike = Union[str, Path]


class LegacyFormatError(ValueError):
    """Raised when a pickle file from ReLePy 0.1 is loaded without opting in."""


def _is_tensor(obj: Any) -> bool:
    return type(obj).__module__.startswith("torch") and hasattr(obj, "detach")


# --------------------------------------------------------------------------- encoding
def _encode(obj: Any, arrays: Dict[str, np.ndarray]) -> Any:
    if _is_tensor(obj):
        name = f"a{len(arrays)}"
        arrays[name] = obj.detach().cpu().numpy()
        return {"__array__": name, "tensor": True}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            raise TypeError("Object arrays cannot be saved without pickle")
        name = f"a{len(arrays)}"
        arrays[name] = obj
        return {"__array__": name, "tensor": False}
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {"__dict__": [[_encode(k, arrays), _encode(v, arrays)] for k, v in obj.items()]}
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(v, arrays) for v in obj]}
    if isinstance(obj, list):
        return [_encode(v, arrays) for v in obj]
    raise TypeError(f"Cannot serialize object of type {type(obj).__name__}")


def _decode(obj: Any, arrays: Dict[str, np.ndarray]) -> Any:
    if isinstance(obj, list):
        return [_decode(v, arrays) for v in obj]
    if not isinstance(obj, dict):
        return obj
    if "__array__" in obj:
        arr = arrays[obj["__array__"]]
        if obj.get("tensor"):
            import torch

            return torch.from_numpy(np.array(arr))
        return arr
    if "__tuple__" in obj:
        return tuple(_decode(v, arrays) for v in obj["__tuple__"])
    if "__dict__" in obj:
        return {_decode(k, arrays): _decode(v, arrays) for k, v in obj["__dict__"]}
    raise ValueError(f"Malformed state entry: {sorted(obj)}")


# ------------------------------------------------------------------------- public API
def save_archive(path: PathLike, metadata: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Write ``metadata`` (JSON-serializable) and ``state`` (may contain arrays) to ``path``."""
    arrays: Dict[str, np.ndarray] = {}
    structure = _encode(state, arrays)
    meta = {"format_version": FORMAT_VERSION, **metadata}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(meta, indent=2))
        zf.writestr("state.json", json.dumps(structure))
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            zf.writestr(f"arrays/{name}.npy", buf.getvalue())


def load_archive(
    path: PathLike, allow_legacy_pickle: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(metadata, state)`` from a file written by :func:`save_archive`."""
    if not zipfile.is_zipfile(path):
        return _load_legacy(path, allow_legacy_pickle)
    with zipfile.ZipFile(path, "r") as zf:
        meta = json.loads(zf.read("metadata.json"))
        version = meta.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported .relepy format version {version!r}")
        structure = json.loads(zf.read("state.json"))
        arrays = {}
        for name in zf.namelist():
            if name.startswith("arrays/") and name.endswith(".npy"):
                with zf.open(name) as f:
                    arrays[name[len("arrays/") : -len(".npy")]] = np.load(
                        io.BytesIO(f.read()), allow_pickle=False
                    )
    return meta, _decode(structure, arrays)


def read_metadata(path: PathLike) -> Dict[str, Any]:
    """Read only the metadata of a ``.relepy`` file (no arrays are loaded)."""
    if not zipfile.is_zipfile(path):
        raise LegacyFormatError(f"{path} is not a ReLePy >= 0.2 archive")
    with zipfile.ZipFile(path, "r") as zf:
        return json.loads(zf.read("metadata.json"))


def _load_legacy(path: PathLike, allow: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not allow:
        raise LegacyFormatError(
            f"{path} is not a ReLePy >= 0.2 archive. Files saved by ReLePy 0.1 use pickle, "
            "which can execute arbitrary code. If you trust this file, load it with "
            "allow_legacy_pickle=True and save it again to convert it."
        )
    with open(path, "rb") as f:
        payload = pickle.load(f)  # noqa: S301 -- explicit opt-in above
    meta = {k: v for k, v in payload.items() if k != "state"}
    return meta, payload["state"]
