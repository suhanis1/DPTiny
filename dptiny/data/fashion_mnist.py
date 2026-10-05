"""Fashion-MNIST loader for DPTiny.

Interface-compatible with ``dptiny.data.mnist.get_mnist``: same arguments,
same return order, same shapes and dtypes. Files are downloaded from the
project's GitHub on first use and cached under ``~/.cache/dptiny``.
The IDX parser is written with NumPy only.
"""
import gzip
import os
import tempfile
import urllib.request
import zlib

import numpy as np

BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

_UINT8_CODE = 0x08


# --------------------------------------------------------------------------
# IDX parsing
# --------------------------------------------------------------------------
def parse_idx(raw):
    """Parse the bytes of an *uncompressed* IDX file into a uint8 ndarray.

    Header layout: 2 zero bytes, 1 data-type code, 1 byte for the number of
    dimensions, then one big-endian uint32 per dimension. Raw data follows.

    Raises ValueError if the file is not a uint8 IDX file or if its size does
    not match the size announced by the header.
    """
    raw = bytes(raw)
    if len(raw) < 4:
        raise ValueError("IDX data too short to contain a header")
    if raw[0] != 0 or raw[1] != 0:
        raise ValueError("Invalid IDX magic number: first two bytes must be zero")
    if raw[2] != _UINT8_CODE:
        raise ValueError(
            f"Unsupported IDX data type 0x{raw[2]:02x}; only uint8 (0x08) is supported"
        )
    ndim = raw[3]
    if ndim < 1:
        raise ValueError("IDX header declares zero dimensions")
    header_size = 4 + 4 * ndim
    if len(raw) < header_size:
        raise ValueError("IDX data too short to contain all dimension sizes")
    dims = tuple(int(d) for d in np.frombuffer(raw, dtype=">u4", count=ndim, offset=4))
    expected = int(np.prod(dims, dtype=np.uint64))
    actual = len(raw) - header_size
    if actual != expected:
        raise ValueError(
            f"IDX size mismatch: header declares {expected} bytes of data "
            f"for shape {dims}, but file contains {actual}"
        )
    return np.frombuffer(raw, dtype=np.uint8, offset=header_size).reshape(dims)


def load_idx_gz(path):
    """Read a gzipped IDX file from ``path`` and return its uint8 array."""
    try:
        with gzip.open(path, "rb") as f:
            raw = f.read()
    except (OSError, EOFError, zlib.error) as exc:
        raise ValueError(f"{path} is not a valid gzip file: {exc}") from exc
    return parse_idx(raw)


# --------------------------------------------------------------------------
# Download and cache
# --------------------------------------------------------------------------
def _cache_dir(data_home):
    if data_home is None:
        data_home = os.path.join(os.path.expanduser("~"), ".cache", "dptiny")
    path = os.path.join(os.path.expanduser(str(data_home)), "fashion-mnist")
    os.makedirs(path, exist_ok=True)
    return path


def _download(url, dest):
    """Download ``url`` to ``dest`` atomically.

    Data is written to a temporary file in the same directory and moved into
    place with ``os.replace`` only after the transfer completes, so an
    interrupted download never leaves a partial file at ``dest``.
    """
    directory = os.path.dirname(dest)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as resp:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_path, dest)
    except BaseException:  # includes KeyboardInterrupt
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _fetch(name, cache):
    dest = os.path.join(cache, FILES[name])
    if not os.path.exists(dest):
        _download(BASE_URL + FILES[name], dest)
    return dest


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def get_fashion_mnist(normalize=True, flatten=True, data_home=None):
    """Load Fashion-MNIST.

    Returns ``(X_train, X_test, y_train, y_test)``:
      * images: float32, in [0, 1] if ``normalize`` else [0, 255];
        shape (N, 784) if ``flatten`` else (N, 1, 28, 28)
      * labels: int32 in [0, 9]
    The split is the standard 60,000 training / 10,000 test images.
    """
    cache = _cache_dir(data_home)
    X_train = load_idx_gz(_fetch("train_images", cache))
    y_train = load_idx_gz(_fetch("train_labels", cache))
    X_test = load_idx_gz(_fetch("test_images", cache))
    y_test = load_idx_gz(_fetch("test_labels", cache))

    for X, y, n in ((X_train, y_train, 60000), (X_test, y_test, 10000)):
        if X.shape != (n, 28, 28) or y.shape != (n,):
            raise ValueError(
                f"Unexpected Fashion-MNIST shapes: images {X.shape}, labels {y.shape}"
            )

    def prep(X):
        X = X.astype(np.float32)
        if normalize:
            X /= 255.0
        return X.reshape(len(X), -1) if flatten else X.reshape(len(X), 1, 28, 28)

    return prep(X_train), prep(X_test), y_train.astype(np.int32), y_test.astype(np.int32)
