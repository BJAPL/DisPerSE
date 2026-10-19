#!/usr/bin/env python3
r"""filaments3d.py — informe de RESULTADOS (solo gráficos y tablas, sin análisis) de filamentos 3D de DisPerSE en UN SOLO ARCHIVO.

Autosuficiente: solo necesita Python >= 3.10 y numpy, scipy, pandas, matplotlib (en Colab ya están).
Archivo generado automáticamente por tools/build_single_file.py a partir del paquete de desarrollo.

Informe de resultados: SOLO gráficos y tablas, sin ningún análisis ni interpretación.

Entrada: uno o más archivos NDskl_ascii de DisPerSE (cualquier nombre, cualquier cantidad).
Salida (carpeta nueva por ejecución, nunca se sobrescribe)::

    <out>/<nombre>/
        INDICE.md                 lista de todo lo generado
        definiciones.md           definición de cada columna/magnitud (no es análisis)
        tablas/                   CSV: filamentos_<etiqueta>.csv, puentes_..., nodos_..., resumen_red.csv,
                                  estadisticos_descriptivos.csv, top_diferencias_longitud.csv, resumen_<etiqueta>.json
        figuras/<etiqueta>/       una figura por tipo (png y pdf) para cada archivo
        figuras/comparacion/      las mismas figuras con todos los archivos lado a lado
        metadatos/                run_manifest.json (hashes SHA-256, versiones, comando), advertencias.txt

Magnitudes (idénticas a las del paquete de desarrollo): L = Σ|p_{i+1}-p_i|; D = |p_N-p_1|; C_eff = L/D (NaN si D=0,
nunca 0); orientación del eje principal plegada a v_z>=0; tensor del conjunto de puntos del arco.

Uso:
    python filaments3d.py archivo1.NDskl archivo2.NDskl ... --units-in kpc/h --units-out Mpc/h
    python filaments3d.py "carpeta/*.NDskl*" --out resultados
    python filaments3d.py A=ruta_a.NDskl B=ruta_b.NDskl          # etiquetas propias
    python filaments3d.py "*.NDskl*" --units-in kpc/h --units-out Mpc/h --animar   # SOLO animaciones (modo aparte, lento)
"""
from __future__ import annotations

__version__ = "0.2.0"

import argparse
import datetime as dt
import glob
import hashlib
import json
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform
import re
import sys
import time
import traceback
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from matplotlib import cm
from matplotlib import colors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as _cc
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union


# ====================================================================================================
# MÓDULO: units
# ====================================================================================================
# Conversión de unidades de longitud, incluyendo las que llevan el factor 1/h.
#
# Una longitud ``x`` en ``kpc/h`` equivale a ``x / h`` kpc físicos. Convertir
# entre unidades con y sin ``h`` requiere el valor de ``h`` (por defecto el de
# MDPL2, 0.678, que debe declararse en la configuración); convertir entre
# unidades del mismo tipo (p. ej. ``kpc/h`` → ``Mpc/h``) no depende de ``h``.


_TO_KPC = {"pc": 1e-3, "kpc": 1.0, "mpc": 1e3, "gpc": 1e6}


class UnitError(ValueError):
    pass


def parse_unit(unit: str) -> Tuple[float, bool]:
    """Devuelve ``(factor_a_kpc, lleva_h)`` para ``'Mpc/h'``, ``'kpc'``, etc."""
    u = unit.strip().lower().replace(" ", "")
    has_h = u.endswith("/h")
    base = u[:-2] if has_h else u
    if base not in _TO_KPC:
        raise UnitError(f"Unidad de longitud no reconocida: {unit!r}. Válidas: pc, kpc, Mpc, Gpc (con o sin '/h').")
    return _TO_KPC[base], has_h


def convert_length(values, from_units: str, to_units: str, h: Optional[float] = None):
    """Convierte ``values`` (escalar o arreglo) entre unidades de longitud."""
    f_from, h_from = parse_unit(from_units)
    f_to, h_to = parse_unit(to_units)
    factor = f_from / f_to
    if h_from != h_to:
        if h is None or not np.isfinite(h) or h <= 0:
            raise UnitError(f"Convertir {from_units} -> {to_units} requiere un valor válido de h.")
        factor *= h ** (int(h_to) - int(h_from))
    return np.asarray(values, dtype=float) * factor


# ====================================================================================================
# MÓDULO: logging_utils
# ====================================================================================================
# Registro de ejecución, versiones y manifiesto reproducible.
#
# Todo lo que ocurre en una ejecución (etapas, comandos externos, datos
# descartados, advertencias) queda en ``<run_dir>/logs/run.log`` y el estado de
# cada etapa en ``<run_dir>/run_manifest.json``.


LOGGER_NAME = "mdpl2f3d"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")


def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configura el logger raíz del paquete (consola + archivo si se da ``log_dir``)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def package_versions() -> Dict[str, str]:
    """Versiones de Python y de las bibliotecas científicas utilizadas."""
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "scipy", "pandas", "matplotlib", "plotly", "tomli"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            out[mod] = "not installed"
    return out


@dataclass
class StageRecord:
    name: str
    status: str = "pending"          # pending | running | ok | failed | skipped
    started: Optional[str] = None
    finished: Optional[str] = None
    seconds: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class RunManifest:
    """Estado de las etapas de una ejecución, serializable a JSON."""

    def __init__(self, run_dir: Path, config_dict: Dict[str, Any], config_hash: str):
        self.run_dir = Path(run_dir)
        self.created = datetime.now(timezone.utc).isoformat()
        self.config = config_dict
        self.config_hash = config_hash
        self.versions = package_versions()
        self.stages: List[StageRecord] = []
        self.inputs: Dict[str, Any] = {}
        self.warnings: List[str] = []

    def add_input(self, label: str, path: Path, compute_hash: bool = True) -> None:
        p = Path(path)
        info: Dict[str, Any] = {"path": str(p), "exists": p.exists()}
        if p.exists() and p.is_file():
            info["bytes"] = p.stat().st_size
            if compute_hash:
                info["sha256"] = sha256_file(p)
        self.inputs[label] = info

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        get_logger().warning(msg)

    def stage(self, name: str) -> "_StageContext":
        rec = StageRecord(name=name)
        self.stages.append(rec)
        return _StageContext(self, rec)

    def save(self) -> Path:
        path = self.run_dir / "run_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "created_utc": self.created,
            "config_sha256": self.config_hash,
            "versions": self.versions,
            "inputs": self.inputs,
            "warnings": self.warnings,
            "stages": [s.__dict__ for s in self.stages],
            "config": self.config,
        }
        path.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        return path

    @property
    def all_ok(self) -> bool:
        return all(s.status in ("ok", "skipped", "running") for s in self.stages)


class _StageContext:
    """Contexto que mide tiempo y registra éxito o fallo de una etapa.

    Un fallo se registra y **se vuelve a lanzar**: el pipeline no continúa con
    datos que no se produjeron.
    """

    def __init__(self, manifest: RunManifest, rec: StageRecord):
        self.manifest, self.rec = manifest, rec

    def __enter__(self) -> StageRecord:
        self.rec.status = "running"
        self.rec.started = datetime.now(timezone.utc).isoformat()
        self._t0 = time.time()
        get_logger().info("== etapa: %s ==", self.rec.name)
        return self.rec

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.rec.finished = datetime.now(timezone.utc).isoformat()
        self.rec.seconds = round(time.time() - self._t0, 3)
        if exc_type is None:
            if self.rec.status == "running":
                self.rec.status = "ok"
        elif exc_type.__name__ == "_SkipStage":
            self.manifest.save()
            return True                      # etapa omitida a propósito: no es un fallo
        else:
            self.rec.status = "failed"
            self.rec.message = f"{exc_type.__name__}: {exc}"
            get_logger().error("etapa %s falló: %s", self.rec.name, self.rec.message)
        self.manifest.save()
        return False


def jsonable(obj: Any) -> Any:
    """Convierte tipos numpy/Path a tipos JSON nativos."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    return obj


def dump_json(path: Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


# ====================================================================================================
# MÓDULO: skeleton_parser
# ====================================================================================================
# Lectura de esqueletos ``NDskl_ascii`` de DisPerSE y modelo de datos de la red.
#
# Formato (``manual/ASCII/ASCIIformat.txt`` y escritor en C de DisPerSE 0.9.25):
#
#     ANDSKEL
#     <ndims>
#     #<comentario>
#     BBOX [x0,y0,z0] [dx,dy,dz]                  (opcional)
#     [CRITICAL POINTS]
#     <ncrit>
#     <type> <x> <y> <z> <value> <pair_id> <boundary>
#      <nfil>
#      <dest_id> <fil_id>                         (nfil líneas)
#     [FILAMENTS]
#     <nfil>
#     <cp1> <cp2> <nsamp>
#      x y z x y z ...                            (nsamp puntos; máx. 256 por línea)
#     [CRITICAL POINTS DATA]  /  [FILAMENTS DATA]   (opcionales)
#
# **Qué es un "filamento" aquí.** En el formato de DisPerSE cada entrada de
# ``[FILAMENTS]`` es un **arco** entre dos puntos críticos. Con ``mse -upSkl``  todos los arcos unen un punto silla de índice 2 con un
# máximo; un puente físico *máximo–silla–máximo* está formado por **dos** arcos
# que comparten la silla. Este módulo conserva el arco como unidad 
# y ofrece :func:`assemble_bridges` para la unidad física completa.


_BBOX_GROUP = re.compile(r"\[([^\]]*)\]")
_SECTIONS = ("[CRITICAL POINTS]", "[FILAMENTS]", "[CRITICAL POINTS DATA]", "[FILAMENTS DATA]")


class SkeletonFormatError(ValueError):
    """El archivo no cumple el formato NDskl_ascii esperado."""


# ----------------------------------------------------------------- modelo
@dataclass
class Skeleton:
    """Esqueleto 3D. Los puntos de todos los arcos se guardan concatenados (CSR):
    el arco ``i`` ocupa ``points[offsets[i]:offsets[i+1]]``."""

    ndims: int
    cp: pd.DataFrame
    links: pd.DataFrame
    filaments: pd.DataFrame                  # cp1, cp2, nsamp (+ columnas extra opcionales)
    points: np.ndarray
    offsets: np.ndarray
    bbox_min: Optional[np.ndarray] = None
    bbox_size: Optional[np.ndarray] = None
    comment: str = ""
    fil_data: Dict[str, np.ndarray] = field(default_factory=dict)
    provenance: Dict[str, object] = field(default_factory=dict)
    # Región de visualización (lo, hi) cuando el BBOX del archivo NO es la región analizada.
    view_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def __post_init__(self) -> None:
        if self.ndims != 3:
            raise SkeletonFormatError(
                f"Este pipeline es exclusivamente 3D; el esqueleto tiene ndims={self.ndims}."
            )
        self.points = np.asarray(self.points, dtype=float)
        self.offsets = np.asarray(self.offsets, dtype=np.int64)

    # -- información básica
    @property
    def n_arcs(self) -> int:
        return len(self.offsets) - 1

    n_filaments = n_arcs  # alias: "filamento" = arco-

    @property
    def n_critical_points(self) -> int:
        return len(self.cp)

    def __len__(self) -> int:
        return self.n_arcs

    def __repr__(self) -> str:
        return (f"Skeleton(arcs={self.n_arcs}, critical_points={self.n_critical_points}, "
                f"points={len(self.points)})")

    def polyline(self, i: int) -> np.ndarray:
        return self.points[self.offsets[i]: self.offsets[i + 1]]

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self.n_arcs):
            yield self.polyline(i)

    def cp_positions(self) -> np.ndarray:
        return self.cp[["x", "y", "z"]].to_numpy(dtype=float)

    @property
    def bbox(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.bbox_min is None or self.bbox_size is None:
            return None
        return self.bbox_min, self.bbox_min + self.bbox_size

    # -- selección
    def select(self, ids) -> "Skeleton":
        """Sub-esqueleto con los arcos indicados (máscara booleana o índices).
        Los arcos se renumeran; ``filaments['orig_id']`` conserva el id original."""
        ids = np.asarray(ids)
        if ids.dtype == bool:
            if ids.shape != (self.n_arcs,):
                raise ValueError("La máscara debe tener un valor por arco.")
            ids = np.flatnonzero(ids)
        ids = ids.astype(np.int64)
        counts = np.diff(self.offsets)[ids]
        new_off = np.concatenate([[0], np.cumsum(counts)])
        pts = np.concatenate([self.polyline(i) for i in ids]) if len(ids) else np.empty((0, 3))
        remap = -np.ones(self.n_arcs, dtype=np.int64)
        remap[ids] = np.arange(len(ids))
        links = self.links[self.links["fil"].isin(ids)].copy()
        links["fil"] = remap[links["fil"].to_numpy()]
        fil = self.filaments.iloc[ids].reset_index(drop=True).copy()
        if "orig_id" not in fil.columns:
            fil["orig_id"] = ids
        else:
            fil["orig_id"] = self.filaments["orig_id"].to_numpy()[ids]
        fd = {}
        if self.fil_data:
            sel = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in ids]) if len(ids) else np.array([], int)
            fd = {k: v[sel] for k, v in self.fil_data.items()}
        cp = self.cp.copy()
        sel_deg = np.bincount(np.r_[fil["cp1"].to_numpy(), fil["cp2"].to_numpy()], minlength=len(cp)) if len(fil) else np.zeros(len(cp), int)
        cp["nfil"] = sel_deg
        return Skeleton(3, cp, links.reset_index(drop=True), fil, pts, new_off,
                        self.bbox_min, self.bbox_size, self.comment, fd, dict(self.provenance), self.view_bounds)

    def scaled(self, factor: float, label: Optional[str] = None) -> "Skeleton":
        """Copia con todas las coordenadas (puntos, puntos críticos, BBOX) multiplicadas por ``factor``
        (conversión de unidades de longitud). No altera la topología; deja constancia en la procedencia."""
        f = float(factor)
        cp = self.cp.copy()
        for c in ("x", "y", "z"):
            cp[c] = cp[c].to_numpy(dtype=float) * f
        prov = dict(self.provenance)
        prov["scaled_by"] = f
        if label:
            prov["units"] = label
        vb = None if self.view_bounds is None else (self.view_bounds[0] * f, self.view_bounds[1] * f)
        return Skeleton(3, cp, self.links.copy(), self.filaments.copy(), self.points * f, self.offsets.copy(),
                        None if self.bbox_min is None else self.bbox_min * f,
                        None if self.bbox_size is None else self.bbox_size * f,
                        self.comment, {k: v.copy() for k, v in self.fil_data.items()}, prov, vb)

    def spacing_median(self) -> float:
        """Mediana de la distancia entre puntos de muestreo consecutivos (misma unidad que las coordenadas)."""
        if len(self.points) < 2:
            return float("nan")
        fid = np.repeat(np.arange(self.n_arcs), np.diff(self.offsets))
        ok = fid[:-1] == fid[1:]
        d = np.linalg.norm(np.diff(self.points, axis=0), axis=1)[ok]
        return float(np.median(d)) if len(d) else float("nan")


# ----------------------------------------------------------------- lectura
class _Cursor:
    def __init__(self, lines: List[str]):
        self.lines, self.i = lines, 0

    def _skip(self) -> None:
        while self.i < len(self.lines) and not self.lines[self.i].strip():
            self.i += 1

    def next(self) -> str:
        self._skip()
        if self.i >= len(self.lines):
            raise SkeletonFormatError("Fin de archivo inesperado: archivo truncado.")
        line = self.lines[self.i]
        self.i += 1
        return line

    def peek(self) -> Optional[str]:
        self._skip()
        return self.lines[self.i].strip() if self.i < len(self.lines) else None


def _floats(text: str) -> np.ndarray:
    try:
        return np.array(text.replace(",", " ").split(), dtype=float)
    except ValueError as e:
        raise SkeletonFormatError(f"Valor no numérico en línea: {text[:80]!r}") from e


def parse_ndskl_ascii(path: Union[str, Path], load_cp_data: bool = True,
                      load_fil_data: bool = True) -> Skeleton:
    """Lee y valida el formato de un ``NDskl_ascii`` 3D.

    Lanza :class:`SkeletonFormatError` ante cabecera inválida, dimensionalidad
    distinta de 3, archivo truncado o conteos inconsistentes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cur = _Cursor(path.read_text().splitlines())

    if cur.peek() is None or cur.next().strip() != "ANDSKEL":
        raise SkeletonFormatError(f"{path.name}: falta la cabecera 'ANDSKEL' (archivo vacío o no es un NDskl_ascii).")
    try:
        ndims = int(cur.next().split()[0])
    except (ValueError, IndexError) as e:
        raise SkeletonFormatError(f"{path.name}: número de dimensiones ilegible.") from e
    if ndims != 3:
        raise SkeletonFormatError(f"{path.name}: ndims={ndims}. Este pipeline es exclusivamente 3D.")

    comment, bbox_min, bbox_size = "", None, None
    line = cur.next().strip()
    while line != "[CRITICAL POINTS]":
        if line.startswith("#"):
            comment = line[1:].strip()
        elif line.startswith("BBOX"):
            g = _BBOX_GROUP.findall(line)
            if len(g) != 2:
                raise SkeletonFormatError(f"Línea BBOX no reconocida: {line!r}")
            bbox_min, bbox_size = _floats(g[0]), _floats(g[1])
            if bbox_min.size != 3 or bbox_size.size != 3:
                raise SkeletonFormatError("BBOX debe tener 3 componentes.")
        else:
            raise SkeletonFormatError(f"Línea inesperada en la cabecera: {line!r}")
        line = cur.next().strip()

    ncrit = int(cur.next().split()[0])
    cp_type = np.empty(ncrit, dtype=np.int64)
    cp_pos = np.empty((ncrit, 3))
    cp_val = np.full(ncrit, np.nan)
    cp_pair = np.arange(ncrit, dtype=np.int64)
    cp_bound = np.zeros(ncrit, dtype=np.int64)
    cp_nfil = np.zeros(ncrit, dtype=np.int64)
    links = []
    for c in range(ncrit):
        tok = cur.next().split()
        if len(tok) < 4:
            raise SkeletonFormatError(f"Punto crítico {c}: línea demasiado corta.")
        cp_type[c] = int(float(tok[0]))
        cp_pos[c] = [float(v) for v in tok[1:4]]
        rest = tok[4:]
        if len(rest) > 0: cp_val[c] = float(rest[0])
        if len(rest) > 1: cp_pair[c] = int(float(rest[1]))
        if len(rest) > 2: cp_bound[c] = int(float(rest[2]))
        nf = int(cur.next().split()[0])
        cp_nfil[c] = nf
        for _ in range(nf):
            d, f = cur.next().split()[:2]
            links.append((c, int(d), int(f)))
    cp = pd.DataFrame({"cp_id": np.arange(ncrit), "type": cp_type, "x": cp_pos[:, 0], "y": cp_pos[:, 1],
                       "z": cp_pos[:, 2], "value": cp_val, "pair_id": cp_pair,
                       "boundary": cp_bound, "nfil": cp_nfil})
    links_df = pd.DataFrame(links, columns=["cp", "dest", "fil"], dtype=np.int64)

    if cur.next().strip() != "[FILAMENTS]":
        raise SkeletonFormatError("Se esperaba la sección [FILAMENTS].")
    nfil = int(cur.next().split()[0])
    cp1 = np.empty(nfil, dtype=np.int64); cp2 = np.empty(nfil, dtype=np.int64)
    nsamp = np.empty(nfil, dtype=np.int64)
    chunks = []
    for k in range(nfil):
        a, b, ns = (int(v) for v in cur.next().split()[:3])
        cp1[k], cp2[k], nsamp[k] = a, b, ns
        need, got, buf = ns * 3, 0, []
        while got < need:
            arr = _floats(cur.next())
            buf.append(arr)
            got += arr.size
        if got != need:
            raise SkeletonFormatError(f"Arco {k}: se esperaban {need} valores y hay {got}.")
        chunks.append(np.concatenate(buf).reshape(ns, 3) if buf else np.empty((0, 3)))
    points = np.concatenate(chunks) if chunks else np.empty((0, 3))
    offsets = np.concatenate([[0], np.cumsum(nsamp)])
    fil_df = pd.DataFrame({"cp1": cp1, "cp2": cp2, "nsamp": nsamp})
    if len(links_df) and (links_df["fil"].max() >= nfil or max(cp1.max(initial=0), cp2.max(initial=0)) >= ncrit):
        raise SkeletonFormatError("Índices de arcos o de puntos críticos fuera de rango.")

    fil_data: Dict[str, np.ndarray] = {}
    while cur.peek() is not None:
        header = cur.next().strip()
        if header not in _SECTIONS:
            raise SkeletonFormatError(f"Sección desconocida: {header!r}")
        nfields = int(cur.next().split()[0])
        names = [cur.next().strip() for _ in range(nfields)]
        rows = []
        while cur.peek() is not None and not cur.peek().startswith("["):
            rows.append(cur.next())
        wanted = (header == "[CRITICAL POINTS DATA]" and load_cp_data) or (
            header == "[FILAMENTS DATA]" and load_fil_data)
        if not wanted or nfields == 0:
            continue
        arr = np.array(" ".join(rows).split(), dtype=float).reshape(-1, nfields)
        if header == "[CRITICAL POINTS DATA]":
            if len(arr) != ncrit:
                raise SkeletonFormatError("[CRITICAL POINTS DATA]: filas != ncrit.")
            for j, name in enumerate(names):
                cp[name if name not in cp.columns else name + "_data"] = arr[:, j]
        else:
            if len(arr) != len(points):
                raise SkeletonFormatError("[FILAMENTS DATA]: filas != número de puntos.")
            for j, name in enumerate(names):
                fil_data[name] = arr[:, j]

    prov = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    return Skeleton(3, cp, links_df, fil_df, points, offsets, bbox_min, bbox_size, comment, fil_data, prov)


def write_ndskl_ascii(skel: Skeleton, path: Union[str, Path]) -> None:
    """Escribe un Skeleton como NDskl_ascii (secciones núcleo; solo para pruebas
    y datos sintéticos; **no** sustituye a ``skelconv``)."""
    out = ["ANDSKEL", "3", "#" + (skel.comment or "mdpl2_filaments_3d")]
    if skel.bbox_min is not None and skel.bbox_size is not None:
        f = lambda v: ",".join(f"{x:g}" for x in v)
        out.append(f"BBOX [{f(skel.bbox_min)}] [{f(skel.bbox_size)}]")
    out += ["[CRITICAL POINTS]", str(skel.n_critical_points)]
    by_cp = {c: g for c, g in skel.links.groupby("cp")}
    pos = skel.cp_positions()
    for c, row in enumerate(skel.cp.itertuples(index=False)):
        out.append(f"{int(row.type)} " + " ".join(f"{v:.9g}" for v in pos[c])
                   + f" {row.value:.9g} {int(row.pair_id)} {int(row.boundary)}")
        g = by_cp.get(c)
        out.append(f" {0 if g is None else len(g)}")
        if g is not None:
            out.extend(f" {int(a)} {int(b)}" for a, b in zip(g["dest"], g["fil"]))
    out += ["[FILAMENTS]", str(skel.n_arcs)]
    for i in range(skel.n_arcs):
        p = skel.polyline(i)
        out.append(f"{int(skel.filaments.cp1.iloc[i])} {int(skel.filaments.cp2.iloc[i])} {len(p)}")
        for s in range(0, len(p), 256):
            out.append(" " + " ".join(f"{v:.9g}" for v in p[s:s + 256].ravel()))
    Path(path).write_text("\n".join(out) + "\n")


# ----------------------------------------------------------------- topología
def cp_degree(skel: Skeleton) -> np.ndarray:
    """Número de arcos incidentes en cada punto crítico (un bucle cuenta 2)."""
    return np.bincount(np.r_[skel.filaments["cp1"].to_numpy(), skel.filaments["cp2"].to_numpy()],
                       minlength=skel.n_critical_points)


def connected_components(skel: Skeleton) -> np.ndarray:
    """Componente conexa de cada arco (arcos unidos, directa o indirectamente,
    por puntos críticos compartidos). Los nodos compartidos NO duplican arcos."""
    if skel.n_arcs == 0:
        return np.empty(0, dtype=np.int64)
    n = skel.n_critical_points
    a, b = skel.filaments["cp1"].to_numpy(), skel.filaments["cp2"].to_numpy()
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    _, lab = _cc(g, directed=False)
    return lab[a]


def arc_roles(skel: Skeleton) -> pd.DataFrame:
    """Para cada arco: ``max_cp`` y ``saddle_cp`` si sus extremos son (silla de
    índice ndims−1, máximo); ``-1`` en otro caso (p. ej. arcos de otro tipo)."""
    t = skel.cp["type"].to_numpy()
    c1, c2 = skel.filaments["cp1"].to_numpy(), skel.filaments["cp2"].to_numpy()
    t1, t2 = t[c1], t[c2]
    mx, sd = skel.ndims, skel.ndims - 1
    max_cp = np.where((t1 == mx) & (t2 == sd), c1, np.where((t2 == mx) & (t1 == sd), c2, -1))
    sad_cp = np.where((t1 == mx) & (t2 == sd), c2, np.where((t2 == mx) & (t1 == sd), c1, -1))
    return pd.DataFrame({"max_cp": max_cp, "saddle_cp": sad_cp})


def duplicate_arcs(skel: Skeleton, decimals: int = 6) -> np.ndarray:
    """Máscara de arcos duplicados (mismo par de extremos y misma geometría,
    con cualquier sentido). El primero de cada grupo NO se marca."""
    seen, dup = {}, np.zeros(skel.n_arcs, dtype=bool)
    c1, c2 = skel.filaments["cp1"].to_numpy(), skel.filaments["cp2"].to_numpy()
    for i in range(skel.n_arcs):
        p = np.round(skel.polyline(i), decimals)
        fwd, bwd = p.tobytes(), p[::-1].tobytes()
        key = (min(c1[i], c2[i]), max(c1[i], c2[i]), min(fwd, bwd))
        if key in seen:
            dup[i] = True
        else:
            seen[key] = i
    return dup


def assemble_bridges(skel: Skeleton, include_dangling: bool = True) -> Tuple[Skeleton, pd.DataFrame]:
    """Une los arcos que comparten una silla de grado 2 en **puentes**
    máximo–silla–máximo (la unidad física de filamento entre dos nodos).

    Cada arco se usa exactamente una vez (no hay doble conteo). Reglas:

    * silla con exactamente 2 arcos → un puente de dos arcos (``complete=True``);
    * silla con 1 arco (típicamente truncada por el borde) → un "medio puente"
      (``complete=False``) si ``include_dangling``;
    * silla con ≥3 arcos o arcos sin el patrón (silla, máximo) → no se ensamblan:
      cada arco queda como entrada propia con ``complete=False`` y ``kind='unassembled'``.

    Devuelve ``(Skeleton_de_puentes, tabla)``; la tabla enlaza cada puente con
    sus arcos de origen. Los ``cp1/cp2`` del Skeleton devuelto son los máximos
    extremos (o la silla en medios puentes).
    """
    roles = arc_roles(skel)
    sad = roles["saddle_cp"].to_numpy()
    by_saddle: Dict[int, List[int]] = {}
    for i, s in enumerate(sad):
        if s >= 0:
            by_saddle.setdefault(int(s), []).append(i)

    polys: List[np.ndarray] = []
    rows = []
    used = np.zeros(skel.n_arcs, dtype=bool)

    def oriented_from_max(i: int) -> np.ndarray:
        """Puntos del arco i ordenados máximo → silla."""
        p = skel.polyline(i)
        return p if skel.filaments["cp1"].iloc[i] == roles["max_cp"].iloc[i] else p[::-1]

    for s, arcs in sorted(by_saddle.items()):
        if len(arcs) == 2:
            a, b = arcs
            pa, pb = oriented_from_max(a), oriented_from_max(b)
            poly = np.vstack([pa, pb[::-1][1:]])           # max_a → silla → max_b (sin repetir la silla)
            polys.append(poly)
            rows.append(dict(saddle_cp=s, max_cp_1=int(roles["max_cp"].iloc[a]), max_cp_2=int(roles["max_cp"].iloc[b]),
                             arc_1=a, arc_2=b, complete=True, kind="bridge"))
            used[[a, b]] = True
        elif len(arcs) == 1 and include_dangling:
            a = arcs[0]
            polys.append(oriented_from_max(a))
            rows.append(dict(saddle_cp=s, max_cp_1=int(roles["max_cp"].iloc[a]), max_cp_2=-1,
                             arc_1=a, arc_2=-1, complete=False, kind="dangling"))
            used[a] = True
    for i in np.flatnonzero(~used):                        # arcos que no encajan en el patrón
        polys.append(skel.polyline(i).copy())
        rows.append(dict(saddle_cp=int(sad[i]), max_cp_1=int(roles["max_cp"].iloc[i]), max_cp_2=-1,
                         arc_1=int(i), arc_2=-1, complete=False, kind="unassembled"))

    table = pd.DataFrame(rows)
    table.insert(0, "bridge_id", np.arange(len(table)))
    counts = np.array([len(p) for p in polys], dtype=np.int64)
    pts = np.concatenate(polys) if polys else np.empty((0, 3))
    fil = pd.DataFrame({"cp1": np.where(table["max_cp_1"] >= 0, table["max_cp_1"], table["saddle_cp"]),
                        "cp2": np.where(table["max_cp_2"] >= 0, table["max_cp_2"], table["saddle_cp"]),
                        "nsamp": counts})
    br = Skeleton(3, skel.cp, pd.DataFrame(columns=["cp", "dest", "fil"], dtype=np.int64), fil, pts,
                  np.concatenate([[0], np.cumsum(counts)]), skel.bbox_min, skel.bbox_size,
                  skel.comment + " [assembled bridges]", {}, dict(skel.provenance), skel.view_bounds)
    get_logger("skeleton").info("puentes: %d completos, %d colgantes, %d sin ensamblar (de %d arcos)",
                                int((table.kind == "bridge").sum()), int((table.kind == "dangling").sum()),
                                int((table.kind == "unassembled").sum()), skel.n_arcs)
    return br, table


# ------------------------------------------------- geometría compartida entre arcos
def segment_table(skel: Skeleton, decimals: int = 6) -> Dict[str, np.ndarray]:
    """Segmentos (pares de puntos consecutivos dentro de un arco) y su **multiplicidad**:
    cuántos arcos recorren exactamente el mismo segmento.

    En la salida real de ``mse -upSkl`` muchos arcos comparten tramos (p. ej. varios
    arcos que salen de un mismo máximo siguen la misma cresta). Sumar las longitudes de
    los arcos cuenta esos tramos varias veces. Un segmento con multiplicidad ``m``
    aporta ``length / m`` a la *longitud geométrica única* y a la parte que le
    corresponde a cada arco (``length_unique_share``).

    Devuelve: ``arc`` (arco de cada segmento), ``length``, ``mult``, ``seg_off``
    (límites de los segmentos de cada arco, ``n_arcs+1``).
    """
    n = skel.n_arcs
    counts = np.diff(skel.offsets)
    nseg = np.maximum(counts - 1, 0)
    seg_off = np.concatenate([[0], np.cumsum(nseg)]).astype(np.int64)
    if len(skel.points) < 2 or nseg.sum() == 0:
        return dict(arc=np.empty(0, np.int64), length=np.empty(0), mult=np.empty(0, np.int64), seg_off=seg_off)
    P = skel.points
    fid = np.repeat(np.arange(n), counts)
    ok = fid[:-1] == fid[1:]
    A, B = np.round(P[:-1][ok], decimals), np.round(P[1:][ok], decimals)
    arc = fid[:-1][ok]
    length = np.linalg.norm(P[1:][ok] - P[:-1][ok], axis=1)
    swap = (A[:, 0] > B[:, 0]) | ((A[:, 0] == B[:, 0]) & (A[:, 1] > B[:, 1])) | \
           ((A[:, 0] == B[:, 0]) & (A[:, 1] == B[:, 1]) & (A[:, 2] > B[:, 2]))
    lo, hi = np.where(swap[:, None], B, A), np.where(swap[:, None], A, B)
    key = np.hstack([lo, hi])
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    # un arco que repite el mismo segmento (lazo degenerado) cuenta una sola vez por arco
    mult = cnt[inv.ravel()]
    return dict(arc=arc, length=length, mult=mult.astype(np.int64), seg_off=seg_off)


def unique_geometric_length(skel: Skeleton, decimals: int = 6) -> float:
    """Longitud total de la red contando cada segmento compartido UNA vez."""
    t = segment_table(skel, decimals)
    return float(np.sum(t["length"] / t["mult"])) if len(t["length"]) else 0.0


def arc_shared_fractions(skel: Skeleton, decimals: int = 6) -> pd.DataFrame:
    """Por arco: ``shared_length_fraction`` (fracción de su longitud que recorren
    también otros arcos) y ``length_unique_share`` (su parte de la longitud única)."""
    t = segment_table(skel, decimals)
    n = skel.n_arcs
    if len(t["length"]) == 0:
        return pd.DataFrame({"shared_length_fraction": np.zeros(n), "length_unique_share": np.zeros(n)})
    tot = np.bincount(t["arc"], weights=t["length"], minlength=n)
    shared = np.bincount(t["arc"], weights=t["length"] * (t["mult"] > 1), minlength=n)
    share = np.bincount(t["arc"], weights=t["length"] / t["mult"], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(tot > 0, shared / tot, 0.0)
    return pd.DataFrame({"shared_length_fraction": frac, "length_unique_share": share})


def empirical_extent(skeletons) -> Dict[str, object]:
    """Extensión espacial empírica (mín/máx de puntos y puntos críticos) de uno o varios esqueletos,
    comparada con el BBOX declarado. Es una COTA INFERIOR de la región analizada: el esqueleto sólo
    cubre donde hay filamentos. Si el BBOX es la caja completa y el esqueleto ocupa una fracción
    pequeña, el BBOX NO es la región analizada (``bbox_is_region = False``)."""
    sks = list(skeletons)
    pts = np.concatenate([np.vstack([sk.points, sk.cp_positions()]) for sk in sks])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = hi - lo
    out: Dict[str, object] = {"lo": lo.tolist(), "hi": hi.tolist(), "size": size.tolist(),
                              "volume": float(np.prod(size)),
                              "aspect_max_over_min": float(size.max() / size.min()) if size.min() > 0 else float("nan")}
    bb = sks[0].bbox_size
    if bb is not None:
        frac = size / bb
        out.update(bbox_size=bb.tolist(), fraction_of_bbox_per_axis=frac.tolist(),
                   bbox_is_region=bool(np.all(frac >= 0.5)))
        b0, b1 = sks[0].bbox
        tol = 1e-6 * float(np.max(bb))
        out["touches_bbox_edge"] = bool(np.any(lo <= b0 + tol) or np.any(hi >= b1 - tol))
    else:
        out.update(bbox_size=None, bbox_is_region=False, touches_bbox_edge=None)
    return out


# ====================================================================================================
# MÓDULO: geometry
# ====================================================================================================
# Geometría de los filamentos 3D.
#
# Para un arco con puntos de muestreo ``p_1 … p_N`` (unidades de la simulación):
#
# ``length``         L = Σ_{i=1}^{N-1} ‖p_{i+1} − p_i‖                       
# ``end_to_end``     D = ‖p_N − p_1‖                                         
# ``curvature_eff``  C_eff = L / D                                           
#
# * C_eff ≈ 1: trayectoria casi recta; C_eff > 1: curvada o tortuosa.
# * **C_eff no está definida si D = 0** (extremos coincidentes, p. ej. un lazo) ni
#   con datos insuficientes. Se devuelve NaN y ``curvature_status`` explica el
#   motivo. (El notebook original devolvía 0 en ese caso; aquí no se sustituye
#   un valor indefinido por 0.)
#
# Tensor (sec. 3.0.4). Principal: **matriz de covarianza** de los puntos respecto
# de su centroide con normalización 1/N (``covariance_mean``):
#
#     I_ij = (1/N) Σ_n (x_i^(n) − x̄_i)(x_j^(n) − x̄_j)
#
# con autovalores ``λ_minor ≤ λ_mid ≤ λ_major``. 
# ``L_inertia = √λ_major``, grosor ``√λ_minor`` (y ``√λ_mid``), orientación =
# autovector de ``λ_major``.
#
# Orientación: un eje no tiene sentido (``v`` y ``−v`` son lo mismo), así que los
# vectores se pliegan al hemisferio ``v_z ≥ 0``. Se reportan: ángulo con el eje Z
# θ_z ∈ [0°, 90°], elevación sobre el plano XY = 90° − θ_z  y azimut φ ∈ [0°, 180°).


TENSOR_CONVENTIONS = ("covariance_mean", "covariance_sum", "mechanical_sum")


# ------------------------------------------------------------- primitivas
def polyline_length(p: np.ndarray) -> float:
    """L = suma de distancias euclidianas entre puntos consecutivos."""
    p = np.asarray(p, dtype=float)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def end_to_end_distance(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    if len(p) < 2:
        return float("nan")
    return float(np.linalg.norm(p[-1] - p[0]))


def effective_curvature(L: float, D: float, rel_tol: float = 1e-9) -> Tuple[float, str]:
    """``(C_eff, estado)``. NaN (no 0) si D = 0 o los datos no son válidos."""
    if not (np.isfinite(L) and np.isfinite(D)):
        return float("nan"), "nonfinite"
    if D <= rel_tol * max(L, 0.0) or D == 0.0:
        return float("nan"), "coincident_endpoints"
    return float(L / D), "ok"


def fold_axis(v: np.ndarray) -> np.ndarray:
    """Plegado de un eje al hemisferio ``v_z ≥ 0`` (si ``v_z = 0``: ``v_y ≥ 0``,
    y luego ``v_x ≥ 0``). Elimina la ambigüedad entre sentidos opuestos."""
    v = np.asarray(v, dtype=float)
    scale = max(1.0, float(np.abs(v).max()))
    for k in (2, 1, 0):
        if abs(v[k]) > 1e-12 * scale:
            return v if v[k] > 0 else -v
    return v


def axis_angles(v: np.ndarray) -> Tuple[float, float, float]:
    """``(θ_z, elevación_XY, φ)`` en grados para un eje (sin sentido)."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        return (np.nan, np.nan, np.nan)
    f = fold_axis(v / n)
    theta_z = float(np.degrees(np.arccos(np.clip(f[2], -1.0, 1.0))))
    phi = float(np.degrees(np.arctan2(f[1], f[0])) % 180.0)
    return theta_z, 90.0 - theta_z, phi


def angle_between_axes(a: np.ndarray, b: np.ndarray) -> float:
    """Ángulo (0°–90°) entre dos ejes sin sentido."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0 or not np.isfinite(na * nb):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(abs(a @ b) / (na * nb), 0.0, 1.0))))


def resample_polyline(p: np.ndarray, step: float) -> np.ndarray:
    """Remuestrea a puntos equiespaciados en longitud de arco (conserva extremos)."""
    p = np.asarray(p, float)
    if len(p) < 2:
        return p.copy()
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return p[:1].copy()
    n = max(int(np.ceil(s[-1] / step)), 1)
    t = np.linspace(0.0, s[-1], n + 1)
    return np.column_stack([np.interp(t, s, p[:, k]) for k in range(3)])


def tensor_eigh(points: np.ndarray, convention: str = "covariance_mean") -> Tuple[np.ndarray, np.ndarray]:
    """Autovalores (ascendentes) y autovectores del tensor según la convención.

    * ``covariance_mean``: ``S/N`` con ``S = Σ c cᵀ``.
    * ``covariance_sum``: ``S``.
    * ``mechanical_sum``: tensor de inercia mecánico ``tr(S)·1 − S`` (masas
      unitarias). Sus autovalores están invertidos respecto de ``S``: el de mayor
      valor corresponde a la dirección *transversal* (ver auditoría).
    """
    if convention not in TENSOR_CONVENTIONS:
        raise ValueError(f"convención de tensor desconocida: {convention}")
    c = points - points.mean(axis=0)
    S = c.T @ c
    if convention == "covariance_mean":
        T = S / len(points)
    elif convention == "covariance_sum":
        T = S
    else:
        T = np.trace(S) * np.eye(3) - S
    w, v = np.linalg.eigh(T)
    return np.clip(w, 0.0, None), v


# ---------------------------------------------------------- métricas por arco
_NAN_COLS = [
    "length", "end_to_end", "curvature_eff", "mean_spacing", "bend_per_length",
    "u_x", "u_y", "u_z", "angle_z_e2e_deg", "elevation_xy_e2e_deg", "azimuth_e2e_deg",
    "cm_x", "cm_y", "cm_z",
    "lambda_major", "lambda_mid", "lambda_minor", "length_inertia", "thickness_minor", "thickness_mid",
    "axis_x", "axis_y", "axis_z", "angle_z_axis_deg", "elevation_xy_axis_deg", "azimuth_axis_deg",
    "length_ratio_geom_tensor", "axis_vs_e2e_deg", "linearity", "planarity", "sphericity",
]


def filament_metrics(skel: Skeleton, convention: str = "covariance_mean",
                     inertia_resample_step: Optional[float] = None, min_points: int = 2,
                     coincident_rel_tol: float = 1e-9) -> pd.DataFrame:
    """Métricas de todos los arcos de ``skel``; una fila por arco (``fil_id`` = índice
    del arco en el archivo). Ver el docstring del módulo para las definiciones."""
    n = skel.n_arcs
    cols: Dict[str, np.ndarray] = {k: np.full(n, np.nan) for k in _NAN_COLS}
    status = np.array(["ok"] * n, dtype=object)
    shape_ok = convention.startswith("covariance")

    for i in range(n):
        p = skel.polyline(i)
        if len(p) < min_points or len(p) < 2:
            status[i] = "too_few_points"
            cols["length"][i] = 0.0 if len(p) < 2 else polyline_length(p)
            continue
        if not np.isfinite(p).all():
            status[i] = "nonfinite"
            continue
        dv = np.diff(p, axis=0)
        seg = np.linalg.norm(dv, axis=1)
        L = float(seg.sum())
        e = p[-1] - p[0]
        D = float(np.linalg.norm(e))
        C, st = effective_curvature(L, D, coincident_rel_tol)
        status[i] = st
        cols["length"][i], cols["end_to_end"][i], cols["curvature_eff"][i] = L, D, C
        cols["mean_spacing"][i] = L / (len(p) - 1)
        if len(p) >= 3 and L > 0:
            a, b, la, lb = dv[:-1], dv[1:], seg[:-1], seg[1:]
            ok = (la > 0) & (lb > 0)
            cosang = np.clip((a[ok] * b[ok]).sum(1) / (la[ok] * lb[ok]), -1.0, 1.0)
            cols["bend_per_length"][i] = float(np.arccos(cosang).sum() / L)
        else:
            cols["bend_per_length"][i] = 0.0
        if st == "ok":
            u = fold_axis(e / D)
            cols["u_x"][i], cols["u_y"][i], cols["u_z"][i] = u
            cols["angle_z_e2e_deg"][i], cols["elevation_xy_e2e_deg"][i], cols["azimuth_e2e_deg"][i] = axis_angles(e)

        q = p if inertia_resample_step is None else resample_polyline(p, inertia_resample_step)
        if len(q) >= 2:
            cm = q.mean(axis=0)
            w, V = tensor_eigh(q, convention)
            cols["cm_x"][i], cols["cm_y"][i], cols["cm_z"][i] = cm
            cols["lambda_minor"][i], cols["lambda_mid"][i], cols["lambda_major"][i] = w
            cols["length_inertia"][i] = np.sqrt(w[2])
            cols["thickness_minor"][i], cols["thickness_mid"][i] = np.sqrt(w[0]), np.sqrt(w[1])
            ax = fold_axis(V[:, 2])
            cols["axis_x"][i], cols["axis_y"][i], cols["axis_z"][i] = ax
            cols["angle_z_axis_deg"][i], cols["elevation_xy_axis_deg"][i], cols["azimuth_axis_deg"][i] = axis_angles(ax)
            if cols["length_inertia"][i] > 0:
                cols["length_ratio_geom_tensor"][i] = L / cols["length_inertia"][i]
            if D > 0:
                cols["axis_vs_e2e_deg"][i] = angle_between_axes(ax, e)
            if shape_ok and w[2] > 0:
                cols["linearity"][i] = (w[2] - w[1]) / w[2]
                cols["planarity"][i] = (w[1] - w[0]) / w[2]
                cols["sphericity"][i] = w[0] / w[2]

    out = pd.DataFrame(cols)
    out.insert(0, "curvature_status", status)
    out.insert(0, "n_points", np.diff(skel.offsets))
    out.insert(0, "cp2", skel.filaments["cp2"].to_numpy())
    out.insert(0, "cp1", skel.filaments["cp1"].to_numpy())
    out.insert(0, "fil_id", np.arange(n))
    return out


def tensor_convention_audit(skel: Skeleton, conventions: Iterable[str] = TENSOR_CONVENTIONS,
                            inertia_resample_step: Optional[float] = None) -> pd.DataFrame:
    """Compara las convenciones de tensor sobre un mismo esqueleto.

    Para cada una informa la mediana de ``L/L_inertia`` y de ``axis_vs_e2e_deg``
    (ángulo entre el eje principal y el vector extremo-a-extremo). Con la
    convención correcta el eje principal es *paralelo* al filamento (mediana
    ≪ 45°).(``L_geom/L_inercia`` → 1 a alta persistencia).
    """
    rows = []
    for c in conventions:
        m = filament_metrics(skel, convention=c, inertia_resample_step=inertia_resample_step)
        ok = m[(m["curvature_status"] == "ok") & (m["n_points"] >= 3)]
        rows.append({
            "convention": c, "n_arcs_used": int(len(ok)),
            "median_L_over_Linertia": float(ok["length_ratio_geom_tensor"].median()) if len(ok) else np.nan,
            "median_axis_vs_e2e_deg": float(ok["axis_vs_e2e_deg"].median()) if len(ok) else np.nan,
            "frac_axis_parallel_lt30deg": float((ok["axis_vs_e2e_deg"] < 30).mean()) if len(ok) else np.nan,
            "interpretation": ("eje principal paralelo al filamento" if len(ok) and ok["axis_vs_e2e_deg"].median() < 45
                               else "eje principal PERPENDICULAR al filamento (¡revisar!)"),
        })
    return pd.DataFrame(rows)


# ====================================================================================================
# MÓDULO: data_validation
# ====================================================================================================
# Validación del catálogo de halos y de los esqueletos leídos.


class ValidationError(ValueError):
    """Los datos no superan la validación."""


@dataclass
class ValidationReport:
    name: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ValidationError(f"[{self.name}] " + " | ".join(self.errors))

    def log(self) -> None:
        lg = get_logger("validation")
        for w in self.warnings:
            lg.warning("[%s] %s", self.name, w)
        for e in self.errors:
            lg.error("[%s] %s", self.name, e)


# ------------------------------------------------------------------ catálogo
def validate_catalog(df: pd.DataFrame, box_size: Optional[float] = None,
                     units: Optional[str] = None) -> ValidationReport:
    """Comprueba un catálogo con columnas ``x, y, z``.

    No modifica los datos. ``box_size`` (en las mismas unidades que las
    coordenadas) permite detectar unidades equivocadas.
    """
    r = ValidationReport("catalog")
    n = len(df)
    r.info["n_rows"] = n
    if n == 0:
        r.errors.append("el catálogo está vacío.")
        return r
    for c in ("x", "y", "z"):
        if c not in df.columns:
            r.errors.append(f"falta la columna {c}.")
    if r.errors:
        return r
    xyz = df[["x", "y", "z"]].to_numpy(dtype=float)
    finite = np.isfinite(xyz).all(axis=1)
    r.info["n_non_finite"] = int((~finite).sum())
    if (~finite).any():
        r.warnings.append(f"{int((~finite).sum())} filas con coordenadas no finitas (serán descartadas).")
    if finite.sum() < 5:
        r.errors.append("hay menos de 5 halos con coordenadas finitas: no es posible triangular.")
        return r
    g = xyz[finite]
    r.info["coord_min"] = g.min(axis=0).tolist()
    r.info["coord_max"] = g.max(axis=0).tolist()
    dup = int(len(g) - len(np.unique(g, axis=0)))
    r.info["n_duplicate_positions"] = dup
    if dup:
        r.warnings.append(f"{dup} halos comparten posición exacta con otro (pueden afectar a la triangulación).")
    if box_size is not None and box_size > 0:
        span = (g.max(axis=0) - g.min(axis=0)).max()
        r.info["max_span"] = float(span)
        if g.max() > box_size * 1.001 or g.min() < -box_size * 0.001:
            r.errors.append(f"hay coordenadas fuera de la caja [0, {box_size:g}] ({units or 'unidades del catálogo'}): "
                            "revise unidades y columnas.")
        elif span < box_size * 1e-4:
            r.warnings.append("el rango de coordenadas es <0.01% de la caja: ¿unidades equivocadas?")
    return r


def clean_catalog(df: pd.DataFrame) -> tuple:
    """Descarta filas con coordenadas no finitas. Devuelve ``(df_limpio, informe)``."""
    xyz = df[["x", "y", "z"]].to_numpy(dtype=float)
    ok = np.isfinite(xyz).all(axis=1)
    rep = {"n_in": int(len(df)), "n_out": int(ok.sum()), "discarded_non_finite": int((~ok).sum())}
    if (~ok).any():
        get_logger("validation").warning("descartadas %d filas con coordenadas no finitas", int((~ok).sum()))
    return df.loc[ok].reset_index(drop=True), rep


# ------------------------------------------------------------------ esqueleto
def validate_skeleton(skel, tol_factor: float = 1e-3) -> ValidationReport:
    """Valida la coherencia interna de un esqueleto leído."""
    pass  # (import interno eliminado en el archivo único)

    r = ValidationReport("skeleton")
    r.info.update(n_arcs=skel.n_arcs, n_critical_points=skel.n_critical_points, n_points=int(len(skel.points)))
    if skel.n_arcs == 0:
        r.warnings.append("el esqueleto no tiene arcos.")
        return r
    if not np.isfinite(skel.points).all():
        r.errors.append("hay coordenadas no finitas en los puntos de muestreo.")
    ns = np.diff(skel.offsets)
    if (ns != skel.filaments["nsamp"].to_numpy()).any():
        r.errors.append("nsamp no coincide con los offsets de los puntos.")
    if (ns < 2).any():
        r.warnings.append(f"{int((ns < 2).sum())} arcos con menos de 2 puntos (longitud no definida).")
    # extremos de cada arco vs posición de sus puntos críticos
    pos = skel.cp_positions()
    c1, c2 = skel.filaments["cp1"].to_numpy(), skel.filaments["cp2"].to_numpy()
    first = skel.points[skel.offsets[:-1]]
    last = skel.points[skel.offsets[1:] - 1]
    sp = skel.spacing_median()
    tol = (sp if np.isfinite(sp) and sp > 0 else 1.0) * tol_factor
    bad = (np.abs(first - pos[c1]).max(axis=1) > tol) | (np.abs(last - pos[c2]).max(axis=1) > tol)
    r.info["n_arcs_endpoint_mismatch"] = int(bad.sum())
    if bad.any():
        r.warnings.append(f"{int(bad.sum())} arcos cuyos extremos no coinciden con sus puntos críticos "
                          f"(tol={tol:.3g}).")
    # consistencia de enlaces y grados
    deg = cp_degree(skel)
    nfil_mismatch = int((deg != skel.cp["nfil"].to_numpy()).sum())
    r.info["n_cp_degree_mismatch"] = nfil_mismatch
    if nfil_mismatch:
        r.warnings.append(f"{nfil_mismatch} puntos críticos con grado distinto de 'nfil' declarado.")
    d = int(duplicate_arcs(skel).sum())
    r.info["n_duplicate_arcs"] = d
    if d:
        r.warnings.append(f"{d} arcos duplicados (se cuentan una sola vez en las estadísticas de red si se filtran).")
    # tipo de arcos
    roles = arc_roles(skel)
    frac = float((roles["max_cp"] >= 0).mean())
    r.info["fraction_arcs_saddle_max"] = frac
    if frac < 1.0:
        r.warnings.append(f"solo {frac:.1%} de los arcos son (silla, máximo): el archivo no parece provenir de '-upSkl'.")
    # puntos dentro de la caja
    if skel.bbox is not None:
        lo, hi = skel.bbox
        eps = 1e-6 * float(np.max(hi - lo))
        out = int(((skel.points < lo - eps) | (skel.points > hi + eps)).any(axis=1).sum())
        r.info["n_points_outside_bbox"] = out
        if out:
            r.warnings.append(f"{out} puntos de muestreo fuera del BBOX del esqueleto.")
    r.info["boundary_flag_counts"] = {int(k): int(v) for k, v in skel.cp["boundary"].value_counts().items()}
    return r


# ====================================================================================================
# MÓDULO: filament_statistics
# ====================================================================================================
# Tablas por filamento y estadísticas (individuales vs. globales de la red).
#
# * **Individuales**: distribución de una propiedad sobre los arcos (media, mediana,
#   desviación estándar, percentiles...).
# * **Globales de la red**: cantidades de la red completa (longitud total, número de
#   nodos, conectividad, componentes, densidad de longitud...).
#
# Tratamiento de la topología (documentado en ``docs/METHODOLOGY.md``):
#
# * Cada arco se cuenta **una vez**; los nodos compartidos no duplican longitud.
# * **Segmentos compartidos**: en la salida real de ``-upSkl`` varios arcos recorren
#   el mismo tramo. ``total_length``  los cuenta
#   varias veces; ``total_length_unique`` los cuenta una sola vez (ver
#   ``skeleton_parser.segment_table``).
# * Los puntos críticos compartidos se reflejan en el **grado** (número de arcos
#   incidentes): una silla de grado 2 une dos arcos (un puente); un máximo de
#   grado k es un nodo de conectividad k.
# * Los filamentos desconectados se identifican por ``component_id``.
# * Los arcos con extremos coincidentes o con < 2 puntos conservan su fila, pero
#   su curvatura es NaN y se excluyen (con conteo) de las estadísticas de C_eff.


DEFAULT_PERCENTILES = (5, 16, 25, 50, 75, 84, 95)

#: Propiedades individuales que se resumen por defecto.
SUMMARY_COLUMNS = ["length", "shared_length_fraction", "end_to_end", "curvature_eff", "n_points", "length_inertia",
                   "thickness_minor", "thickness_mid", "length_ratio_geom_tensor", "bend_per_length",
                   "angle_z_axis_deg", "elevation_xy_axis_deg", "angle_z_e2e_deg"]


def describe_series(x: Iterable[float], percentiles: Sequence[float] = DEFAULT_PERCENTILES) -> Dict[str, Any]:
    """Estadísticos robustos ante NaN y muestras pequeñas (std = NaN si n < 2)."""
    a = np.asarray(list(x) if not isinstance(x, (np.ndarray, pd.Series)) else x, dtype=float)
    n_total = int(a.size)
    a = a[np.isfinite(a)]
    out: Dict[str, Any] = {"n_total": n_total, "n": int(a.size)}
    if a.size == 0:
        out.update(mean=np.nan, std=np.nan, median=np.nan, min=np.nan, max=np.nan)
        out.update({f"p{p:g}": np.nan for p in percentiles})
        return out
    out.update(mean=float(a.mean()), std=float(a.std(ddof=1)) if a.size > 1 else np.nan,
               median=float(np.median(a)), min=float(a.min()), max=float(a.max()))
    for p, v in zip(percentiles, np.percentile(a, percentiles)):
        out[f"p{p:g}"] = float(v)
    return out


# ---------------------------------------------------------------- tablas
def build_filament_table(skel: Skeleton, sigma: float, sigma_origin: str = "thesis",
                         convention: str = "covariance_mean",
                         inertia_resample_step: Optional[float] = None,
                         coincident_rel_tol: float = 1e-9) -> pd.DataFrame:
    """Tabla por arco: métricas geométricas + identificadores + contexto topológico.

    El identificador estable dentro de la ejecución es ``uid = 's<sigma>_a<fil_id>'``;
    **no** es comparable entre umbrales (cada ejecución de DisPerSE renumera).
    """
    m = filament_metrics(skel, convention=convention, inertia_resample_step=inertia_resample_step,
                         coincident_rel_tol=coincident_rel_tol)
    roles = arc_roles(skel)
    deg = cp_degree(skel)
    cp = skel.cp
    tag = f"{sigma:.3g}"
    m.insert(0, "uid", [f"s{tag}_a{i:06d}" for i in m["fil_id"]])
    m.insert(1, "sigma", float(sigma))
    m.insert(2, "sigma_origin", sigma_origin)
    m["skeleton_sha256"] = str(skel.provenance.get("sha256", ""))[:12]
    m["cp1_type"] = cp["type"].to_numpy()[m["cp1"]]
    m["cp2_type"] = cp["type"].to_numpy()[m["cp2"]]
    m["max_cp"], m["saddle_cp"] = roles["max_cp"].to_numpy(), roles["saddle_cp"].to_numpy()
    m["max_degree"] = np.where(m["max_cp"] >= 0, deg[np.clip(m["max_cp"], 0, None)], -1)
    m["saddle_degree"] = np.where(m["saddle_cp"] >= 0, deg[np.clip(m["saddle_cp"], 0, None)], -1)
    m["component_id"] = connected_components(skel)
    b = cp["boundary"].to_numpy()
    m["touches_boundary"] = (b[m["cp1"]] > 0) | (b[m["cp2"]] > 0)
    m["is_duplicate"] = duplicate_arcs(skel)
    sh = arc_shared_fractions(skel)
    m["shared_length_fraction"] = sh["shared_length_fraction"].to_numpy()
    m["length_unique_share"] = sh["length_unique_share"].to_numpy()
    for col in ("persistence_nsigmas", "persistence_ratio"):         # persistencia intrínseca (si el archivo la trae)
        if col in cp.columns:
            sad = np.clip(m["saddle_cp"].to_numpy(), 0, None)
            mx = np.clip(m["max_cp"].to_numpy(), 0, None)
            m[f"saddle_{col}"] = np.where(m["saddle_cp"] >= 0, cp[col].to_numpy()[sad], np.nan)
            m[f"max_{col}"] = np.where(m["max_cp"] >= 0, cp[col].to_numpy()[mx], np.nan)
    if "field_value" in skel.fil_data and len(skel.points):
        fv = skel.fil_data["field_value"]
        m["mean_field_value"] = np.add.reduceat(fv, skel.offsets[:-1]) / np.maximum(np.diff(skel.offsets), 1)
    return m


def build_node_table(skel: Skeleton, sigma: float) -> pd.DataFrame:
    """Puntos críticos con su grado en el esqueleto (conectividad)."""
    t = skel.cp.copy()
    t["degree"] = cp_degree(skel)
    t.insert(0, "sigma", float(sigma))
    return t


def build_bridge_table(skel: Skeleton, sigma: float, convention: str = "covariance_mean",
                       inertia_resample_step: Optional[float] = None) -> pd.DataFrame:
    """Puentes máximo–silla–máximo (unidad física completa) con sus métricas.

    Vista ALTERNATIVA (que analiza arcos); se mantiene separada.
    """
    br, tab = assemble_bridges(skel)
    m = filament_metrics(br, convention=convention, inertia_resample_step=inertia_resample_step)
    t = tab.merge(m.drop(columns=["cp1", "cp2"]), left_on="bridge_id", right_on="fil_id").drop(columns=["fil_id"])
    t.insert(1, "sigma", float(sigma))
    t.insert(0, "uid", [f"s{sigma:.3g}_b{i:06d}" for i in t["bridge_id"]])
    return t


# ----------------------------------------------------------------- resúmenes
def network_summary(skel: Skeleton, table: pd.DataFrame, nodes: pd.DataFrame, sigma: float,
                    volume: Optional[float] = None,
                    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
                    columns: Sequence[str] = SUMMARY_COLUMNS) -> Dict[str, Any]:
    """Resumen de un umbral: ``network`` (globales) e ``individual`` (por propiedad).

    ``table`` es la salida de :func:`build_filament_table`. Los arcos duplicados
    se excluyen de las estadísticas de red para evitar doble conteo.
    """
    t = table[~table["is_duplicate"]] if "is_duplicate" in table else table
    L = t["length"].to_numpy(dtype=float)
    total = float(np.nansum(L))
    unique_total = float(unique_geometric_length(skel))
    maxima = nodes[nodes["type"] == skel.ndims]
    saddles = nodes[nodes["type"] == skel.ndims - 1]
    comp_len = t.groupby("component_id")["length"].sum() if len(t) else pd.Series(dtype=float)
    net: Dict[str, Any] = {
        "sigma": float(sigma),
        "n_arcs": int(len(t)),
        "n_duplicate_arcs_excluded": int(len(table) - len(t)),
        "n_critical_points": int(len(nodes)),
        "n_maxima_in_skeleton": int((maxima["degree"] > 0).sum()),
        "n_saddles_in_skeleton": int((saddles["degree"] > 0).sum()),
        "total_length": total,                       # Σ longitudes de arco 
        "total_length_unique": unique_total,         # longitud geométrica: cada segmento compartido UNA vez
        "overlap_fraction": float(1.0 - unique_total / total) if total > 0 else np.nan,
        "n_components": int(t["component_id"].nunique()) if len(t) else 0,
        "largest_component_length_fraction": float(comp_len.max() / total) if total > 0 else np.nan,
        "n_arcs_curvature_defined": int(t["curvature_eff"].notna().sum()),
        "n_arcs_curvature_undefined": int(t["curvature_eff"].isna().sum()),
        "n_arcs_touching_boundary": int(t["touches_boundary"].sum()) if "touches_boundary" in t else 0,
        "node_connectivity": describe_series(maxima.loc[maxima["degree"] > 0, "degree"], percentiles),
        "saddle_degree_counts": {int(k): int(v) for k, v in saddles["degree"].value_counts().sort_index().items()},
    }
    if volume and volume > 0:
        net["volume"] = float(volume)
        net["length_density"] = total / float(volume)
        net["length_density_unique"] = unique_total / float(volume)
    ind = {c: describe_series(t[c], percentiles) for c in columns if c in t.columns}
    return {"network": net, "individual": ind}


def flatten_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Aplana ``network_summary`` en una fila (para tablas de comparación entre umbrales)."""
    row: Dict[str, Any] = {}
    for k, v in summary["network"].items():
        if isinstance(v, dict) and "mean" in v:
            for kk, vv in v.items():
                row[f"{k}__{kk}"] = vv
        elif not isinstance(v, dict):
            row[k] = v
    for prop, d in summary["individual"].items():
        for kk, vv in d.items():
            row[f"{prop}__{kk}"] = vv
    return row


def histogram(values: Iterable[float], bins: int | Sequence[float] = 30, log: bool = False,
              density: bool = True) -> Dict[str, np.ndarray]:
    """Histograma con bordes explícitos (log-espaciados si ``log``)."""
    a = np.asarray(list(values) if not isinstance(values, (np.ndarray, pd.Series)) else values, dtype=float)
    a = a[np.isfinite(a)]
    if log:
        a = a[a > 0]
    if a.size == 0:
        return {"edges": np.array([]), "counts": np.array([])}
    if isinstance(bins, int):
        edges = np.geomspace(a.min(), a.max(), bins + 1) if log and a.min() < a.max() else np.linspace(a.min(), a.max() + (a.max() == a.min()), bins + 1)
    else:
        edges = np.asarray(bins, float)
    counts, edges = np.histogram(a, bins=edges, density=density)
    return {"edges": edges, "counts": counts}


# ====================================================================================================
# MÓDULO: visualization_3d
# ====================================================================================================
# Visualización científica de la red de filamentos 3D.
#
# Cada figura incluye: título descriptivo, unidades en los ejes, barra de color
# (cuando se colorea por una propiedad), el umbral de persistencia y una línea con
# los parámetros relevantes. Se guarda en PNG y PDF.
#
# Las proyecciones XY/XZ/YZ son **solo ayudas visuales** del análisis 3D (todos
# los números se calculan en 3D).


LABELS = {
    "length": "longitud L", "end_to_end": "distancia entre extremos D", "curvature_eff": "curvatura efectiva C_eff = L/D",
    "length_inertia": "longitud por tensor √λ_mayor", "thickness_minor": "grosor √λ_menor",
    "n_points": "número de puntos de muestreo", "angle_z_axis_deg": "ángulo del eje con Z [°]",
    "elevation_xy_axis_deg": "elevación sobre el plano XY [°]", "length_ratio_geom_tensor": "L / L_inercia",
    "angle_z_e2e_deg": "ángulo extremo-extremo con Z [°]", "bend_per_length": "giro por unidad de longitud",
}
LENGTH_PROPS = {"length", "end_to_end", "length_inertia", "thickness_minor", "thickness_mid"}


def _label(col: str, units: str) -> str:
    base = LABELS.get(col, col)
    return f"{base} [{units}]" if col in LENGTH_PROPS else base


def save_figure(fig, base: Path, formats: Sequence[str] = ("png", "pdf"), dpi: int = 300) -> List[Path]:
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for f in formats:
        p = base.with_suffix("." + f.lstrip("."))
        fig.savefig(p, dpi=dpi, bbox_inches="tight", metadata={"Software": "mdpl2_filaments_3d"} if f == "png" else None)
        out.append(p)
    plt.close(fig)
    return out


def footer(fig, text: str) -> None:
    fig.text(0.5, 0.005, text, ha="center", va="bottom", fontsize=7, color="0.35")


def _arc_colors(table: pd.DataFrame, col: Optional[str], vmin=None, vmax=None, cmap="viridis", log=False):
    n = len(table)
    if col is None:
        return np.zeros((n, 4)) + (0.1, 0.3, 0.7, 0.8), None
    v = table[col].to_numpy(float)
    ok = np.isfinite(v)
    lo = np.nanmin(v[ok]) if vmin is None and ok.any() else (vmin if vmin is not None else 0.0)
    hi = np.nanmax(v[ok]) if vmax is None and ok.any() else (vmax if vmax is not None else 1.0)
    if hi <= lo:
        hi = lo + 1e-9
    norm = colors.LogNorm(max(lo, 1e-12), hi) if log and lo > 0 else colors.Normalize(lo, hi)
    cols = matplotlib.colormaps[cmap](norm(np.where(ok, v, lo)))
    cols[~ok] = (0.6, 0.6, 0.6, 0.8)
    return cols, norm


def view_limits(skel: Skeleton):
    """Límites de dibujo: view_bounds (si el BBOX no es la región analizada) > BBOX > extensión de los puntos."""
    if skel.view_bounds is not None:
        return skel.view_bounds
    if skel.bbox is not None:
        return skel.bbox
    return skel.points.min(0), skel.points.max(0)


def _equal_box(ax, skel: Skeleton):
    lo, hi = view_limits(skel)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-9)))


def _lines(skel: Skeleton, dims=(0, 1, 2), max_arcs: Optional[int] = None):
    idx = range(skel.n_arcs) if max_arcs is None else np.linspace(0, skel.n_arcs - 1, min(max_arcs, skel.n_arcs)).astype(int)
    return [skel.polyline(i)[:, list(dims)] for i in idx], list(idx)


def plot_network_3d(skel: Skeleton, table: pd.DataFrame, sigma: float, color_by: Optional[str] = "length",
                    units: str = "Mpc/h", params: str = "", ax=None, vmin=None, vmax=None,
                    cmap: str = "viridis", log: bool = False, max_arcs: Optional[int] = None,
                    elev: float = 22, azim: float = -58, show_nodes: bool = True, standalone: bool = True):
    """Red de filamentos en 3D coloreada por una propiedad por arco."""
    own = ax is None
    if own:
        fig = plt.figure(figsize=(7.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    segs, idx = _lines(skel, max_arcs=max_arcs)
    cols, norm = _arc_colors(table.iloc[idx], color_by, vmin, vmax, cmap, log)
    ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=1.0), autolim=False)
    if show_nodes:
        mx = skel.cp[skel.cp["type"] == 3]
        ax.scatter(mx["x"], mx["y"], mx["z"], s=8, c="crimson", depthshade=False, label="máximos (nodos)")
        ax.legend(loc="upper left", fontsize=7)
    _equal_box(ax, skel)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(f"x [{units}]"); ax.set_ylabel(f"y [{units}]"); ax.set_zlabel(f"z [{units}]")
    ax.set_title(f"Red de filamentos 3D — persistencia {sigma:g}σ ({len(table)} arco(s))", fontsize=10)
    if norm is not None and own:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.16, label=_label(color_by, units))
    if own and standalone:
        footer(fig, params)
    return fig, ax, norm


def plot_projections(skel: Skeleton, table: pd.DataFrame, sigma: float, color_by: Optional[str] = "length",
                     units: str = "Mpc/h", params: str = "", cmap: str = "viridis", vmin=None, vmax=None,
                     log: bool = False):
    """Proyecciones XY, XZ, YZ del esqueleto 3D (solo visualización)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    cols, norm = _arc_colors(table, color_by, vmin, vmax, cmap, log)
    names = "xyz"
    lo, hi = view_limits(skel)
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        segs, _ = _lines(skel, dims=(i, j))
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.9))
        ax.set_xlim(lo[i], hi[i]); ax.set_ylim(lo[j], hi[j]); ax.set_aspect("equal")
        ax.set_xlabel(f"{names[i]} [{units}]"); ax.set_ylabel(f"{names[j]} [{units}]")
        ax.set_title(f"Proyección {names[i].upper()}{names[j].upper()}", fontsize=10)
    fig.suptitle(f"Proyecciones del esqueleto 3D — persistencia {sigma:g}σ (solo visualización; análisis 3D)", fontsize=11)
    if norm is not None:
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.8, pad=0.02, label=_label(color_by, units))
    footer(fig, params)
    return fig


def plot_threshold_comparison_3d(skels: Dict[float, Skeleton], tables: Dict[float, pd.DataFrame],
                                 color_by: str = "length", units: str = "Mpc/h", params: str = "",
                                 cmap: str = "viridis", log: bool = False):
    """Un panel 3D por umbral con escala de color COMÚN (comparable)."""
    sig = sorted(skels)
    allv = np.concatenate([tables[s][color_by].to_numpy(float) for s in sig])
    allv = allv[np.isfinite(allv)]
    vmin, vmax = (float(allv.min()), float(allv.max())) if len(allv) else (0.0, 1.0)
    fig = plt.figure(figsize=(6.2 * len(sig), 6.2))
    norm = None
    for k, s in enumerate(sig):
        ax = fig.add_subplot(1, len(sig), k + 1, projection="3d")
        _, _, norm = plot_network_3d(skels[s], tables[s], s, color_by, units, "", ax=ax, vmin=vmin, vmax=vmax,
                                     cmap=cmap, log=log, standalone=False, show_nodes=(k == 0))
    if norm is not None:
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=fig.axes, shrink=0.6, pad=0.06,
                     label=_label(color_by, units))
    fig.suptitle("Comparación 3D entre umbrales de persistencia (escala de color común)", fontsize=12)
    footer(fig, params)
    return fig


def plot_threshold_projections(skels: Dict[float, Skeleton], tables: Dict[float, pd.DataFrame], color_by: str = "length",
                               units: str = "Mpc/h", params: str = "", cmap: str = "viridis"):
    """Cuadrícula (umbral × plano) de proyecciones, escala de color común."""
    sig = sorted(skels)
    allv = np.concatenate([tables[s][color_by].to_numpy(float) for s in sig]); allv = allv[np.isfinite(allv)]
    vmin, vmax = (float(allv.min()), float(allv.max())) if len(allv) else (0.0, 1.0)
    fig, axes = plt.subplots(len(sig), 3, figsize=(13, 4.2 * len(sig)), squeeze=False)
    norm = colors.Normalize(vmin, vmax if vmax > vmin else vmin + 1e-9)
    names = "xyz"
    for r, s in enumerate(sig):
        cols, _ = _arc_colors(tables[s], color_by, vmin, vmax, cmap)
        lo, hi = view_limits(skels[s])
        for c, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
            ax = axes[r, c]
            segs, _ = _lines(skels[s], dims=(i, j))
            ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.8))
            ax.set_xlim(lo[i], hi[i]); ax.set_ylim(lo[j], hi[j]); ax.set_aspect("equal")
            ax.set_xlabel(f"{names[i]} [{units}]"); ax.set_ylabel(f"{names[j]} [{units}]")
            ax.set_title(f"{s:g}σ — {names[i].upper()}{names[j].upper()}", fontsize=9)
    fig.suptitle("Proyecciones por umbral de persistencia (solo visualización; análisis 3D)", fontsize=12)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.6, pad=0.02, label=_label(color_by, units))
    footer(fig, params)
    return fig


def plot_distributions(tables: Dict[float, pd.DataFrame], columns: Sequence[str] = ("length", "end_to_end", "curvature_eff", "thickness_minor"),
                       units: str = "Mpc/h", params: str = "", bins: int = 30, log_x: Sequence[str] = ("length", "end_to_end", "thickness_minor", "length_inertia")):
    """Histogramas normalizados superpuestos por umbral (una columna de la figura por propiedad)."""
    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)
    for ax, col in zip(axes[0], columns):
        pooled = np.concatenate([t[col].to_numpy(float) for t in tables.values()])
        pooled = pooled[np.isfinite(pooled)]
        use_log = col in log_x and (pooled > 0).any()
        if use_log:
            pooled = pooled[pooled > 0]
        if pooled.size == 0:
            continue
        edges = np.geomspace(pooled.min(), pooled.max(), bins + 1) if use_log and pooled.min() < pooled.max() else np.linspace(pooled.min(), pooled.max() + (pooled.min() == pooled.max()), bins + 1)
        for s in sorted(tables):
            v = tables[s][col].to_numpy(float); v = v[np.isfinite(v)]
            if use_log:
                v = v[v > 0]
            if len(v):
                ax.hist(v, bins=edges, density=True, histtype="step", linewidth=1.6, label=f"{s:g}σ (n={len(v)})")
        if use_log:
            ax.set_xscale("log")
        ax.set_xlabel(_label(col, units)); ax.set_ylabel("densidad"); ax.legend(fontsize=7)
    fig.suptitle("Distribuciones por umbral de persistencia (histogramas normalizados)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    footer(fig, params)
    return fig


def plot_summary_vs_sigma(comparison: pd.DataFrame, stats: Sequence[str] = ("n_arcs", "total_length", "length__median", "curvature_eff__median", "end_to_end__median", "thickness_minor__median"),
                          units: str = "Mpc/h", params: str = ""):
    """Estadísticos globales vs σ. Círculos = umbrales de la utilizados; cuadrados = extensión."""
    stats = [s for s in stats if s in comparison.columns]
    n = len(stats)
    cols_n = min(3, n)
    rows_n = int(np.ceil(n / cols_n))
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(4.6 * cols_n, 3.6 * rows_n), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, s in zip(axes.ravel(), stats):
        d = comparison.sort_values("sigma")
        ax.plot(d["sigma"], d[s], "-", color="0.6", zorder=1)
        th = d["sigma_origin"] == "thesis"
        ax.scatter(d.loc[th, "sigma"], d.loc[th, s], marker="o", s=45, c="C0", zorder=3, label="umbral utilizado")
        if (~th).any():
            ax.scatter(d.loc[~th, "sigma"], d.loc[~th, s], marker="s", s=28, c="C1", zorder=2, label="extensión")
        ax.set_xlabel("umbral de persistencia σ"); ax.set_ylabel(_label(s.split("__")[0], units) + (" (mediana)" if s.endswith("median") else ""))
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Propiedades de la red vs. umbral de persistencia", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    footer(fig, params)
    return fig


def plot_L_vs_C(tables: Dict[float, pd.DataFrame], units: str = "Mpc/h", params: str = ""):
    """Curvatura efectiva vs longitud, un panel por umbral."""
    sig = sorted(tables)
    fig, axes = plt.subplots(1, len(sig), figsize=(5.2 * len(sig), 4.6), squeeze=False)
    for ax, s in zip(axes[0], sig):
        t = tables[s]; ok = t["curvature_eff"].notna() & (t["length"] > 0)
        ax.scatter(t.loc[ok, "length"], t.loc[ok, "curvature_eff"], s=10, alpha=0.6, edgecolor="none")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.axhline(1, color="k", lw=0.8, ls="--")
        ax.set_xlabel(_label("length", units)); ax.set_ylabel(_label("curvature_eff", units))
        ax.set_title(f"{s:g}σ (n={int(ok.sum())})", fontsize=10); ax.grid(alpha=0.3, which="both")
    fig.suptitle("Curvatura efectiva vs longitud (C_eff = 1: trayectoria recta)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    footer(fig, params)
    return fig


def plot_L_geom_vs_tensor(tables: Dict[float, pd.DataFrame], units: str = "Mpc/h", params: str = ""):
    """Longitud geométrica vs longitud por tensor."""
    sig = sorted(tables)
    fig, axes = plt.subplots(1, len(sig), figsize=(5.2 * len(sig), 4.8), squeeze=False)
    for ax, s in zip(axes[0], sig):
        t = tables[s]; ok = (t["length"] > 0) & (t["length_inertia"] > 0)
        ax.scatter(t.loc[ok, "length"], t.loc[ok, "length_inertia"], s=10, alpha=0.6, edgecolor="none")
        if ok.any():
            m = float(max(t.loc[ok, "length"].max(), t.loc[ok, "length_inertia"].max()))
            ax.plot([0, m], [0, m], "k--", lw=0.8, label="identidad")
        ax.set_xlabel(_label("length", units)); ax.set_ylabel(_label("length_inertia", units))
        ax.set_title(f"{s:g}σ (n={int(ok.sum())})", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Longitud geométrica vs longitud por tensor de inercia", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    footer(fig, params)
    return fig


def plot_orientation(tables: Dict[float, pd.DataFrame], params: str = ""):
    """Distribución del ángulo del eje principal con Z (plegado a 0°–90°) y del azimut.
    Bajo isotropía, cos(θ_z) es uniforme: se dibuja la expectativa."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for s in sorted(tables):
        t = tables[s]
        a = t["angle_z_axis_deg"].to_numpy(float); a = a[np.isfinite(a)]
        z = t["azimuth_axis_deg"].to_numpy(float); z = z[np.isfinite(z)]
        if len(a):
            axes[0].hist(a, bins=np.linspace(0, 90, 19), density=True, histtype="step", linewidth=1.6, label=f"{s:g}σ (n={len(a)})")
        if len(z):
            axes[1].hist(z, bins=np.linspace(0, 180, 19), density=True, histtype="step", linewidth=1.6, label=f"{s:g}σ")
    th = np.linspace(0, 90, 200)
    axes[0].plot(th, np.sin(np.radians(th)) * np.pi / 180, "k--", lw=1, label="isotropía (esfera)")
    axes[1].axhline(1 / 180, color="k", ls="--", lw=1, label="uniforme")
    axes[0].set_xlabel("ángulo del eje principal con Z, θ_z [°] (0° = vertical)"); axes[0].set_ylabel("densidad")
    axes[1].set_xlabel("azimut del eje en el plano XY, φ [°]"); axes[1].set_ylabel("densidad")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle("Orientación 3D del eje principal (ejes sin sentido, plegados)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    footer(fig, params)
    return fig


def plotly_network(skel: Skeleton, table: pd.DataFrame, sigma: float, color_by: str, path: Path, units: str = "Mpc/h") -> Optional[Path]:
    """Versión interactiva (HTML) de la red 3D. Devuelve None si plotly no está instalado."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    v = table[color_by].to_numpy(float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    norm = colors.Normalize(lo, hi if hi > lo else lo + 1e-9)
    fig = go.Figure()
    for i in range(skel.n_arcs):
        p = skel.polyline(i)
        c = colors.to_hex(matplotlib.colormaps['viridis'](norm(v[i]))) if np.isfinite(v[i]) else "#999999"
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines", line=dict(color=c, width=3),
                                   name=f"arco {i}", hovertext=f"{color_by}={v[i]:.4g}", showlegend=False))
    fig.update_layout(title=f"Red de filamentos 3D — {sigma:g}σ (color: {color_by})",
                      scene=dict(xaxis_title=f"x [{units}]", yaxis_title=f"y [{units}]", zaxis_title=f"z [{units}]", aspectmode="data"))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


# ====================================================================================================
# MÓDULO: report
# ====================================================================================================
# Informe de resultados: SOLO gráficos y tablas, sin ningún análisis ni interpretación.
#
# Entrada: uno o más archivos NDskl_ascii de DisPerSE (cualquier nombre, cualquier cantidad).
# Salida (carpeta nueva por ejecución, nunca se sobrescribe)::
#
#     <out>/<nombre>/
#         INDICE.md                 lista de todo lo generado
#         definiciones.md           definición de cada columna/magnitud (no es análisis)
#         tablas/                   CSV: filamentos_<etiqueta>.csv, puentes_..., nodos_..., resumen_red.csv,
#                                   estadisticos_descriptivos.csv, top_diferencias_longitud.csv, resumen_<etiqueta>.json
#         figuras/<etiqueta>/       una figura por tipo (png y pdf) para cada archivo
#         figuras/comparacion/      las mismas figuras con todos los archivos lado a lado
#         metadatos/                run_manifest.json (hashes SHA-256, versiones, comando), advertencias.txt
#
# Magnitudes (idénticas a las del paquete de desarrollo): L = Σ|p_{i+1}-p_i|; D = |p_N-p_1|; C_eff = L/D (NaN si D=0,
# nunca 0); orientación del eje principal plegada a v_z>=0; tensor del conjunto de puntos del arco.
#
# Uso:
#     python filaments3d.py archivo1.NDskl archivo2.NDskl ... --units-in kpc/h --units-out Mpc/h
#     python filaments3d.py "carpeta/*.NDskl*" --out resultados
#     python filaments3d.py A=ruta_a.NDskl B=ruta_b.NDskl          # etiquetas propias
#     python filaments3d.py "*.NDskl*" --units-in kpc/h --units-out Mpc/h --animar   # SOLO animaciones (modo aparte, lento)


SIGMA_RE = re.compile(r"(?:^|[_\-.])(?:s|nsig|sigma)[_\-]?(\d+(?:\.\d+)?)(?=$|[_\-.])", re.I)
STAT_COLUMNS = ["length", "end_to_end", "curvature_eff", "n_points", "length_inertia", "thickness_mid", "thickness_minor",
                "angle_z_axis_deg", "elevation_xy_axis_deg", "angle_z_e2e_deg", "shared_length_fraction",
                "delta_tensor_minus_geom", "sigma_pct", "ratio_tensor_over_geom"]
GRID_COLS = 3


# ============================================================ entradas y etiquetas
def infer_sigma(name: str) -> Optional[float]:
    """σ deducido del nombre del archivo (`_s3`, `_nsig5`, `sigma_10`...) o None si no es unívoco."""
    ms = set(SIGMA_RE.findall(Path(name).name))
    return float(next(iter(ms))) if len(ms) == 1 else None


def _safe(text: str) -> str:
    """Nombre seguro para carpeta/archivo: sin extensión final corta, sin espacios ni puntos."""
    name = Path(text).name
    suf = Path(name).suffix
    if suf and len(suf) <= 9 and Path(name).stem:
        name = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_-") or "archivo"


def expand_inputs(items: Sequence[str]) -> List[Tuple[Optional[str], Path]]:
    """Acepta rutas, comodines ("*.NDskl*") y `ETIQUETA=ruta`. Devuelve [(etiqueta|None, ruta)] sin repetidos."""
    out: List[Tuple[Optional[str], Path]] = []
    seen = set()
    for it in items:
        label, spec = (None, it)
        if "=" in it and not Path(it).exists() and not any(c in it.split("=", 1)[0] for c in "*?[/\\"):
            label, spec = it.split("=", 1)
        hits = sorted(glob.glob(spec)) if any(c in spec for c in "*?[") else [spec]
        if not hits:
            raise SystemExit(f"error: el patrón '{spec}' no coincide con ningún archivo.")
        if label and len(hits) > 1:
            raise SystemExit(f"error: la etiqueta '{label}' se aplicó a {len(hits)} archivos; use una etiqueta por archivo.")
        for h in hits:
            p = Path(h)
            if not p.is_file():
                raise SystemExit(f"error: no existe el archivo {h}")
            if p.resolve() in seen:
                continue
            seen.add(p.resolve())
            out.append((label, p))
    if not out:
        raise SystemExit("error: no se indicó ningún archivo.")
    return out


def assign_labels(entries: List[Tuple[Optional[str], Path]]) -> List[Dict[str, object]]:
    """Etiqueta y título de cada archivo. Si TODOS tienen un σ único y distinto en el nombre, se etiquetan `s<σ>` y se
    ordenan por σ; en otro caso la etiqueta es el nombre del archivo (en el orden dado)."""
    sig = [infer_sigma(p.name) for _, p in entries]
    by_sigma = all(s is not None for s in sig) and len(set(sig)) == len(sig) and all(l is None for l, _ in entries)
    rows = []
    for (lab, p), s in zip(entries, sig):
        if lab:
            label, title = _safe(lab), lab
        elif by_sigma:
            label, title = f"s{s:g}", f"σ={s:g}"
        else:
            label, title = _safe(p.name), p.name
        rows.append({"label": label, "title": title, "path": p, "sigma": s})
    if by_sigma:
        rows.sort(key=lambda r: r["sigma"])
    used: Dict[str, int] = {}
    for r in rows:                                   # etiquetas únicas (se usan como nombre de carpeta/archivo)
        k = used.get(r["label"], 0)
        used[r["label"]] = k + 1
        if k:
            r["label"] = f"{r['label']}_{k + 1}"
    return rows


def short(text: str, n: int = 38) -> str:
    return text if len(text) <= n else text[: n // 2 - 1] + "…" + text[-(n // 2 - 1):]


# ============================================================ datos por archivo
@dataclass
class Item:
    label: str
    title: str
    path: Path
    sigma: Optional[float]
    skel: Skeleton
    table: pd.DataFrame
    nodes: pd.DataFrame
    bridges: pd.DataFrame
    summary: Dict[str, object]
    units: str
    tensor: str
    warnings: List[str] = field(default_factory=list)


def add_tensor_vs_geometry(t: pd.DataFrame) -> pd.DataFrame:
    """Δ = L_tensor − L_geom, σ% = |Δ|/L_geom·100, ratio = L_tensor/L_geom (solo arcos con L>0 y L_tensor>0)."""
    Lg, Lt = t["length"].to_numpy(float), t["length_inertia"].to_numpy(float)
    ok = np.isfinite(Lg) & np.isfinite(Lt) & (Lg > 0) & (Lt > 0)
    d = np.where(ok, Lt - Lg, np.nan)
    t["delta_tensor_minus_geom"] = d
    t["sigma_pct"] = np.where(ok, np.abs(d) / np.where(ok, Lg, 1) * 100.0, np.nan)
    t["ratio_tensor_over_geom"] = np.where(ok, Lt / np.where(ok, Lg, 1), np.nan)
    return t


def load_item(row: Dict[str, object], units_in: Optional[str], units_out: Optional[str], h: float, tensor: str) -> Item:
    p: Path = row["path"]                                     # type: ignore[assignment]
    sk = parse_ndskl_ascii(p)
    v = validate_skeleton(sk)                                  # en las unidades ORIGINALES del archivo
    v.raise_if_errors()
    warns = [f"{row['label']}: {w}" for w in v.warnings]
    units = units_in or "unidades del archivo"
    if units_in and units_out and units_in != units_out:
        sk = sk.scaled(float(convert_length(1.0, units_in, units_out, h=h)), units_out)
        units = units_out
    sigma = float(row["sigma"]) if row["sigma"] is not None else float("nan")
    t = build_filament_table(sk, sigma, "archivo", tensor)
    t.insert(0, "etiqueta", row["label"])
    t.insert(1, "archivo", p.name)
    t = add_tensor_vs_geometry(t)
    nd = build_node_table(sk, sigma)
    br = build_bridge_table(sk, sigma, tensor)
    return Item(str(row["label"]), str(row["title"]), p, row["sigma"], sk, t, nd, br, {}, units, tensor, warns)  # type: ignore[arg-type]


# ============================================================ paneles (una figura por tipo; 1 panel por archivo)
def _finite(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def _empty(ax, msg="sin datos válidos"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="0.4")
    ax.set_xticks([]); ax.set_yticks([])


def _hist_panel(col: str, xlabel_key: Optional[str] = None, color: str = "steelblue", xlabel: Optional[str] = None):
    def draw(ax, it: Item):
        v = _finite(it.table[col])
        ax.set_title(f"{short(it.title)}  (n={len(v)})", fontsize=9)
        if v.size == 0:
            return _empty(ax)
        ax.hist(v, bins=50, color=color, alpha=0.75, edgecolor="black", linewidth=0.3)
        ax.set_xlabel(xlabel or _label(col, it.units))
        ax.set_ylabel("número de filamentos")
        ax.grid(True, linestyle="--", alpha=0.5)
    return draw


def _scatter_panel(xcol: str, ycol: str):
    def draw(ax, it: Item):
        x, y = it.table[xcol].to_numpy(float), it.table[ycol].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.set_title(f"{short(it.title)}  (n={int(ok.sum())})", fontsize=9)
        if not ok.any():
            return _empty(ax)
        ax.scatter(x[ok], y[ok], s=4, alpha=0.35, color="darkgreen", edgecolors="none", rasterized=True)
        ax.set_xlabel(_label(xcol, it.units)); ax.set_ylabel(_label(ycol, it.units))
        ax.grid(True, linestyle="--", alpha=0.5)
    return draw


def _scatter_geom_tensor(ax, it: Item):
    x, y = it.table["length"].to_numpy(float), it.table["length_inertia"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    ax.set_title(f"{short(it.title)}  (n={int(ok.sum())})", fontsize=9)
    if not ok.any():
        return _empty(ax)
    ax.scatter(x[ok], y[ok], s=4, alpha=0.35, color="purple", edgecolors="none", rasterized=True)
    lim = [min(x[ok].min(), y[ok].min()), max(x[ok].max(), y[ok].max())]
    ax.plot(lim, lim, "k--", lw=0.8, label="y = x")
    ax.set_xlabel(_label("length", it.units)); ax.set_ylabel(_label("length_inertia", it.units))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=7); ax.grid(True, linestyle="--", alpha=0.5)


def _sample(a: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(a) <= n:
        return a
    return a[np.random.default_rng(seed).choice(len(a), n, replace=False)]


def _frame3d(ax, it: Item, title_extra: str = ""):
    _equal_box(ax, it.skel)
    ax.set_xlabel(f"X [{it.units}]"); ax.set_ylabel(f"Y [{it.units}]"); ax.set_zlabel(f"Z [{it.units}]")
    ax.set_title(f"{short(it.title)}{title_extra}", fontsize=9)
    ax.view_init(elev=22, azim=-58)


def _lines_panel(color_by: Optional[str]):
    def draw(ax, it: Item):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        segs, idx = _lines(it.skel)
        cols, norm = _arc_colors(it.table.iloc[idx], color_by)
        if color_by is None:
            cols = np.tile((0.1, 0.3, 0.8, 0.5), (len(idx), 1))
        col = Line3DCollection(segs, colors=cols, linewidths=0.7)
        col.set_rasterized(True)
        ax.add_collection3d(col, autolim=False)
        _frame3d(ax, it, f"  ({len(idx)} arcos)")
        if norm is not None:
            from matplotlib import cm
            ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, shrink=0.55, pad=0.14,
                               label=_label(color_by, it.units))
    return draw


def _points_panel(ax, it: Item):
    P = it.skel.points
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c="b", alpha=0.1, marker=".", s=3, rasterized=True)
    _frame3d(ax, it, f"  ({len(P)} puntos)")


def _density_panel(ax, it: Item, n_fit: int = 5000, n_show: int = 30000):
    from scipy.stats import gaussian_kde
    P = it.skel.points
    show = _sample(P, n_show, 1)
    try:
        kde = gaussian_kde(_sample(P, n_fit, 2).T)
        d = kde(show.T)
        o = d.argsort()
        sc = ax.scatter(show[o, 0], show[o, 1], show[o, 2], c=d[o], cmap="viridis", marker=".", alpha=0.5, s=3, rasterized=True)
        ax.figure.colorbar(sc, ax=ax, shrink=0.55, pad=0.14, label="densidad KDE (u. relativas)")
    except Exception as e:                                     # p. ej. puntos coplanares (matriz singular)
        ax.scatter(show[:, 0], show[:, 1], show[:, 2], c="0.5", marker=".", alpha=0.3, s=3, rasterized=True)
        ax.text2D(0.02, 0.95, f"KDE no disponible: {type(e).__name__}", transform=ax.transAxes, fontsize=7)
    note = f" (KDE: ajuste {min(n_fit, len(P))} pts, se muestran {min(n_show, len(P))} de {len(P)})"
    _frame3d(ax, it, "\n" + note)


def _nice_length(x: float) -> float:
    """Mayor valor de la forma {1,2,5}·10^k que no supera x (para la barra de escala)."""
    e = 10.0 ** np.floor(np.log10(x))
    return max(m * e for m in (1, 2, 5) if m * e <= x)


def densified_points(it: Item, step: float):
    """Puntos equiespaciados (≈`step`) a lo largo de todos los arcos (sin unir arcos distintos) y el peso de longitud de
    cada uno. Sirve para dibujar la red como una línea continua y para estimar la longitud de filamento por volumen."""
    P, off = it.skel.points, it.skel.offsets
    A, B = P[:-1], P[1:]
    ok = np.ones(len(A), bool)
    ok[off[1:-1] - 1] = False
    A, B = A[ok], B[ok]
    seglen = np.linalg.norm(B - A, axis=1)
    k = np.maximum(1, np.ceil(seglen / step).astype(np.int64))
    seg = np.repeat(np.arange(len(A)), k)
    first = np.cumsum(k) - k
    t = (np.arange(len(seg)) - np.repeat(first, k) + 0.5) / np.repeat(k, k)
    return A[seg] + (B[seg] - A[seg]) * t[:, None], (seglen / k)[seg]


def density_at_points(it: Item, n_vox: int = 110, max_points: int = 150000):
    """Densidad 3D de longitud de arco (suma de arcos; los tramos compartidos cuentan una vez por arco) por voxel, suavizada
    (gaussiana de 1.2 voxel) y evaluada sobre los puntos de los filamentos. Devuelve (puntos, densidad)."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    lo, hi = view_limits(it.skel)
    size = hi - lo
    vox = float(size.max()) / n_vox
    Q, w = densified_points(it, 0.5 * vox)
    shape = np.maximum(8, np.ceil(size / vox).astype(int))
    H, _ = np.histogramdd(Q, bins=shape, range=list(zip(lo, hi)), weights=w)
    H = gaussian_filter(H, 1.2)
    idx = (Q - lo) / size * shape - 0.5
    d = map_coordinates(H, idx.T, order=1, mode="nearest")
    if len(Q) > max_points:
        keep = np.random.default_rng(0).choice(len(Q), max_points, replace=False)
        Q, d = Q[keep], d[keep]
    return Q, d


def projected_density(it: Item, i: int, j: int, nbins: int = 700):
    """Mapa 2D de longitud de arco por píxel en la proyección (eje i, eje j), con los límites de dibujo del esqueleto.
    Los tramos compartidos por varios arcos suman una vez por arco. Suavizado gaussiano de 1.2 píxeles."""
    from scipy.ndimage import gaussian_filter
    lo, hi = view_limits(it.skel)
    wi, wj = hi[i] - lo[i], hi[j] - lo[j]
    nx = nbins if wi >= wj else max(50, int(round(nbins * wi / wj)))
    ny = nbins if wj >= wi else max(50, int(round(nbins * wj / wi)))
    pix = max(wi / nx, wj / ny)
    Q, w = densified_points(it, 0.5 * pix)                      # arcos interpolados a ≤ 0.5 píxel: línea continua
    H, _, _ = np.histogram2d(Q[:, i], Q[:, j], bins=[nx, ny], range=[[lo[i], hi[i]], [lo[j], hi[j]]], weights=w)
    return gaussian_filter(H, 1.2), (lo[i], hi[i], lo[j], hi[j])


def _density_map_panel(i: int, j: int):
    names = "XYZ"

    def draw(ax, it: Item, letter: str = ""):
        H, ext = projected_density(it, i, j)
        ax.set_facecolor("black")
        pos = H[H > 0]
        if pos.size:
            img = np.log1p(H / np.median(pos))
            ax.imshow(img.T, origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=np.percentile(img[H > 0], 99.7),
                      interpolation="bilinear", aspect="equal")
        mx = it.skel.cp[it.skel.cp["type"] == 3]
        if len(mx):
            ax.scatter(mx[["x", "y", "z"][i]], mx[["x", "y", "z"][j]], s=7, c="gold", marker="*", linewidths=0, label="máximos (nodos)")
            ax.legend(loc="upper right", fontsize=6, facecolor="black", edgecolor="0.4", labelcolor="white")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel(f"{names[i]} [{it.units}]", color="white"); ax.set_ylabel(f"{names[j]} [{it.units}]", color="white")
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("0.5")
        ax.set_title(f"{letter}{short(it.title)}", color="white", fontsize=10, loc="left")
        bar = _nice_length(0.15 * (ext[1] - ext[0]))                # barra de escala
        x0, y0 = ext[0] + 0.05 * (ext[1] - ext[0]), ext[2] + 0.06 * (ext[3] - ext[2])
        ax.plot([x0, x0 + bar], [y0, y0], color="white", lw=2.5, solid_capstyle="butt")
        unit = it.units.split("/")[0] + ("/h" if "/h" in it.units else "")
        ax.text(x0 + bar / 2, y0 + 0.012 * (ext[3] - ext[2]), f"{bar:g} {unit}", color="white", ha="center", va="bottom", fontsize=7)
        t = it.table[~it.table["is_duplicate"]] if "is_duplicate" in it.table else it.table
        box = (f"Puntos    {len(it.skel.points):>9,}\nFilamentos{len(t):>9,}\n"
               f"L mediana {np.nanmedian(t['length']):>9.4g}\nL total   {np.nansum(t['length']):>9.4g}")
        ax.text(0.98, 0.03, box, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, family="monospace", color="white",
                bbox=dict(facecolor="black", edgecolor="0.5", alpha=0.85, boxstyle="round,pad=0.35"))
    draw.dark = True
    return draw


def _density_map_3d_panel():
    def draw(ax, it: Item, letter: str = ""):
        Q, d = density_at_points(it)
        o = d.argsort()
        pos = d[d > 0]
        img = np.log1p(d / np.median(pos)) if pos.size else d
        vmax = np.percentile(img[d > 0], 99.7) if pos.size else 1.0
        ax.scatter(Q[o, 0], Q[o, 1], Q[o, 2], c=img[o], cmap="inferno", vmin=0, vmax=vmax, s=2.6, alpha=0.8,
                   linewidths=0, marker=".", depthshade=False, rasterized=True)
        mx = it.skel.cp[it.skel.cp["type"] == 3]
        if len(mx):
            ax.scatter(mx["x"], mx["y"], mx["z"], s=9, c="gold", marker="*", linewidths=0, depthshade=False,
                       rasterized=True, label="máximos (nodos)")
            ax.legend(loc="upper right", fontsize=6, facecolor="black", edgecolor="0.4", labelcolor="white")
        _equal_box(ax, it.skel)
        ax.view_init(elev=22, azim=-58)
        ax.set_facecolor("black")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.set_pane_color((0, 0, 0, 1))
                axis._axinfo["grid"]["color"] = (0.35, 0.35, 0.35, 0.5)
            except Exception:
                pass
        ax.set_xlabel(f"X [{it.units}]", color="white"); ax.set_ylabel(f"Y [{it.units}]", color="white")
        ax.set_zlabel(f"Z [{it.units}]", color="white")
        ax.tick_params(colors="white", labelsize=7)
        ax.set_title(f"{letter}{short(it.title)}", color="white", fontsize=10, loc="left")
        # barra de escala 3D (a lo largo de X, en la esquina inferior)
        lo, hi = view_limits(it.skel)
        bar = _nice_length(0.15 * (hi[0] - lo[0]))
        y0, z0, x0 = lo[1] + 0.02 * (hi[1] - lo[1]), lo[2], lo[0] + 0.04 * (hi[0] - lo[0])
        ax.plot([x0, x0 + bar], [y0, y0], [z0, z0], color="white", lw=2.5, solid_capstyle="butt")
        unit = it.units.split("/")[0] + ("/h" if "/h" in it.units else "")
        ax.text(x0 + bar / 2, y0, z0, f"{bar:g} {unit}", color="white", fontsize=7, ha="center", va="top")
        t = it.table[~it.table["is_duplicate"]] if "is_duplicate" in it.table else it.table
        box = (f"Puntos    {len(it.skel.points):>9,}\nFilamentos{len(t):>9,}\n"
               f"L mediana {np.nanmedian(t['length']):>9.4g}\nL total   {np.nansum(t['length']):>9.4g}")
        ax.text2D(0.01, 0.93, box, transform=ax.transAxes, ha="left", va="top", fontsize=6.5, family="monospace",
                  color="white", bbox=dict(facecolor="black", edgecolor="0.5", alpha=0.85, boxstyle="round,pad=0.35"))
    draw.dark = True
    return draw


PANELS: List[Tuple[str, str, Callable]] = [
    # (nombre de archivo, tipo, función)
    ("mapa_densidad_3d", "3d", _density_map_3d_panel()),
    ("mapa_densidad_proyeccion_xy", "2d", _density_map_panel(0, 1)),
    ("mapa_densidad_proyeccion_xz", "2d", _density_map_panel(0, 2)),
    ("mapa_densidad_proyeccion_yz", "2d", _density_map_panel(1, 2)),
    ("network3d_lineas", "3d", _lines_panel(None)),
    ("network3d_color_longitud", "3d", _lines_panel("length")),
    ("network3d_puntos", "3d", _points_panel),
    ("densidad3d", "3d", _density_panel),
    ("hist_distancia_extremos", "2d", _hist_panel("end_to_end", color="steelblue")),
    ("hist_longitud", "2d", _hist_panel("length", color="seagreen")),
    ("hist_curvatura", "2d", _hist_panel("curvature_eff", color="darkorange")),
    ("scatter_curvatura_vs_longitud", "2d", _scatter_panel("length", "curvature_eff")),
    ("hist_tensor_longitud_principal", "2d", _hist_panel("length_inertia", color="slateblue")),
    ("hist_tensor_grosor_menor", "2d", _hist_panel("thickness_minor", color="firebrick")),
    ("hist_tensor_grosor_medio", "2d", _hist_panel("thickness_mid", color="goldenrod")),
    ("hist_tensor_orientacion_angulo_z", "2d", _hist_panel("angle_z_axis_deg", color="teal")),
    ("scatter_longitud_vs_grosor_menor", "2d", _scatter_panel("length", "thickness_minor")),
    ("scatter_longitud_geom_vs_tensor", "2d", _scatter_geom_tensor),
    ("hist_delta_tensor_menos_geom", "2d", _hist_panel("delta_tensor_minus_geom", color="royalblue")),
    ("hist_sigma_porcentaje", "2d", _hist_panel("sigma_pct", color="crimson", xlabel="σ = |Δ| / L_geom [%]")),
    ("hist_ratio_tensor_sobre_geom", "2d", _hist_panel("ratio_tensor_over_geom", color="mediumpurple", xlabel="L_tensor / L_geom")),
]


def make_figure(items: Sequence[Item], kind: str, drawer: Callable, params: str):
    import matplotlib.pyplot as plt
    n = len(items)
    ncol = min(GRID_COLS, n)
    nrow = int(np.ceil(n / ncol))
    dark = getattr(drawer, "dark", False)
    w, hgt = (7.2, 6.2) if kind == "3d" else ((6.4, 6.0) if dark else (5.6, 4.3))
    fig = plt.figure(figsize=(w * ncol, hgt * nrow), facecolor="black" if dark else "white")
    for i, it in enumerate(items):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d" if kind == "3d" else None)
        if dark:
            drawer(ax, it, f"({chr(97 + i)}) " if n > 1 and i < 26 else "")
        else:
            drawer(ax, it)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    footer(fig, params)
    if dark:
        fig.texts[-1].set_color("0.8")
    return fig


def animate_item(it: Item, path: Path, frames: int, params: str) -> Optional[Path]:
    """GIF con rotación 360° (modo aparte `--animar`). Requiere Pillow (viene con matplotlib)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    segs, idx = _lines(it.skel)
    cols, _ = _arc_colors(it.table.iloc[idx], "length")
    ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=0.6), autolim=False)
    _frame3d(ax, it, f"  ({len(idx)} arcos)")
    footer(fig, params)
    ani = FuncAnimation(fig, lambda k: ax.view_init(elev=22, azim=-58 + 360.0 * k / frames), frames=frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(path, writer=PillowWriter(fps=15), dpi=70)
    plt.close(fig)
    return path


# ============================================================ tablas
def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def scalar_network_row(it: Item) -> Dict[str, object]:
    net = it.summary["network"]                                # type: ignore[index]
    row: Dict[str, object] = {"etiqueta": it.label, "archivo": it.path.name, "sigma_en_nombre": it.sigma,
                              "unidades": it.units, "convencion_tensor": it.tensor}
    for k, v in net.items():                                   # type: ignore[union-attr]
        if k == "sigma":                                       # ya está como sigma_en_nombre
            continue
        if isinstance(v, dict):
            if k == "node_connectivity":
                row.update({f"conectividad_nodos__{kk}": vv for kk, vv in v.items()})
            else:
                row[k] = json.dumps(v)
        else:
            row[k] = v
    return row


def descriptive_table(items: Sequence[Item]) -> pd.DataFrame:
    rows = []
    for it in items:
        for c in STAT_COLUMNS:
            if c in it.table.columns:
                rows.append({"etiqueta": it.label, "variable": c, **describe_series(it.table[c])})
    return pd.DataFrame(rows)


def top_differences(items: Sequence[Item], k: int = 10) -> pd.DataFrame:
    parts = []
    for it in items:
        t = it.table.dropna(subset=["delta_tensor_minus_geom"])
        t = t.reindex(t["delta_tensor_minus_geom"].abs().sort_values(ascending=False).index).head(k)
        parts.append(t[["etiqueta", "fil_id", "length", "length_inertia", "delta_tensor_minus_geom", "sigma_pct",
                        "ratio_tensor_over_geom", "n_points"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


DEFINITIONS = """# Definiciones (no es un análisis)

Todas las longitudes están en las unidades indicadas en `resumen_red.csv` (columna `unidades`).

| columna | definición |
|---|---|
| `fil_id` | índice del arco en el archivo (un "filamento" del archivo es un arco punto crítico ↔ punto crítico) |
| `length` | L = Σ‖p_{i+1} − p_i‖ sobre los puntos de muestreo del arco |
| `end_to_end` | D = ‖p_N − p_1‖ |
| `curvature_eff` | C_eff = L / D. **NaN** (no 0) si D = 0 o el arco tiene < 2 puntos; `curvature_status` indica el motivo |
| `n_points` | número de puntos de muestreo del arco |
| `length_inertia`, `thickness_mid`, `thickness_minor` | √λ_mayor, √λ_medio, √λ_menor del tensor de los puntos del arco (convención en `resumen_red.csv`: `covariance_mean` = covarianza con 1/N, eje mayor paralelo al filamento) |
| `axis_x/y/z` | eje principal (autovector del mayor autovalor), plegado a v_z ≥ 0 (un eje no tiene sentido) |
| `angle_z_axis_deg` | ángulo entre el eje principal y +Z, en [0°, 90°] |
| `elevation_xy_axis_deg` | 90° − `angle_z_axis_deg` (elevación sobre el plano XY) |
| `angle_z_e2e_deg`, `u_x/u_y/u_z` | lo mismo para el vector entre extremos |
| `delta_tensor_minus_geom` | Δ = `length_inertia` − `length` (solo arcos con ambos > 0) |
| `sigma_pct` | |Δ| / `length` · 100 |
| `ratio_tensor_over_geom` | `length_inertia` / `length` |
| `shared_length_fraction` | fracción de la longitud del arco que recorre segmentos compartidos con otros arcos |
| `length_unique_share` | parte de la longitud del arco contada sin repetir segmentos compartidos |
| `component_id` | componente conexa de la red (por puntos críticos compartidos) |
| `touches_boundary` | algún extremo es un punto crítico marcado como frontera en el archivo |
| `is_duplicate` | el arco repite exactamente la geometría de otro |
| `total_length` (resumen) | Σ longitudes de arco (los tramos compartidos se cuentan una vez por arco) |
| `total_length_unique` (resumen) | longitud geométrica de la red: cada segmento compartido se cuenta una sola vez |
| `volume`, `volume_source` | volumen usado para las densidades: región declarada con `--region`, o la caja que envuelve los puntos (cota inferior) |
| `puentes_<etiqueta>.csv` | unidades máximo–silla–máximo ensambladas a partir de los arcos (vista alternativa a la de arcos) |
| `mapa_densidad_proyeccion_xy/xz/yz` | mismo mapa de densidad proyectado sobre un plano (2D): longitud de arco por píxel, arcos interpolados a 0.5 píxel, suavizado gaussiano de 1.2 píxeles, escala log1p/mediana, normalizada por panel |
| `mapa_densidad_3d` | longitud de arco por voxel (suma de arcos: los tramos compartidos cuentan una vez por arco) evaluada sobre los filamentos interpolados; suavizado gaussiano de 1.2 voxel, escala log1p/mediana, normalizada por panel; las estrellas son los máximos |
"""


# ============================================================ ejecución
def set_region_and_volume(items: Sequence[Item], region, warnings: List[str]):
    """Fija los límites de dibujo (`view_bounds`) y devuelve (volumen, fuente, extensión empírica)."""
    ext = empirical_extent([it.skel for it in items])
    if region:
        size = np.array([b[1] - b[0] for b in region], float)
        volume, vsrc = float(np.prod(size)), "región declarada (--region)"
        lo = np.array([b[0] for b in region], float)
        hi = np.array([b[1] for b in region], float)
    else:
        volume, vsrc = float(ext["volume"]), "extensión empírica de los puntos (cota inferior)"
        warnings.append("No se declaró --region: el volumen de las densidades es el de la caja que envuelve los puntos de "
                        "todos los archivos (cota inferior de la región real).")
        if ext.get("bbox_size") is not None and not ext.get("bbox_is_region"):
            warnings.append(f"El BBOX del archivo ({np.round(ext['bbox_size'], 4).tolist()}) no es la región ocupada por el esqueleto; "
                            "las figuras usan la extensión empírica con 2 % de margen.")
        pad = 0.02 * np.asarray(ext["size"], float)
        lo, hi = np.asarray(ext["lo"]) - pad, np.asarray(ext["hi"]) + pad
    for it in items:
        it.skel.view_bounds = (lo, hi)
    return volume, vsrc, ext


def prepare_run_dir(out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    d = out / name
    if d.exists():
        d = out / f"{name}_{dt.datetime.now():%Y%m%d_%H%M%S}"
        k = 1
        while d.exists():
            k += 1
            d = out / f"{name}_{dt.datetime.now():%Y%m%d_%H%M%S}_{k}"
    d.mkdir(parents=True)
    return d


def run_report(files: Sequence[str], out: str = "resultados", name: str = "resultados_filamentos",
               units_in: Optional[str] = None, units_out: Optional[str] = None, h: float = 0.678,
               region: Optional[Tuple[Tuple[float, float], ...]] = None, tensor: str = "covariance_mean",
               formats: Sequence[str] = ("png", "pdf"), dpi: int = 200,
               per_file_figures: bool = True, comparison_figures: bool = True, argv: Optional[Sequence[str]] = None) -> Path:
    matplotlib.use("Agg")
    if units_out and not units_in:
        raise SystemExit("error: --units-out requiere --units-in (no se adivinan las unidades del archivo).")
    rows = assign_labels(expand_inputs(files))
    run = prepare_run_dir(Path(out), name)
    tab_dir, fig_dir, meta_dir = run / "tablas", run / "figuras", run / "metadatos"
    for d in (tab_dir, fig_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []
    errors: List[str] = []
    inputs: Dict[str, Dict[str, object]] = {}
    items: List[Item] = []
    print(f"Resultados en: {run}")
    for i, r in enumerate(rows, 1):
        p: Path = r["path"]                                    # type: ignore[assignment]
        inputs[str(r["label"])] = {"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
        print(f"[{i}/{len(rows)}] {r['label']}: leyendo y calculando métricas ({p.name})", flush=True)
        try:
            items.append(load_item(r, units_in, units_out, h, tensor))
        except Exception as e:                                 # un archivo defectuoso no detiene a los demás
            errors.append(f"{r['label']} ({p.name}): {type(e).__name__}: {e}")
            print(f"    ERROR: {errors[-1]}", file=sys.stderr)
    if not items:
        (meta_dir / "advertencias.txt").write_text("\n".join(errors) + "\n", encoding="utf-8")
        raise SystemExit("error: ningún archivo pudo procesarse:\n  " + "\n  ".join(errors))

    volume, vsrc, ext = set_region_and_volume(items, region, warnings)
    for it in items:
        it.summary = network_summary(it.skel, it.table, it.nodes, float(it.sigma) if it.sigma is not None else float("nan"), volume=volume)
        it.summary["network"]["volume_source"] = vsrc          # type: ignore[index]
        warnings += it.warnings

    # --- tablas
    print("Escribiendo tablas…", flush=True)
    files_out: List[Path] = []
    for it in items:
        files_out.append(write_csv(it.table, tab_dir / f"filamentos_{it.label}.csv"))
        files_out.append(write_csv(it.bridges, tab_dir / f"puentes_{it.label}.csv"))
        files_out.append(write_csv(it.nodes, tab_dir / f"nodos_{it.label}.csv"))
        pj = tab_dir / f"resumen_{it.label}.json"
        pj.write_text(json.dumps(json_safe(it.summary), indent=2, ensure_ascii=False), encoding="utf-8")
        files_out.append(pj)
    files_out.append(write_csv(pd.DataFrame([scalar_network_row(it) for it in items]), tab_dir / "resumen_red.csv"))
    files_out.append(write_csv(descriptive_table(items), tab_dir / "estadisticos_descriptivos.csv"))
    files_out.append(write_csv(top_differences(items), tab_dir / "top_diferencias_longitud.csv"))
    (run / "definiciones.md").write_text(DEFINITIONS, encoding="utf-8")

    # --- figuras
    fmts = list(formats)
    nfig = 0
    if fmts:
        for pname, kind, drawer in PANELS:
            if per_file_figures:
                for it in items:
                    try:
                        par = f"{it.title} | archivo: {it.path.name} | unidades={it.units} | tensor={it.tensor}"
                        files_out += save_figure(make_figure([it], kind, drawer, par), fig_dir / it.label / pname, fmts, dpi)
                        nfig += 1
                    except Exception as e:
                        warnings.append(f"figura {pname} ({it.label}) no generada: {type(e).__name__}: {e}")
            if comparison_figures and len(items) > 1:
                try:
                    par = f"{len(items)} archivos | unidades={items[0].units} | tensor={tensor}"
                    files_out += save_figure(make_figure(items, kind, drawer, par), fig_dir / "comparacion" / pname, fmts, dpi)
                    nfig += 1
                except Exception as e:
                    warnings.append(f"figura de comparación {pname} no generada: {type(e).__name__}: {e}")
            print(f"  figura {pname} lista", flush=True)

    # --- metadatos
    (meta_dir / "advertencias.txt").write_text("\n".join(warnings + [f"ERROR: {e}" for e in errors]) + "\n", encoding="utf-8")
    manifest = {"fecha_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "comando": " ".join(argv if argv is not None else sys.argv),
                "python": platform.python_version(), "versiones": package_versions(), "entradas": inputs,
                "parametros": {"units_in": units_in, "units_out": units_out, "h": h, "tensor": tensor, "region": region,
                               "formatos": fmts, "dpi": dpi, "volumen": volume, "volumen_fuente": vsrc},
                "extension_empirica": ext, "archivos_con_error": errors, "n_figuras": nfig}
    (meta_dir / "run_manifest.json").write_text(json.dumps(json_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    idx = ["# Índice de resultados", "", f"Archivos procesados: {len(items)}" + (f" (con error: {len(errors)})" if errors else ""), ""]
    idx += [f"* `{it.label}` ← `{it.path.name}`  ({len(it.table)} arcos)" for it in items]
    idx += ["", "Definiciones de las columnas: `definiciones.md`. Advertencias: `metadatos/advertencias.txt`.", "", "## Archivos generados", ""]
    idx += [f"* `{Path(f).relative_to(run).as_posix()}`" for f in sorted(set(map(str, files_out)))]
    (run / "INDICE.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    print(f"Listo: {len(items)} archivo(s), {nfig} figura(s) → {run}")
    if warnings:
        print(f"Advertencias ({len(warnings)}): ver metadatos/advertencias.txt")
    return run


def run_animations(files: Sequence[str], out: str = "resultados", name: str = "animaciones",
                   units_in: Optional[str] = None, units_out: Optional[str] = None, h: float = 0.678,
                   region=None, frames: int = 60) -> Path:
    """MODO APARTE (`--animar`): solo GIF de rotación 3D por archivo. No calcula tablas ni figuras (lento)."""
    matplotlib.use("Agg")
    if units_out and not units_in:
        raise SystemExit("error: --units-out requiere --units-in (no se adivinan las unidades del archivo).")
    rows = assign_labels(expand_inputs(files))
    run = prepare_run_dir(Path(out), name)
    print(f"Animaciones en: {run}")
    items: List[Item] = []
    errors: List[str] = []
    for i, r in enumerate(rows, 1):
        p: Path = r["path"]                                    # type: ignore[assignment]
        print(f"[{i}/{len(rows)}] {r['label']}: leyendo {p.name}", flush=True)
        try:
            sk = parse_ndskl_ascii(p)
            validate_skeleton(sk).raise_if_errors()
            units = units_in or "unidades del archivo"
            if units_in and units_out and units_in != units_out:
                sk = sk.scaled(float(convert_length(1.0, units_in, units_out, h=h)), units_out)
                units = units_out
            lengths = [float(np.linalg.norm(np.diff(sk.polyline(k), axis=0), axis=1).sum()) for k in range(sk.n_arcs)]
            items.append(Item(str(r["label"]), str(r["title"]), p, r["sigma"], sk, pd.DataFrame({"length": lengths}),  # type: ignore[arg-type]
                              pd.DataFrame(), pd.DataFrame(), {}, units, "-"))
        except Exception as e:
            errors.append(f"{r['label']} ({p.name}): {type(e).__name__}: {e}")
            print(f"    ERROR: {errors[-1]}", file=sys.stderr)
    if not items:
        raise SystemExit("error: ningún archivo pudo procesarse:\n  " + "\n  ".join(errors))
    warns: List[str] = []
    set_region_and_volume(items, region, warns)
    for k, it in enumerate(items, 1):
        print(f"[{k}/{len(items)}] {it.label}: generando GIF ({frames} cuadros)…", flush=True)
        animate_item(it, run / f"{it.label}_rotacion3d.gif", frames, f"{it.title} | archivo: {it.path.name} | unidades={it.units}")
    (run / "advertencias.txt").write_text("\n".join(warns + [f"ERROR: {e}" for e in errors]) + "\n", encoding="utf-8")
    print(f"Listo: {len(items)} animación(es) → {run}")
    return run


def json_safe(obj):
    """Convierte numpy/NaN a JSON válido (NaN → null)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    return str(obj)


# ============================================================ línea de comandos
def parse_region(text: str):
    v = [float(t) for t in text.split(",")]
    if len(v) != 6 or not (v[0] < v[1] and v[2] < v[3] and v[4] < v[5]):
        raise SystemExit("error: --region = x0,x1,y0,y1,z0,z1 con min < max (en las unidades de salida).")
    return (v[0], v[1]), (v[2], v[3]), (v[4], v[5])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", metavar="ARCHIVO", help="NDskl_ascii de DisPerSE: rutas, comodines o ETIQUETA=ruta (cualquier cantidad).")
    p.add_argument("--out", default="resultados", help="carpeta donde se crea la carpeta de la ejecución (def. resultados).")
    p.add_argument("--name", default="resultados_filamentos", help="nombre de la carpeta de la ejecución (no se sobrescribe).")
    p.add_argument("--units-in", help="unidades de las coordenadas del archivo (p. ej. kpc/h, Mpc/h). No se adivinan.")
    p.add_argument("--units-out", help="unidades de salida; convierte desde --units-in.")
    p.add_argument("--h", type=float, default=0.678, help="h, solo si convierte entre unidades con/sin h (def. 0.678).")
    p.add_argument("--region", help="x0,x1,y0,y1,z0,z1 de la región analizada (en unidades de salida); define volumen y límites de figuras.")
    p.add_argument("--tensor", default="covariance_mean", choices=["covariance_mean", "covariance_sum", "mechanical_sum"],
                   help="convención del tensor (def. covariance_mean).")
    p.add_argument("--formats", default="png,pdf", help="formatos de figura separados por coma (def. png,pdf; vacío = sin figuras).")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-per-file", action="store_true", help="no generar las figuras individuales de cada archivo.")
    p.add_argument("--no-comparison", action="store_true", help="no generar las figuras con todos los archivos juntos.")
    p.add_argument("--animar", action="store_true",
                   help="MODO APARTE: genera SOLO los GIF de rotación 3D (en su propia carpeta) y nada más; es lento, por eso no se mezcla con el resto.")
    p.add_argument("--frames", type=int, default=60, help="cuadros del GIF en modo --animar (def. 60).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        region = parse_region(a.region) if a.region else None
        if a.animar:
            run_animations(a.files, a.out, "animaciones" if a.name == "resultados_filamentos" else a.name, a.units_in, a.units_out,
                           a.h, region, a.frames)
        else:
            run_report(a.files, a.out, a.name, a.units_in, a.units_out, a.h, region, a.tensor,
                       [f.strip() for f in a.formats.split(",") if f.strip()], a.dpi,
                       not a.no_per_file, not a.no_comparison, argv=["filaments3d.py"] + list(argv if argv is not None else sys.argv[1:]))
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
