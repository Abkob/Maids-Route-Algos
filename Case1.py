"""
Van Territory Planner — 4-Pipeline Benchmark
  A) MCF + OSRM Path          — cost-first two-stage baseline
  B) P-Median + OSRM Path     — territory-first two-stage baseline
  C) MCF + OR-Tools Path      — same zones as A, better local ordering
  D) Full OR-Tools CVRP       — integrated single-stage benchmark
All pipelines share one explicit route model. Routes follow real streets.
"""
import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import distance_matrix as scipy_dm
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import networkx as nx
import contextily as ctx
from pyproj import Transformer
import overpy
import requests
import tempfile, os, time, warnings, math
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────
#  PAGE CONFIG & STYLE
# ──────────────────────────────────────────────────────
st.set_page_config(page_title="Van Territory Planner", page_icon="🚐",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif}
.block-container{padding-top:1.15rem}
.stNumberInput input{font-size:1.6rem!important;font-weight:600!important;
  text-align:center;font-family:'IBM Plex Mono',monospace!important}
.stButton>button{background:#0f172a;color:#e2e8f0;border:1px solid #334155;
  border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:.8rem;
  letter-spacing:.05em;transition:all .2s;padding:.45rem 1rem}
.stButton>button:hover{background:#1e3a5f;border-color:#60a5fa;color:#fff}
.stButton>button[kind="primary"]{background:#1d4ed8;border-color:#3b82f6;color:#fff;
  font-size:.9rem;padding:.55rem 1.2rem}
.stButton>button[kind="primary"]:hover{background:#2563eb}
.pipeline-card{background:#0d1526;border:2px solid #1e3a5f;border-radius:10px;
  padding:1rem 1.2rem;margin-bottom:.6rem}
.pipeline-title{font-family:'IBM Plex Mono',monospace;font-size:.95rem;
  font-weight:600;color:#e2e8f0;margin-bottom:.3rem}
.pipeline-sub{font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:#60a5fa}
.pipeline-desc{font-size:.78rem;color:#94a3b8;margin-top:.4rem;line-height:1.5}
.metric-card{background:#0f172a;border:1px solid #1e293b;border-radius:6px;
  padding:.7rem 1rem;margin-bottom:.5rem}
.metric-label{font-family:'IBM Plex Mono',monospace;font-size:.66rem;color:#64748b;
  text-transform:uppercase;letter-spacing:.08em}
.metric-value{font-family:'IBM Plex Mono',monospace;font-size:1.3rem;font-weight:600;color:#e2e8f0}
.metric-good{color:#4ade80}.metric-warn{color:#facc15}.metric-bad{color:#f87171}
.insight-box{background:#0d1f3c;border-left:3px solid #3b82f6;border-radius:4px;
  padding:.65rem .9rem;font-size:.75rem;font-family:monospace;color:#94a3b8;margin-top:.6rem}
.osrm-badge{background:#064e3b;border:1px solid #10b981;color:#6ee7b7;
  padding:2px 8px;border-radius:12px;font-family:monospace;font-size:.68rem;margin-left:.4rem}
[data-testid="stSidebar"]{background:#0a0f1e;border-right:1px solid #1e293b}
[data-testid="stSidebar"] label{font-family:'IBM Plex Mono',monospace;
  font-size:.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em}
h1{font-family:'IBM Plex Mono',monospace;font-size:1.25rem!important;color:#e2e8f0;line-height:1.25!important}
.page-title{display:flex;align-items:center;gap:.5rem;margin:0 0 .15rem 0;padding-top:.1rem;
  font-family:'IBM Plex Mono',monospace;font-size:1.9rem;font-weight:600;line-height:1.2;color:#e2e8f0}
.page-title-icon{display:inline-flex;align-items:center;line-height:1;transform:translateY(1px)}
.stTabs [data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace;font-size:.72rem}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
#  COORDINATE UTILITIES
# ──────────────────────────────────────────────────────
_T = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def to_wm(ll):   x,y=_T.transform(ll[:,0],ll[:,1]); return np.column_stack([x,y])
def norm(ll,bb): n=np.zeros_like(ll); n[:,0]=(ll[:,0]-bb[0])/(bb[2]-bb[0]); n[:,1]=(ll[:,1]-bb[1])/(bb[3]-bb[1]); return n
def denorm(n,bb): ll=np.zeros_like(n); ll[:,0]=n[:,0]*(bb[2]-bb[0])+bb[0]; ll[:,1]=n[:,1]*(bb[3]-bb[1])+bb[1]; return ll

CITIES = {
    "🇱🇧 Beirut":    (35.476,33.856,35.563,33.915),
    "🇦🇪 Dubai":     (55.230,25.185,55.345,25.245),
    "🇸🇦 Riyadh":    (46.680,24.630,46.820,24.750),
    "🇪🇬 Cairo":     (31.215,30.030,31.330,30.115),
    "🇬🇧 London":    (-0.160,51.490,-0.060,51.545),
    "🇺🇸 New York":  (-74.010,40.705,-73.940,40.775),
    "🇫🇷 Paris":     (2.290,48.840,2.410,48.895),
    "🇯🇵 Tokyo":     (139.695,35.655,139.775,35.715),
    "🇸🇬 Singapore": (103.810,1.285,103.890,1.365),
}

POI_FILTERS = {
    "⛽ All delivery POIs": ('node["shop"]','node["amenity"~"restaurant|cafe|pharmacy|bank|hotel"]'),
    "🏪 Shops":             ('node["shop"]',),
    "🍽️ Food & Beverage":  ('node["amenity"~"restaurant|cafe|bar|fast_food"]',),
    "🏥 Healthcare":        ('node["amenity"~"pharmacy|hospital|clinic|doctors"]',),
    "🏦 Banks & Services":  ('node["amenity"~"bank|atm|post_office"]',),
}

PALETTE = ["#3b82f6","#f59e0b","#10b981","#ef4444","#8b5cf6",
           "#06b6d4","#f97316","#84cc16","#ec4899","#14b8a6",
           "#a855f7","#eab308","#22d3ee","#fb923c","#4ade80"]

# ──────────────────────────────────────────────────────
#  OSRM  — real road-network distances & geometries
# ──────────────────────────────────────────────────────
# OSRM public API — HTTP only (the demo server does not serve HTTPS)
# Per official docs: http://router.project-osrm.org/{service}/v1/{profile}/{coordinates}
OSRM_BASES = [
    "http://router.project-osrm.org",           # official demo server
    "http://routing.openstreetmap.de/routed-car", # OSM mirror
]
OSRM_CHUNK = 100  # public server handles up to ~200 coordinates per table request


def _osrm_table_chunk(coords, base):
    """
    Fetch NxN duration (seconds) AND distance (meters) matrices from OSRM table API.
    Returns (dur_matrix, dist_matrix) or (None, None) on failure.
    """
    coord_str = ";".join(f"{lon},{lat}" for lon, lat in coords)
    url = f"{base}/table/v1/driving/{coord_str}?annotations=duration,distance"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data.get("code") == "Ok":
                dur  = np.array(data["durations"], dtype=float)
                # OSRM may omit 'distances' on some servers — handle gracefully
                dist = np.array(data["distances"], dtype=float) if "distances" in data else None
                return dur, dist
    except Exception:
        pass
    return None, None


def osrm_distance_matrix(lonlat: np.ndarray, status_cb=None):
    """
    Build full NxN duration (seconds) and distance (meters) matrices via OSRM table API.
    Returns (dur_matrix, dist_matrix, osrm_used).
    dist_matrix may be None if the server doesn't return distances.
    Falls back to Euclidean duration on failure; dist_matrix will be None in fallback.
    """
    n = len(lonlat)
    def _clean(mat):
        finite = mat[np.isfinite(mat)]
        fill = finite.max() * 2 if len(finite) > 0 else 1e6
        return np.where(np.isfinite(mat), mat, fill)

    for base in OSRM_BASES:
        try:
            coord_str = ";".join(f"{lon},{lat}" for lon, lat in lonlat)
            url = f"{base}/table/v1/driving/{coord_str}?annotations=duration,distance"
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if data.get("code") == "Ok":
                    dur = np.array(data["durations"], dtype=float)
                    if dur.shape == (n, n):
                        dur  = _clean(dur)
                        dist = None
                        if "distances" in data:
                            d = np.array(data["distances"], dtype=float)
                            dist = _clean(d) if d.shape == (n, n) else None
                        return dur, dist, True
        except Exception:
            continue

    # Fallback — Euclidean (unitless, not seconds/meters)
    if status_cb:
        status_cb("⚠️ OSRM unreachable — using Euclidean distance (approximate)")
    n_pts = norm(lonlat, (lonlat[:,0].min(), lonlat[:,1].min(),
                           lonlat[:,0].max(), lonlat[:,1].max()))
    euc = scipy_dm(n_pts, n_pts)
    return euc, None, False


def _clean_osrm_values(mat: np.ndarray) -> np.ndarray:
    finite = mat[np.isfinite(mat)]
    fill = finite.max() * 2 if len(finite) > 0 else 1e6
    return np.where(np.isfinite(mat), mat, fill)


def _approx_block_matrices(src_lonlat: np.ndarray, dst_lonlat: np.ndarray,
                           speed_kmh: float = 32.0):
    """
    Fallback rectangular travel matrices from straight-line geography.
    Distance is meters; duration assumes a fixed average road speed.
    """
    shape = (len(src_lonlat), len(dst_lonlat))
    if shape[0] == 0 or shape[1] == 0:
        return np.zeros(shape), np.zeros(shape)

    lon1 = np.radians(src_lonlat[:, 0])[:, None]
    lat1 = np.radians(src_lonlat[:, 1])[:, None]
    lon2 = np.radians(dst_lonlat[:, 0])[None, :]
    lat2 = np.radians(dst_lonlat[:, 1])[None, :]
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
    dist_m = 6_371_000.0 * c
    dur_s = dist_m / max(speed_kmh * 1000.0 / 3600.0, 1e-9)
    return dur_s, dist_m


def osrm_rectangular_matrix(src_lonlat: np.ndarray, dst_lonlat: np.ndarray):
    """
    Fetch a rectangular OSRM table block for source -> destination travel.
    Returns (dur_s, dist_m, osrm_used_for_block).
    Falls back to straight-line meters + fixed-speed seconds when OSRM fails.
    """
    shape = (len(src_lonlat), len(dst_lonlat))
    if shape[0] == 0 or shape[1] == 0:
        zeros = np.zeros(shape)
        return zeros, zeros, True

    block_step = max(1, OSRM_CHUNK // 2)

    for base in OSRM_BASES:
        dur_out = np.full(shape, np.nan, dtype=float)
        dist_out = np.full(shape, np.nan, dtype=float)
        dist_ok = True
        failed = False

        for i0 in range(0, len(src_lonlat), block_step):
            src_block = src_lonlat[i0:i0 + block_step]
            for j0 in range(0, len(dst_lonlat), block_step):
                dst_block = dst_lonlat[j0:j0 + block_step]
                coords = np.vstack([src_block, dst_block])
                coord_str = ";".join(f"{lon},{lat}" for lon, lat in coords)
                src_idx = ";".join(str(i) for i in range(len(src_block)))
                dst_idx = ";".join(str(len(src_block) + j) for j in range(len(dst_block)))
                url = (
                    f"{base}/table/v1/driving/{coord_str}"
                    f"?annotations=duration,distance&sources={src_idx}&destinations={dst_idx}"
                )
                try:
                    r = requests.get(url, timeout=20)
                    if r.status_code != 200:
                        failed = True
                        break
                    data = r.json()
                    if data.get("code") != "Ok":
                        failed = True
                        break
                    dur_chunk = np.array(data["durations"], dtype=float)
                    if dur_chunk.shape != (len(src_block), len(dst_block)):
                        failed = True
                        break
                    dur_out[i0:i0 + len(src_block), j0:j0 + len(dst_block)] = dur_chunk

                    if "distances" in data:
                        dist_chunk = np.array(data["distances"], dtype=float)
                        if dist_chunk.shape == (len(src_block), len(dst_block)):
                            dist_out[i0:i0 + len(src_block), j0:j0 + len(dst_block)] = dist_chunk
                        else:
                            dist_ok = False
                    else:
                        dist_ok = False
                except Exception:
                    failed = True
                    break
            if failed:
                break

        if not failed and np.isfinite(dur_out).all():
            dur_out = _clean_osrm_values(dur_out)
            if dist_ok and np.isfinite(dist_out).all():
                dist_out = _clean_osrm_values(dist_out)
            else:
                dist_out = None
            return dur_out, dist_out, True

    dur_fb, dist_fb = _approx_block_matrices(src_lonlat, dst_lonlat)
    return dur_fb, dist_fb, False


def osrm_route_geometry(ordered_lonlat: np.ndarray) -> list[tuple] | None:
    """
    Get road-following polyline for an ordered stop sequence.
    Returns list of (x_wm, y_wm) Web Mercator pairs, or None on failure.
    Use ordered_route_geometry() when you also need distance/duration.
    """
    result = ordered_route_geometry(ordered_lonlat)
    return result[0] if result else None


def ordered_route_geometry(ordered_lonlat: np.ndarray):
    """
    Fetch OSRM /route/ for a sequence of stops in the given order.
    Returns (geom_wm, dist_m, dur_s) or None on failure.
      geom_wm : list of (x, y) Web Mercator tuples — actual street polyline
      dist_m  : total road distance in meters  (float)
      dur_s   : total road duration in seconds (float)
    The input MUST be in the correct visit order — OSRM honours the sequence.
    """
    if len(ordered_lonlat) < 2:
        return None
    coord_str = ";".join(f"{lon},{lat}" for lon, lat in ordered_lonlat)
    for base in OSRM_BASES:
        try:
            url = (f"{base}/route/v1/driving/{coord_str}"
                   f"?overview=full&geometries=geojson&continue_straight=false")
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if data.get("code") == "Ok":
                    route   = data["routes"][0]
                    coords  = route["geometry"]["coordinates"]
                    dist_m  = float(route.get("distance", 0.0))   # meters
                    dur_s   = float(route.get("duration", 0.0))   # seconds
                    arr     = np.array(coords)
                    wm      = to_wm(arr)
                    return list(map(tuple, wm)), dist_m, dur_s
        except Exception:
            continue
    return None


def _approx_path_metrics(ordered_lonlat: np.ndarray):
    if len(ordered_lonlat) < 2:
        return 0.0, 0.0
    dur_seg, dist_seg = _approx_block_matrices(ordered_lonlat[:-1], ordered_lonlat[1:])
    return float(np.trace(dist_seg)), float(np.trace(dur_seg))


def build_route_geometries(route_sequences: dict, close_loop: bool = False):
    """
    Route OSRM geometry/metrics for already-ordered stop sequences.
    route_sequences : {vehicle: lonlat_array_in_exact_visit_order}
    Returns (geoms, route_dist, route_dur, all_osrm_exact).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_one(args):
        v, ordered_ll = args
        ordered_ll = np.asarray(ordered_ll, dtype=float)
        if len(ordered_ll) < 2:
            return v, None, 0.0, 0.0, True
        if close_loop and not np.allclose(ordered_ll[0], ordered_ll[-1]):
            ordered_ll = np.vstack([ordered_ll, ordered_ll[:1]])
        result = ordered_route_geometry(ordered_ll)
        if result:
            geom, dist_m, dur_s = result
            return v, geom, dist_m, dur_s, True
        dist_m, dur_s = _approx_path_metrics(ordered_ll)
        return v, None, dist_m, dur_s, False

    tasks = [(v, seq) for v, seq in route_sequences.items() if len(seq) >= 1]
    geoms, route_dist, route_dur = {}, {}, {}
    all_osrm_exact = True
    with ThreadPoolExecutor(max_workers=max(1, min(8, len(tasks)))) as pool:
        futures = {pool.submit(_fetch_one, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            v, geom, dist_m, dur_s, exact = fut.result()
            route_dist[v] = dist_m
            route_dur[v] = dur_s
            all_osrm_exact &= exact
            if geom:
                geoms[v] = geom
    return geoms, route_dist, route_dur, all_osrm_exact


# ──────────────────────────────────────────────────────
#  OSM FETCH
# ──────────────────────────────────────────────────────
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

def fetch_osm(bbox, filters, max_pts):
    lat0,lon0,lat1,lon1 = bbox[1],bbox[0],bbox[3],bbox[2]
    bb = f"{lat0},{lon0},{lat1},{lon1}"
    nodes = "\n  ".join(f"{f}({bb});" for f in filters)
    q = f"[out:json][timeout:30];\n(\n  {nodes}\n);\nout body {max_pts};"
    for ep in OVERPASS_ENDPOINTS:
        try:
            r = overpy.Overpass(url=ep).query(q)
            if r.nodes:
                pts = np.array([[float(n.lon),float(n.lat)] for n in r.nodes])
                tags = [n.tags for n in r.nodes]
                return pts, tags
        except Exception:
            continue
    return None, []


# ──────────────────────────────────────────────────────
#  EMPTY CLUSTER FIX
# ──────────────────────────────────────────────────────
def fix_empty(pts, L, C, K):
    for _ in range(K * 3):
        sizes = np.bincount(L, minlength=K)
        empty = np.where(sizes == 0)[0]
        if not len(empty): break
        tk = empty[0]; dk = int(sizes.argmax())
        di = np.where(L == dk)[0]
        steal = di[int(np.linalg.norm(pts[di]-C[dk], axis=1).argmax())]
        L[steal] = tk; C[tk] = pts[steal].copy()
        C[dk] = pts[L==dk].mean(0)
    return L, C


def greedy_capacitated_assignment(cost_to_anchor: np.ndarray, cap: int) -> np.ndarray:
    """
    Greedy capacity-respecting fallback assignment.
    Every stop is assigned exactly once unless the instance is infeasible.
    """
    n, K = cost_to_anchor.shape
    loads = np.zeros(K, dtype=int)
    labels = -np.ones(n, dtype=int)
    order = np.argsort(cost_to_anchor.min(axis=1))
    for i in order:
        for k in np.argsort(cost_to_anchor[i]):
            if loads[k] < cap:
                labels[i] = int(k)
                loads[k] += 1
                break
    if (labels < 0).any():
        raise ValueError("Capacitated assignment failed: not enough vehicle capacity.")
    return labels


def _shared_seeded_starts(pts_ll: np.ndarray, K: int) -> np.ndarray:
    """
    Shared vehicle start seeds used by all pipelines when seeded starts are selected.
    These are independent of any pipeline-specific assignment to keep the benchmark fair.
    """
    bbox = (pts_ll[:, 0].min(), pts_ll[:, 1].min(),
            pts_ll[:, 0].max(), pts_ll[:, 1].max())
    pts_n = norm(pts_ll, bbox)
    centers_n = KMeans(K, init="k-means++", n_init=8, max_iter=100,
                       random_state=0).fit(pts_n).cluster_centers_
    order = np.lexsort((centers_n[:, 1], centers_n[:, 0]))
    return denorm(centers_n[order], bbox)


def build_shared_route_model(pts_ll: np.ndarray, K: int, cap_pct: int,
                             cost_matrix: np.ndarray, start_policy: str,
                             end_policy: str, demand_per_stop: float = 1.0):
    """
    Shared operational route model used by every pipeline.
    Starts/ends live here so A/B/C/D are evaluated under identical assumptions.
    """
    n = len(pts_ll)
    cap_stops = int(np.ceil(n / K * cap_pct / 100))
    shared_depot_ll = pts_ll.mean(axis=0)

    if start_policy == "centroid":
        starts_ll = np.repeat(shared_depot_ll[np.newaxis, :], K, axis=0)
    elif start_policy == "seeded":
        starts_ll = _shared_seeded_starts(pts_ll, K)
    elif start_policy == "random":
        bbox = (pts_ll[:, 0].min(), pts_ll[:, 1].min(),
                pts_ll[:, 0].max(), pts_ll[:, 1].max())
        rng = np.random.default_rng(42)
        starts_ll = np.column_stack([
            rng.uniform(bbox[0], bbox[2], K),
            rng.uniform(bbox[1], bbox[3], K),
        ])
    else:
        raise ValueError(f"Unknown start_policy: {start_policy}")

    if end_policy == "return_to_start":
        ends_ll = starts_ll.copy()
    elif end_policy == "return_to_depot":
        ends_ll = np.repeat(shared_depot_ll[np.newaxis, :], K, axis=0)
    elif end_policy == "open":
        ends_ll = None
    else:
        raise ValueError(f"Unknown end_policy: {end_policy}")

    start_to_stop_dur, _, start_exact = osrm_rectangular_matrix(starts_ll, pts_ll)
    if ends_ll is not None:
        stop_to_end_dur, _, end_exact = osrm_rectangular_matrix(pts_ll, ends_ll)
    else:
        stop_to_end_dur = np.zeros((n, K), dtype=float)
        end_exact = True

    median_stop = float(np.nanmedian(cost_matrix[np.isfinite(cost_matrix)])) if np.isfinite(cost_matrix).any() else 0.0
    median_start = float(np.nanmedian(start_to_stop_dur[np.isfinite(start_to_stop_dur)])) if np.isfinite(start_to_stop_dur).any() else 0.0
    scale_ratio = (max(median_stop, median_start) / max(min(median_stop, median_start), 1e-9)
                   if median_stop > 0 and median_start > 0 else 1.0)

    return {
        "n": n,
        "K": K,
        "points_ll": pts_ll,
        "cost_matrix": cost_matrix,
        "cap_pct": int(cap_pct),
        "start_policy": start_policy,
        "end_policy": end_policy,
        "roundtrip": bool(ends_ll is not None and np.allclose(starts_ll, ends_ll)),
        "starts_ll": starts_ll,
        "ends_ll": ends_ll,
        "shared_depot_ll": shared_depot_ll,
        "start_to_stop_dur": start_to_stop_dur,
        "stop_to_end_dur": stop_to_end_dur,
        "cap_stops": cap_stops,
        "demand_per_stop": float(demand_per_stop),
        "capacity_units": float(cap_stops * demand_per_stop),
        "demands": np.full(n, float(demand_per_stop)),
        "routing_cost_basis": "OSRM duration with explicit start/end legs",
        "terminal_costs_exact": bool(start_exact and end_exact),
        "scale_ratio": float(scale_ratio),
    }


def realign_labels_to_shared_starts(labels: np.ndarray, centers: np.ndarray,
                                    route_model: dict, meta: dict | None = None):
    """
    Relabel staged-pipeline territories so van k always refers to shared start k.
    Without this, A/B/C can sequence the right territory with the wrong van origin.
    """
    K = route_model["K"]
    relabel_cost = np.zeros((K, K), dtype=float)

    for old_k in range(K):
        idx = np.where(labels == old_k)[0]
        if len(idx) == 0:
            relabel_cost[old_k] = 0.0
            continue
        access = route_model["start_to_stop_dur"][:, idx].mean(axis=1)
        if route_model["end_policy"] != "open":
            egress = route_model["stop_to_end_dur"][idx, :].mean(axis=0)
        else:
            egress = np.zeros(K, dtype=float)
        relabel_cost[old_k] = access + egress

    old_ids, new_ids = linear_sum_assignment(relabel_cost)
    mapping = {int(old): int(new) for old, new in zip(old_ids, new_ids)}

    new_labels = np.array([mapping[int(k)] for k in labels], dtype=int)
    new_centers = np.zeros_like(centers)
    for old_k, new_k in mapping.items():
        if old_k < len(centers):
            new_centers[new_k] = centers[old_k]

    new_meta = None
    if meta is not None:
        new_meta = dict(meta)
        if "assignment_cost_per_vehicle" in meta:
            new_meta["assignment_cost_per_vehicle"] = {
                mapping.get(int(k), int(k)): float(v)
                for k, v in meta["assignment_cost_per_vehicle"].items()
            }
        for key in ("seed_indices", "median_indices"):
            if key in meta and len(meta[key]) == K:
                remapped = [None] * K
                for old_k, value in enumerate(meta[key]):
                    remapped[mapping.get(old_k, old_k)] = value
                new_meta[key] = remapped
        new_meta["cluster_start_mapping"] = mapping

    return new_labels, new_centers, new_meta, mapping


def compute_vehicle_path_cost(order: list[int], vehicle: int, route_model: dict) -> float:
    if len(order) == 0:
        return 0.0
    cost = float(route_model["start_to_stop_dur"][vehicle, order[0]])
    for i in range(len(order) - 1):
        cost += float(route_model["cost_matrix"][order[i], order[i + 1]])
    if route_model["end_policy"] != "open":
        cost += float(route_model["stop_to_end_dur"][order[-1], vehicle])
    return cost


def _two_opt_path(order: list[int], vehicle: int, route_model: dict) -> list[int]:
    if len(order) < 4:
        return order
    best = order[:]
    best_cost = compute_vehicle_path_cost(best, vehicle, route_model)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best) + 1):
                cand = best[:i] + best[i:j][::-1] + best[j:]
                cand_cost = compute_vehicle_path_cost(cand, vehicle, route_model)
                if cand_cost + 1e-9 < best_cost:
                    best = cand
                    best_cost = cand_cost
                    improved = True
                    break
            if improved:
                break
    return best


def heuristic_sequence_route(stop_indices: list[int], vehicle: int, route_model: dict):
    """
    A/B sequencing: deterministic path heuristic on the OSRM duration matrix.
    This uses the same starts/ends as the rest of the benchmark without relying on OSRM /trip.
    """
    remaining = list(map(int, stop_indices))
    if not remaining:
        return [], 0.0

    first = min(
        remaining,
        key=lambda s: route_model["start_to_stop_dur"][vehicle, s] +
        (0.0 if route_model["end_policy"] == "open"
         else route_model["stop_to_end_dur"][s, vehicle])
    )
    route = [int(first)]
    remaining.remove(first)

    while remaining:
        best_stop, best_pos, best_cost = None, None, float("inf")
        for stop in remaining:
            for pos in range(len(route) + 1):
                cand = route[:pos] + [int(stop)] + route[pos:]
                cand_cost = compute_vehicle_path_cost(cand, vehicle, route_model)
                if cand_cost < best_cost:
                    best_stop, best_pos, best_cost = int(stop), pos, cand_cost
        route.insert(best_pos, best_stop)
        remaining.remove(best_stop)

    route = _two_opt_path(route, vehicle, route_model)
    return route, compute_vehicle_path_cost(route, vehicle, route_model)


# ──────────────────────────────────────────────────────
#  PIPELINE A — Min-Cost Flow
# ──────────────────────────────────────────────────────
def run_mcf(pts_ll, K, cap_pct, cost_matrix, return_meta=False):
    """pts_ll = lon/lat array, cost_matrix = NxN (OSRM seconds or Euclidean)"""
    n = len(pts_ll)
    cap = int(np.ceil(n / K * cap_pct / 100))
    # seed centers with KMeans on normalized coords
    pts_n = norm(pts_ll, (pts_ll[:,0].min(), pts_ll[:,1].min(),
                           pts_ll[:,0].max(), pts_ll[:,1].max()))
    seeds_n = KMeans(K, init="k-means++", n_init=8, max_iter=100,
                     random_state=0).fit(pts_n).cluster_centers_
    # Find which real point is closest to each KMeans centroid → seed representative.
    # Ensure all K seeds map to DISTINCT points (avoid duplicate columns in cost matrix).
    seed_idx = []
    used = set()
    for s in seeds_n:
        dists = np.linalg.norm(pts_n - s, axis=1)
        order = np.argsort(dists)
        for candidate in order:
            if int(candidate) not in used:
                seed_idx.append(int(candidate))
                used.add(int(candidate))
                break

    # C[i, v] = cost from stop i to van v's seed representative.
    # This approximates the assignment cost — flow minimises total assignment cost
    # subject to capacity. Not exact global optimum but strongly guided by
    # the OSRM cost matrix (road distances / durations).
    C = cost_matrix[:, seed_idx]   # (n, K)

    G = nx.DiGraph()
    G.add_node("S", demand=-n); G.add_node("T", demand=n)
    for i in range(n):   G.add_edge("S", f"p{i}", capacity=1, weight=0)
    for v in range(K):   G.add_edge(f"v{v}", "T", capacity=cap, weight=0)
    for i in range(n):
        for v in range(K):
            G.add_edge(f"p{i}", f"v{v}", capacity=1, weight=int(C[i,v]*10))
    L = np.zeros(n, dtype=int)
    try:
        flow = nx.min_cost_flow(G)
        for i in range(n):
            for v in range(K):
                if flow.get(f"p{i}",{}).get(f"v{v}",0) > 0:
                    L[i] = v; break
    except Exception:
        L = greedy_capacitated_assignment(C, cap)

    # Compute centers as centroids of assigned points (in normalized space)
    Cn = np.array([pts_n[L==k].mean(0) if (L==k).any() else pts_n.mean(0)
                   for k in range(K)])
    L, Cn = fix_empty(pts_n, L, Cn, K)
    if not return_meta:
        return L, Cn

    per_vehicle = {k: float(C[L == k, k].sum()) for k in range(K)}
    meta = {
        "assignment_cost_total": float(sum(per_vehicle.values())),
        "assignment_cost_per_vehicle": per_vehicle,
        "assignment_cost_basis": "OSRM stop-to-seed duration",
        "capacity_enforced": True,
        "seed_indices": seed_idx,
    }
    return L, Cn, meta


# ──────────────────────────────────────────────────────
#  PIPELINE B — Capacitated P-Median
# ──────────────────────────────────────────────────────
def run_pmedian(pts_ll, K, cap_pct, max_iter, cost_matrix, return_meta=False):
    n = len(pts_ll)
    cap = int(np.ceil(n / K * cap_pct / 100))
    pts_n = norm(pts_ll, (pts_ll[:,0].min(), pts_ll[:,1].min(),
                           pts_ll[:,0].max(), pts_ll[:,1].max()))
    # Initialize medians with KMeans
    km_labels = KMeans(K, init="k-means++", n_init=8, max_iter=100,
                       random_state=0).fit(pts_n).labels_
    L = km_labels.copy()

    median_idx = [0] * K
    for _ in range(max_iter):
        # Use OSRM cost matrix for assignment distances
        nL = -np.ones(n, dtype=int); ld = np.zeros(K, dtype=int)
        # For each cluster find current median index (min sum of distances)
        median_idx = []
        for k in range(K):
            m = L==k
            if not m.any():
                median_idx.append(0)
                continue
            idx_k = np.where(m)[0]
            sub = cost_matrix[np.ix_(idx_k, idx_k)]
            median_idx.append(idx_k[int(sub.sum(1).argmin())])

        # Assign each point to nearest median respecting capacity
        D_to_medians = cost_matrix[:, median_idx]  # (n, K)
        for i in np.argsort(D_to_medians.min(1)):
            for k in np.argsort(D_to_medians[i]):
                if ld[k] < cap: nL[i]=k; ld[k]+=1; break
            if nL[i] < 0:
                raise ValueError("P-Median assignment exceeded vehicle capacity.")

        if np.all(nL == L): break
        L = nL

    # Compute centers in normalized space
    Cn = np.array([pts_n[L==k].mean(0) if (L==k).any() else pts_n.mean(0)
                   for k in range(K)])
    L, Cn = fix_empty(pts_n, L, Cn, K)
    if not return_meta:
        return L, Cn

    final_median_idx = []
    per_vehicle = {}
    for k in range(K):
        idx_k = np.where(L == k)[0]
        if len(idx_k):
            sub = cost_matrix[np.ix_(idx_k, idx_k)]
            med = int(idx_k[int(sub.sum(axis=1).argmin())])
            final_median_idx.append(med)
            per_vehicle[k] = float(cost_matrix[np.ix_(idx_k, [med])].sum())
        else:
            final_median_idx.append(0)
            per_vehicle[k] = 0.0
    meta = {
        "assignment_cost_total": float(sum(per_vehicle.values())),
        "assignment_cost_per_vehicle": per_vehicle,
        "assignment_cost_basis": "OSRM stop-to-median duration",
        "capacity_enforced": True,
        "median_indices": final_median_idx,
    }
    return L, Cn, meta





# (old analytics removed — see analytics() below with full metrics)


# ──────────────────────────────────────────────────────
#  STATIC MAP — draws OSRM geometries OR straight lines
# ──────────────────────────────────────────────────────
def make_map(ll_pts, L, ll_ctr_n, bbox, title, road_geoms=None, tile_src=None,
             route_model=None):
    """
    road_geoms: dict {k: [(x_wm, y_wm), ...]} — actual street polylines.
    If None or missing for a cluster, falls back to straight lines.
    """
    ll_ctr = denorm(ll_ctr_n, bbox) if ll_ctr_n.shape[1] == 2 else ll_ctr_n
    # recompute proper lonlat centers from labels
    bbox_pts = (ll_pts[:,0].min(), ll_pts[:,1].min(),
                ll_pts[:,0].max(), ll_pts[:,1].max())
    pts_n = norm(ll_pts, bbox_pts)

    K = max(int(L.max())+1, len(ll_ctr))
    colors = [PALETTE[k%len(PALETTE)] for k in range(K)]
    wm = to_wm(ll_pts); cw = to_wm(ll_ctr)

    fig, ax = plt.subplots(figsize=(8,7))
    fig.patch.set_facecolor("#0a0f1e"); ax.set_facecolor("#0d1526")
    xp = (wm[:,0].max()-wm[:,0].min())*0.12 or 500
    yp = (wm[:,1].max()-wm[:,1].min())*0.12 or 500
    ax.set_xlim(wm[:,0].min()-xp, wm[:,0].max()+xp)
    ax.set_ylim(wm[:,1].min()-yp, wm[:,1].max()+yp)

    for src in [tile_src, ctx.providers.CartoDB.DarkMatter]:
        if src is None: continue
        try: ctx.add_basemap(ax,crs="EPSG:3857",source=src,attribution=False,zoom="auto"); break
        except Exception: continue

    # Draw routes — OSRM street geometry if available, else straight lines
    for k in range(K):
        m = L==k
        if not m.any(): continue
        if road_geoms and k in road_geoms and road_geoms[k]:
            geom = np.array(road_geoms[k])
            ax.plot(geom[:,0], geom[:,1], "-", color=colors[k],
                    alpha=0.75, lw=2.2, zorder=3, solid_capstyle="round")
        # always draw stop dots on top of road lines
        ax.scatter(wm[m,0], wm[m,1], s=22, color=colors[k],
                   alpha=0.9, edgecolors="none", zorder=5)

    # halos
    for k in range(K):
        m = L==k
        if m.any() and k<len(cw):
            r = np.linalg.norm(wm[m]-cw[k], axis=1).max()*0.55
            ax.add_patch(plt.Circle(cw[k], r, color=colors[k], alpha=0.07, zorder=1))

    # centers
    for k,c in enumerate(cw):
        ax.scatter(c[0],c[1], s=160, color=colors[k], edgecolors="white",
                   linewidths=1.8, zorder=7, marker="*")
        ax.annotate(str(k+1), xy=(c[0],c[1]), xytext=(0,12),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color="white", fontweight="bold", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.2", fc=colors[k], alpha=0.75, ec="none"),
                    zorder=8)

    # configured starts / ends so the map matches the actual route model
    if route_model is not None:
        start_wm = to_wm(route_model["starts_ll"])
        for k, s in enumerate(start_wm):
            ax.scatter(s[0], s[1], s=85, color=colors[k], edgecolors="white",
                       linewidths=1.4, zorder=9, marker="D", alpha=0.95)
        if route_model["end_policy"] != "open":
            end_wm = to_wm(route_model["ends_ll"])
            for k, e in enumerate(end_wm):
                if np.allclose(route_model["starts_ll"][k], route_model["ends_ll"][k]):
                    continue
                ax.scatter(e[0], e[1], s=65, color=colors[k], edgecolors="white",
                           linewidths=1.1, zorder=9, marker="s", alpha=0.9)

    # OSRM badge
    has_roads = road_geoms and any(road_geoms.values())
    badge = "🛣 Real road routes (OSRM)" if has_roads else "📐 Euclidean lines (OSRM unavailable)"
    ax.text(0.01, 0.01, badge, transform=ax.transAxes, fontsize=6.5,
            color="#6ee7b7" if has_roads else "#f87171",
            fontfamily="monospace", va="bottom", zorder=12,
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="#064e3b" if has_roads else "#7f1d1d", alpha=0.8, ec="none"))
    ax.text(0.01, 0.06, "halos = assignment overlays, not exact optimized boundaries",
            transform=ax.transAxes, fontsize=6.2, color="#94a3b8",
            fontfamily="monospace", va="bottom", zorder=12,
            bbox=dict(boxstyle="round,pad=0.25", fc="#0f172a", alpha=0.75, ec="none"))

    ax.set_axis_off()
    ax.set_title(title, fontsize=10, color="#94a3b8",
                 fontfamily="monospace", pad=8, fontweight="600")
    fig.tight_layout(pad=0.4)
    return fig


# ──────────────────────────────────────────────────────
#  ANIMATION — vehicles follow OSRM street geometry
# ──────────────────────────────────────────────────────
def _param(path):
    d = np.linalg.norm(np.diff(path, axis=0), axis=1)
    c = np.concatenate([[0], np.cumsum(d)]); t = c[-1]
    return (c/t if t>0 else c), path


def make_gif(ll_pts, L, ll_ctr_n, bbox, road_geoms,
             tile_src, route_model=None, n_frames=140, fps=20, trail=18):
    ll_ctr = denorm(ll_ctr_n, bbox)
    K = max(int(L.max())+1, len(ll_ctr))
    colors = [PALETTE[k%len(PALETTE)] for k in range(K)]
    wm = to_wm(ll_pts); cw = to_wm(ll_ctr)
    dep = cw.mean(0)
    start_wm = to_wm(route_model["starts_ll"]) if route_model is not None else None
    end_wm = (to_wm(route_model["ends_ll"])
              if route_model is not None and route_model["end_policy"] != "open"
              else None)

    paths = []
    for k in range(K):
        if road_geoms and k in road_geoms and road_geoms[k]:
            # use the actual OSRM street geometry as the vehicle path
            path = np.array(road_geoms[k])
        else:
            # fallback: straight lines under the same configured route model
            m = L == k
            start_pt = dep if start_wm is None else start_wm[k]
            if not m.any():
                end_pt = start_pt if end_wm is None else end_wm[k]
                paths.append(_param(np.vstack([start_pt, end_pt])))
                continue
            seq = [start_pt[np.newaxis], wm[m]]
            if end_wm is not None:
                seq.append(end_wm[k][np.newaxis])
            paths.append(_param(np.vstack(seq)))
            continue
        paths.append(_param(path))

    fig, ax = plt.subplots(figsize=(7,6.2))
    fig.patch.set_facecolor("#0a0f1e"); ax.set_facecolor("#0d1526")
    xp=(wm[:,0].max()-wm[:,0].min())*0.13 or 600
    yp=(wm[:,1].max()-wm[:,1].min())*0.13 or 600
    ax.set_xlim(wm[:,0].min()-xp, wm[:,0].max()+xp)
    ax.set_ylim(wm[:,1].min()-yp, wm[:,1].max()+yp)

    for src in [tile_src, ctx.providers.CartoDB.DarkMatter]:
        if src is None: continue
        try: ctx.add_basemap(ax,crs="EPSG:3857",source=src,attribution=False,zoom="auto"); break
        except Exception: continue

    for k in range(K):
        m = L==k
        if m.any():
            ax.scatter(wm[m,0],wm[m,1], s=14, color=colors[k], alpha=0.35, edgecolors="none", zorder=3)
            r = np.linalg.norm(wm[m]-cw[k],axis=1).max()*0.55
            ax.add_patch(plt.Circle(cw[k], r, color=colors[k], alpha=0.06, zorder=2))

    if route_model is None:
        ax.scatter(dep[0],dep[1], s=220, color="#ef4444", marker="D",
                   edgecolors="white", linewidths=2, zorder=5)
    else:
        for k, s in enumerate(start_wm):
            ax.scatter(s[0], s[1], s=110, color=colors[k], marker="D",
                       edgecolors="white", linewidths=1.5, zorder=5)
        if end_wm is not None:
            for k, e in enumerate(end_wm):
                if np.allclose(route_model["starts_ll"][k], route_model["ends_ll"][k]):
                    continue
                ax.scatter(e[0], e[1], s=90, color=colors[k], marker="s",
                           edgecolors="white", linewidths=1.2, zorder=5)

    dots   = [ax.plot([],[], "o", ms=13, color=c, markeredgecolor="white",
                       markeredgewidth=1.4, zorder=10)[0] for c in colors]
    trails = [ax.plot([],[], "-", color=c, alpha=0.6, lw=2.5, zorder=9)[0] for c in colors]
    status = ax.text(0.02,0.97,"", transform=ax.transAxes, fontsize=7,
                     color="#64748b", fontfamily="monospace", va="top", zorder=11)

    hist = [[] for _ in range(K)]

    def frame(fi):
        t = fi / n_frames
        arts = []
        for k,(cum,path) in enumerate(paths):
            tl = t % 1.0
            idx = np.clip(np.searchsorted(cum,tl,side="right")-1, 0, len(path)-2)
            frac = (tl-cum[idx]) / max(cum[idx+1]-cum[idx], 1e-9)
            pos = path[idx] + np.clip(frac,0,1)*(path[idx+1]-path[idx])
            dots[k].set_data([pos[0]],[pos[1]])
            hist[k].append(pos.copy())
            if len(hist[k]) > trail: hist[k].pop(0)
            if len(hist[k]) > 1:
                h = np.array(hist[k]); trails[k].set_data(h[:,0],h[:,1])
            arts += [dots[k], trails[k]]
        status.set_text(f"DISPATCH  {int((t%1)*100):3d}%")
        arts.append(status)
        return arts

    ani = FuncAnimation(fig, frame, frames=n_frames,
                        interval=int(1000/fps), blit=True)
    tmp = tempfile.mktemp(suffix=".gif")
    ani.save(tmp, writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)
    return tmp


# ──────────────────────────────────────────────────────
#  CHARTS
# ──────────────────────────────────────────────────────
def size_chart(sizes):
    colors = [PALETTE[k%len(PALETTE)] for k in range(len(sizes))]
    fig, ax = plt.subplots(figsize=(5,2.8))
    fig.patch.set_facecolor("#0a0f1e"); ax.set_facecolor("#0d1526")
    ax.bar(range(1,len(sizes)+1), sizes, color=colors, edgecolor="#0a0f1e")
    mv = np.mean(sizes)
    ax.axhline(mv, color="#f59e0b", ls="--", lw=1.4, label=f"mean={mv:.1f}", alpha=0.9)
    ax.set_xlabel("Van", fontsize=7, color="#64748b", fontfamily="monospace")
    ax.set_ylabel("Stops", fontsize=7, color="#64748b", fontfamily="monospace")
    ax.set_title("Stops per Van", fontsize=8, color="#94a3b8", fontfamily="monospace")
    ax.tick_params(colors="#475569", labelsize=6)
    ax.legend(fontsize=6, labelcolor="#94a3b8", facecolor="#1e293b", edgecolor="#334155")
    ax.grid(axis="y", color="#1e293b", lw=0.6, alpha=0.7)
    for s in ax.spines.values(): s.set_edgecolor("#1e293b")
    fig.tight_layout(pad=0.4); return fig



# ──────────────────────────────────────────────────────
#  PIPELINE C — MCF + OR-Tools per-cluster path sequencing
#  Same assignment as A, but with a stronger local route solver.
# ──────────────────────────────────────────────────────
def solve_single_vehicle_path(stop_indices: list[int], vehicle: int,
                              route_model: dict, time_limit_s: int = 4):
    """
    OR-Tools single-vehicle path under the shared route model.
    Used by Pipeline C so its route model exactly matches A/B/D.
    """
    stop_indices = list(map(int, stop_indices))
    if len(stop_indices) <= 1:
        return stop_indices, compute_vehicle_path_cost(stop_indices, vehicle, route_model)

    sub_cost = route_model["cost_matrix"][np.ix_(stop_indices, stop_indices)]
    start_to_stop = route_model["start_to_stop_dur"][vehicle:vehicle + 1, stop_indices]
    stop_to_end = None
    if route_model["end_policy"] != "open":
        stop_to_end = route_model["stop_to_end_dur"][stop_indices, vehicle][:, np.newaxis]

    full, starts, ends = _build_ortools_data_model(
        sub_cost, start_to_stop, stop_to_end, 1, route_model["cap_stops"],
        route_model["end_policy"])

    mgr = pywrapcp.RoutingIndexManager(len(full), 1, starts, ends)
    mdl = pywrapcp.RoutingModel(mgr)
    cb = mdl.RegisterTransitCallback(
        lambda i, j: int(full[mgr.IndexToNode(i)][mgr.IndexToNode(j)]))
    mdl.SetArcCostEvaluatorOfAllVehicles(cb)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_s

    sol = mdl.SolveWithParameters(params)
    if not sol:
        order, _ = heuristic_sequence_route(stop_indices, vehicle, route_model)
        return order, compute_vehicle_path_cost(order, vehicle, route_model)

    local_order = []
    idx = mdl.Start(0)
    while not mdl.IsEnd(idx):
        node = mgr.IndexToNode(idx)
        if node < len(stop_indices):
            local_order.append(node)
        idx = sol.Value(mdl.NextVar(idx))

    order = [stop_indices[i] for i in local_order]
    return order, compute_vehicle_path_cost(order, vehicle, route_model)


def run_pipeline_c(labels, centers, route_model, time_limit_s=8):
    """
    Pipeline C: same assignment as A, but sequenced with OR-Tools path routing.
    """
    ordered_global, per_vehicle_cost = {}, {}
    per_cluster = max(2, time_limit_s // max(route_model["K"], 1))
    for k in range(route_model["K"]):
        stop_indices = list(np.where(labels == k)[0])
        order, cost = solve_single_vehicle_path(
            stop_indices, k, route_model, time_limit_s=per_cluster)
        ordered_global[k] = order
        per_vehicle_cost[k] = cost
    return labels.copy(), centers.copy(), ordered_global, per_vehicle_cost


# ──────────────────────────────────────────────────────
#  PIPELINE D — Full OR-Tools CVRP
#  Single-stage: assignment + routing solved simultaneously.
#  Supports explicit start nodes and explicit end-mode choices.
# ──────────────────────────────────────────────────────

def _build_ortools_data_model(customer_cost, start_to_stop, stop_to_end, K, cap,
                              end_mode):
    """
    Build the augmented OR-Tools graph for explicit starts and explicit end mode.

    Node layout:
      0 .. n-1        = customer stops
      n .. n+K-1      = per-vehicle start nodes
      n+K .. n+2K-1   = per-vehicle end nodes when needed

    end_mode:
      "return_to_start"  -> end at the same start node
      "return_to_depot"  -> end at a shared depot location
      "open"             -> dummy zero-cost end node
    """
    del cap  # capacity is enforced by OR-Tools dimensions, not by the matrix itself

    n = customer_cost.shape[0]
    scale = 100_000
    cust = np.round(customer_cost * scale).astype(int)
    start_block = np.round(start_to_stop * scale).astype(int)
    end_block = None if stop_to_end is None else np.round(stop_to_end * scale).astype(int)

    max_cost = max(
        int(np.max(cust)) if cust.size else 1,
        int(np.max(start_block)) if start_block.size else 1,
        int(np.max(end_block)) if end_block is not None and end_block.size else 1,
    )
    big_m = max(1_000_000, max_cost * 100 + 1)
    start_nodes = list(range(n, n + K))

    if end_mode == "return_to_start":
        full = np.full((n + K, n + K), big_m, dtype=int)
        full[:n, :n] = cust
        for v, s_node in enumerate(start_nodes):
            full[s_node, :n] = start_block[v]
            full[:n, s_node] = end_block[:, v]
            full[s_node, s_node] = 0
        return full, start_nodes, start_nodes[:]

    if end_mode not in ("return_to_depot", "open"):
        raise ValueError(f"Unknown end_mode: {end_mode}")

    end_nodes = list(range(n + K, n + 2 * K))
    full = np.full((n + 2 * K, n + 2 * K), big_m, dtype=int)
    full[:n, :n] = cust

    for v, s_node in enumerate(start_nodes):
        full[s_node, :n] = start_block[v]
        full[s_node, end_nodes[v]] = 0
        full[s_node, s_node] = 0

    for v, e_node in enumerate(end_nodes):
        full[e_node, e_node] = 0
        full[:n, e_node] = 0 if end_mode == "open" else end_block[:, v]

    return full, start_nodes, end_nodes


def _build_route_sequence(start_ll, stop_nodes, pts_ll, end_ll=None):
    if len(stop_nodes) == 0:
        if end_ll is None or np.allclose(start_ll, end_ll):
            return np.array([start_ll], dtype=float)
        return np.vstack([start_ll, end_ll]).astype(float)

    seq = [np.asarray(start_ll, dtype=float)[np.newaxis, :], pts_ll[stop_nodes]]
    if end_ll is not None:
        seq.append(np.asarray(end_ll, dtype=float)[np.newaxis, :])
    return np.vstack(seq).astype(float)


def run_pipeline_d(pts_ll, route_model, time_limit_s=15):
    """
    Pipeline D: Full OR-Tools CVRP with the shared route model.
    OR-Tools decides assignment and visit order simultaneously.

    Returns (labels, centers, ordered_global_idx, route_lonlat_by_vehicle, meta).
      labels                 : (n,) van index per stop
      centers                : (K,2) normalized customer centroids
      ordered_global_idx     : {v: [customer indices in solved visit order]}
      route_lonlat_by_vehicle: {v: lon/lat sequence including start/end when used}
      meta                   : route/start-end metadata for display and diagnostics
    """
    n = len(pts_ll)
    K = route_model["K"]
    cap = route_model["cap_stops"]
    pts_bbox = (pts_ll[:, 0].min(), pts_ll[:, 1].min(),
                pts_ll[:, 0].max(), pts_ll[:, 1].max())
    pts_n = norm(pts_ll, pts_bbox)
    starts_ll = route_model["starts_ll"]
    ends_ll = route_model["ends_ll"]
    stop_to_end = None if route_model["end_policy"] == "open" else route_model["stop_to_end_dur"]

    full_mat, starts, ends = _build_ortools_data_model(
        route_model["cost_matrix"], route_model["start_to_stop_dur"],
        stop_to_end, K, cap, route_model["end_policy"])

    mgr = pywrapcp.RoutingIndexManager(len(full_mat), K, starts, ends)
    mdl = pywrapcp.RoutingModel(mgr)

    cb = mdl.RegisterTransitCallback(
        lambda i, j: int(full_mat[mgr.IndexToNode(i)][mgr.IndexToNode(j)]))
    mdl.SetArcCostEvaluatorOfAllVehicles(cb)

    dcb = mdl.RegisterUnaryTransitCallback(lambda i: 1 if mgr.IndexToNode(i) < n else 0)
    mdl.AddDimensionWithVehicleCapacity(dcb, 0, [cap] * K, True, "Capacity")
    cap_dim = mdl.GetDimensionOrDie("Capacity")
    for v in range(K):
        mdl.solver().Add(cap_dim.CumulVar(mdl.End(v)) >= 1)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_s

    sol = mdl.SolveWithParameters(params)

    labels = np.zeros(n, dtype=int)
    centers = np.zeros((K, 2))
    ordered_global = {}
    route_lonlat = {}

    if sol:
        for v in range(K):
            idx = mdl.Start(v)
            stop_nodes = []
            while not mdl.IsEnd(idx):
                node = mgr.IndexToNode(idx)
                if node < n:
                    stop_nodes.append(node)
                    labels[node] = v
                idx = sol.Value(mdl.NextVar(idx))
            ordered_global[v] = stop_nodes
            route_lonlat[v] = _build_route_sequence(
                starts_ll[v], stop_nodes, pts_ll,
                None if ends_ll is None else ends_ll[v])
            if stop_nodes:
                centers[v] = pts_n[stop_nodes].mean(0)
    else:
        labels, centers, _ = run_mcf(
            pts_ll, K, route_model["cap_pct"], route_model["cost_matrix"],
            return_meta=True)
        for v in range(K):
            stop_nodes = list(np.where(labels == v)[0])
            ordered_global[v] = stop_nodes
            route_lonlat[v] = _build_route_sequence(
                starts_ll[v], stop_nodes, pts_ll,
                None if ends_ll is None else ends_ll[v])

    meta = {
        "start_policy": route_model["start_policy"],
        "end_policy": route_model["end_policy"],
        "start_lonlat": starts_ll,
        "end_lonlat": ends_ll,
        "shared_depot_lonlat": route_model["shared_depot_ll"],
        "start_end_osrm_exact": bool(route_model["terminal_costs_exact"]),
    }
    return labels, centers, ordered_global, route_lonlat, meta


def build_route_sequences_from_orders(pts_ll, ordered_global, route_model):
    route_lonlat = {}
    for v in range(route_model["K"]):
        stop_nodes = ordered_global.get(v, [])
        route_lonlat[v] = _build_route_sequence(
            route_model["starts_ll"][v],
            stop_nodes,
            pts_ll,
            None if route_model["end_policy"] == "open" else route_model["ends_ll"][v],
        )
    return route_lonlat


def _fmt_lonlat(ll):
    if ll is None:
        return "None"
    return f"{float(ll[0]):.5f}, {float(ll[1]):.5f}"


def build_pipeline_summary(assignment_method: str, sequencing_method: str,
                           integration_mode: str, assignment_cost_basis: str,
                           route_model: dict, capacity_treatment: str,
                           intended_use: str, road_network_usage: str | None = None):
    end_label = {
        "open": "open path",
        "return_to_start": "fixed start -> return to start",
        "return_to_depot": "fixed start -> shared depot",
    }[route_model["end_policy"]]
    return {
        "assignment_method": assignment_method,
        "sequencing_method": sequencing_method,
        "integration_mode": integration_mode,
        "assignment_cost_basis": assignment_cost_basis,
        "routing_cost_basis": route_model["routing_cost_basis"],
        "road_network_usage": road_network_usage or (
            "assignment uses OSRM customer matrix; routing uses OSRM route geometry "
            "with the shared start/end model"
        ),
        "route_model": end_label,
        "capacity_treatment": capacity_treatment,
        "intended_use": intended_use,
    }


def build_route_logs(letter, labels, ordered_global, route_dist, route_dur,
                     assignment_cost_per_vehicle, route_model_cost_per_vehicle,
                     route_model, validation):
    rows = []
    val_map = validation.get("vehicle_feasible", {})
    for v in range(route_model["K"]):
        order = ordered_global.get(v, [])
        rows.append({
            "pipeline": letter,
            "van": v + 1,
            "assignment_cost_s": round(float(assignment_cost_per_vehicle.get(v, 0.0)), 2),
            "routing_cost_s": round(float(route_model_cost_per_vehicle.get(v, 0.0)), 2),
            "ordered_stops": ",".join(str(int(i)) for i in order),
            "start_node": _fmt_lonlat(route_model["starts_ll"][v]),
            "end_node": _fmt_lonlat(None if route_model["end_policy"] == "open" else route_model["ends_ll"][v]),
            "total_demand": round(len(order) * route_model["demand_per_stop"], 2),
            "capacity": round(route_model["capacity_units"], 2),
            "road_km": round(float(route_dist.get(v, 0.0)) / 1000.0, 3),
            "drive_min": round(float(route_dur.get(v, 0.0)) / 60.0, 2),
            "open_route": route_model["end_policy"] == "open",
            "feasible": bool(val_map.get(v, True)),
        })
    return rows


def validate_pipeline_result(labels, ordered_global, route_sequences, route_model):
    checks = []
    vehicle_feasible = {}
    n = route_model["n"]
    K = route_model["K"]

    checks.append({
        "check": "customer matrix dimensions",
        "ok": route_model["cost_matrix"].shape == (n, n),
        "detail": f"{route_model['cost_matrix'].shape} vs expected {(n, n)}",
    })
    checks.append({
        "check": "start-stop matrix dimensions",
        "ok": route_model["start_to_stop_dur"].shape == (K, n),
        "detail": f"{route_model['start_to_stop_dur'].shape} vs expected {(K, n)}",
    })
    checks.append({
        "check": "stop-end matrix dimensions",
        "ok": route_model["stop_to_end_dur"].shape == (n, K),
        "detail": f"{route_model['stop_to_end_dur'].shape} vs expected {(n, K)}",
    })
    checks.append({
        "check": "cost scale consistency",
        "ok": route_model["scale_ratio"] < 1000,
        "detail": f"terminal/stop median ratio={route_model['scale_ratio']:.2f}",
    })

    route_seen = []
    for v in range(K):
        assigned = list(np.where(labels == v)[0])
        ordered = list(map(int, ordered_global.get(v, [])))
        seq = np.asarray(route_sequences.get(v, np.empty((0, 2))), dtype=float)
        ok = True
        detail = []

        if len(ordered) != len(set(ordered)):
            ok = False
            detail.append("duplicate stops in order")
        if sorted(ordered) != sorted(assigned):
            ok = False
            detail.append("route order does not match assigned stops")
        if len(seq) > 0 and not np.allclose(seq[0], route_model["starts_ll"][v]):
            ok = False
            detail.append("missing configured start")
        if route_model["end_policy"] == "open":
            if len(seq) >= 2 and np.allclose(seq[0], seq[-1]):
                ok = False
                detail.append("open route closes back to start")
        else:
            if len(seq) == 0 or not np.allclose(seq[-1], route_model["ends_ll"][v]):
                ok = False
                detail.append("missing configured end")
            if route_model["end_policy"] == "return_to_start" and len(seq) >= 2:
                if not np.allclose(seq[0], seq[-1]):
                    ok = False
                    detail.append("expected roundtrip to same start")
        demand_total = len(ordered) * route_model["demand_per_stop"]
        if demand_total > route_model["capacity_units"] + 1e-9:
            ok = False
            detail.append("capacity violation")
        vehicle_feasible[v] = ok
        route_seen.extend(ordered)

    seen_sorted = sorted(route_seen)
    checks.append({
        "check": "all stops assigned exactly once",
        "ok": seen_sorted == list(range(n)),
        "detail": f"seen={len(route_seen)} unique={len(set(route_seen))} expected={n}",
    })
    checks.append({
        "check": "all vehicle routes feasible",
        "ok": all(vehicle_feasible.values()) if vehicle_feasible else False,
        "detail": ", ".join(f"V{v+1}={'ok' if ok else 'fail'}" for v, ok in vehicle_feasible.items()),
    })

    feasible = all(check["ok"] for check in checks)
    return {"feasible": feasible, "checks": checks, "vehicle_feasible": vehicle_feasible}


def build_integrity_row(letter, summary, route_model, validation):
    return {
        "Pipeline": letter,
        "Start Policy": route_model["start_policy"],
        "End Policy": route_model["end_policy"],
        "Roundtrip": bool(route_model["roundtrip"]),
        "Capacity Enforced": True,
        "Assignment Cost Basis": summary["assignment_cost_basis"],
        "Routing Cost Basis": summary["routing_cost_basis"],
        "Feasible": bool(validation["feasible"]),
    }


def _json_ready(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def matrix_stats(mat):
    if mat is None:
        return None
    arr = np.asarray(mat, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"shape": list(arr.shape), "finite_count": 0}
    return {
        "shape": list(arr.shape),
        "finite_count": int(finite.size),
        "min": round(float(finite.min()), 6),
        "max": round(float(finite.max()), 6),
        "mean": round(float(finite.mean()), 6),
        "median": round(float(np.median(finite)), 6),
    }


def _assignment_stop_detail(stop_idx: int, assigned_vehicle: int,
                            assignment_meta: dict, route_model: dict):
    detail = {
        "assignment_anchor_type": None,
        "assignment_anchor_index": None,
        "assignment_stage_cost_s": None,
    }
    if "seed_indices" in assignment_meta and len(assignment_meta["seed_indices"]) > assigned_vehicle:
        anchor = int(assignment_meta["seed_indices"][assigned_vehicle])
        detail.update({
            "assignment_anchor_type": "seed_stop",
            "assignment_anchor_index": anchor,
            "assignment_stage_cost_s": round(float(route_model["cost_matrix"][stop_idx, anchor]), 6),
        })
    elif "median_indices" in assignment_meta and len(assignment_meta["median_indices"]) > assigned_vehicle:
        anchor = int(assignment_meta["median_indices"][assigned_vehicle])
        detail.update({
            "assignment_anchor_type": "median_stop",
            "assignment_anchor_index": anchor,
            "assignment_stage_cost_s": round(float(route_model["cost_matrix"][stop_idx, anchor]), 6),
        })
    return detail


def build_vehicle_leg_trace(vehicle: int, order: list[int], route_model: dict):
    legs = []
    if not order:
        return legs

    start_ll = route_model["starts_ll"][vehicle]
    first = int(order[0])
    legs.append({
        "leg": 1,
        "from_type": "start",
        "from_id": f"start_{vehicle+1}",
        "from_lonlat": _json_ready(start_ll),
        "to_type": "stop",
        "to_id": first,
        "to_lonlat": _json_ready(route_model["points_ll"][first]),
        "cost_s": round(float(route_model["start_to_stop_dur"][vehicle, first]), 6),
    })

    leg_no = 2
    for prev_stop, next_stop in zip(order[:-1], order[1:]):
        prev_stop = int(prev_stop)
        next_stop = int(next_stop)
        legs.append({
            "leg": leg_no,
            "from_type": "stop",
            "from_id": prev_stop,
            "from_lonlat": _json_ready(route_model["points_ll"][prev_stop]),
            "to_type": "stop",
            "to_id": next_stop,
            "to_lonlat": _json_ready(route_model["points_ll"][next_stop]),
            "cost_s": round(float(route_model["cost_matrix"][prev_stop, next_stop]), 6),
        })
        leg_no += 1

    if route_model["end_policy"] != "open":
        last = int(order[-1])
        legs.append({
            "leg": leg_no,
            "from_type": "stop",
            "from_id": last,
            "from_lonlat": _json_ready(route_model["points_ll"][last]),
            "to_type": "end",
            "to_id": f"end_{vehicle+1}",
            "to_lonlat": _json_ready(route_model["ends_ll"][vehicle]),
            "cost_s": round(float(route_model["stop_to_end_dur"][last, vehicle]), 6),
        })
    return legs


def build_pipeline_debug_payload(letter, labels, centers, ordered_global, route_sequences,
                                 route_dist, route_dur, route_model_cost_per_vehicle,
                                 assignment_meta, route_model, validation, summary,
                                 ana, include_full_matrices=False):
    order_pos = {}
    for v, order in ordered_global.items():
        for pos, stop_idx in enumerate(order):
            order_pos[int(stop_idx)] = (int(v), int(pos))

    stop_details = []
    for stop_idx in range(route_model["n"]):
        v = int(labels[stop_idx])
        pos = order_pos.get(stop_idx, (v, None))[1]
        order = ordered_global.get(v, [])
        prev_stop = int(order[pos - 1]) if pos is not None and pos > 0 else None
        next_stop = int(order[pos + 1]) if pos is not None and pos < len(order) - 1 else None
        in_cost = (float(route_model["start_to_stop_dur"][v, stop_idx]) if pos == 0
                   else float(route_model["cost_matrix"][prev_stop, stop_idx]) if prev_stop is not None
                   else None)
        if next_stop is not None:
            out_cost = float(route_model["cost_matrix"][stop_idx, next_stop])
        elif route_model["end_policy"] != "open" and pos is not None:
            out_cost = float(route_model["stop_to_end_dur"][stop_idx, v])
        else:
            out_cost = None
        stop_details.append({
            "stop_idx": int(stop_idx),
            "stop_lonlat": _json_ready(route_model["points_ll"][stop_idx]),
            "assigned_vehicle": v + 1,
            "route_position": None if pos is None else pos + 1,
            "demand": round(float(route_model["demand_per_stop"]), 6),
            "model_inbound_cost_s": None if in_cost is None else round(in_cost, 6),
            "model_outbound_cost_s": None if out_cost is None else round(out_cost, 6),
            "previous_stop": prev_stop,
            "next_stop": next_stop,
            **_assignment_stop_detail(stop_idx, v, assignment_meta, route_model),
        })

    vehicles = []
    for v in range(route_model["K"]):
        order = list(map(int, ordered_global.get(v, [])))
        vehicles.append({
            "vehicle": v + 1,
            "start_lonlat": _json_ready(route_model["starts_ll"][v]),
            "end_lonlat": None if route_model["end_policy"] == "open" else _json_ready(route_model["ends_ll"][v]),
            "ordered_stops": order,
            "total_demand": round(float(len(order) * route_model["demand_per_stop"]), 6),
            "capacity": round(float(route_model["capacity_units"]), 6),
            "model_route_cost_s": round(float(route_model_cost_per_vehicle.get(v, 0.0)), 6),
            "road_distance_m": round(float(route_dist.get(v, 0.0)), 6),
            "road_duration_s": round(float(route_dur.get(v, 0.0)), 6),
            "feasible": bool(validation.get("vehicle_feasible", {}).get(v, False)),
            "leg_trace": build_vehicle_leg_trace(v, order, route_model),
            "route_sequence_lonlat": _json_ready(np.asarray(route_sequences.get(v, np.empty((0, 2))), dtype=float)),
        })

    payload = {
        "pipeline": letter,
        "summary": _json_ready(summary),
        "metrics": {
            "assignment_cost_total_s": round(float(ana.get("assignment_cost_total_s", 0.0)), 6),
            "model_route_cost_total_s": round(float(ana.get("routing_cost_total_s", 0.0)), 6),
            "road_duration_total_s": round(float(ana.get("total_dur_s", 0.0)), 6),
            "road_distance_total_m": round(float(ana.get("total_dist_m", 0.0)), 6),
            "feasible": bool(ana.get("feasible", False)),
        },
        "assignment_meta": _json_ready(assignment_meta),
        "labels": _json_ready(labels),
        "centers_normalized": _json_ready(centers),
        "ordered_global": _json_ready(ordered_global),
        "validation": _json_ready(validation),
        "vehicle_routes": vehicles,
        "stop_details": stop_details,
    }

    if include_full_matrices:
        payload["customer_cost_matrix_s"] = _json_ready(route_model["cost_matrix"])
        if "seed_indices" in assignment_meta:
            payload["assignment_anchor_matrix_s"] = _json_ready(
                route_model["cost_matrix"][:, assignment_meta["seed_indices"]]
            )
        elif "median_indices" in assignment_meta:
            payload["assignment_anchor_matrix_s"] = _json_ready(
                route_model["cost_matrix"][:, assignment_meta["median_indices"]]
            )
    return payload


def build_benchmark_debug_report(config, route_model, pipelines_payload,
                                 dist_matrix=None, include_full_matrices=False):
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": _json_ready(config),
        "route_model": {
            "start_policy": route_model["start_policy"],
            "end_policy": route_model["end_policy"],
            "roundtrip": bool(route_model["roundtrip"]),
            "capacity_units": round(float(route_model["capacity_units"]), 6),
            "demand_per_stop": round(float(route_model["demand_per_stop"]), 6),
            "routing_cost_basis": route_model["routing_cost_basis"],
            "terminal_costs_exact": bool(route_model["terminal_costs_exact"]),
            "starts_ll": _json_ready(route_model["starts_ll"]),
            "ends_ll": _json_ready(route_model["ends_ll"]),
            "shared_depot_ll": _json_ready(route_model["shared_depot_ll"]),
            "matrix_stats": {
                "customer_cost_s": matrix_stats(route_model["cost_matrix"]),
                "customer_distance_m": matrix_stats(dist_matrix),
                "start_to_stop_s": matrix_stats(route_model["start_to_stop_dur"]),
                "stop_to_end_s": matrix_stats(route_model["stop_to_end_dur"]),
            },
        },
        "pipelines": _json_ready(pipelines_payload),
        "benchmark_integrity_checks": [],
    }
    if include_full_matrices:
        report["route_model"]["customer_cost_matrix_s"] = _json_ready(route_model["cost_matrix"])
        report["route_model"]["customer_distance_matrix_m"] = _json_ready(dist_matrix)
        report["route_model"]["start_to_stop_matrix_s"] = _json_ready(route_model["start_to_stop_dur"])
        report["route_model"]["stop_to_end_matrix_s"] = _json_ready(route_model["stop_to_end_dur"])
    return report


def render_debug_report_text(report):
    lines = []
    lines.append("BENCHMARK TRACE EXPORT")
    lines.append(f"Generated at: {report.get('generated_at', 'n/a')}")
    cfg = report.get("config", {})
    lines.append(
        f"Config: city={cfg.get('city')} poi={cfg.get('poi_type')} "
        f"points={cfg.get('n_points_loaded')} vans={cfg.get('n_vans')}"
    )
    rm = report.get("route_model", {})
    lines.append(
        f"Shared route model: start={rm.get('start_policy')} end={rm.get('end_policy')} "
        f"roundtrip={rm.get('roundtrip')} demand/stop={rm.get('demand_per_stop')} "
        f"capacity={rm.get('capacity_units')}"
    )
    lines.append(f"Routing cost basis: {rm.get('routing_cost_basis')}")
    lines.append(f"Terminal start/end costs exact: {rm.get('terminal_costs_exact')}")
    lines.append(f"Starts: {rm.get('starts_ll')}")
    if rm.get("ends_ll") is not None:
        lines.append(f"Ends: {rm.get('ends_ll')}")
    matrix_stats_payload = rm.get("matrix_stats", {})
    if matrix_stats_payload:
        lines.append("Matrix stats:")
        for key, stats in matrix_stats_payload.items():
            lines.append(f"  {key}: {stats}")
    lines.append("")

    for letter, payload in report.get("pipelines", {}).items():
        lines.append(f"PIPELINE {letter}")
        summary = payload.get("summary", {})
        metrics = payload.get("metrics", {})
        lines.append(f"Assignment: {summary.get('assignment_method')}")
        lines.append(f"Sequencing: {summary.get('sequencing_method')}")
        lines.append(f"Mode: {summary.get('integration_mode')}")
        lines.append(f"Assignment cost basis: {summary.get('assignment_cost_basis')}")
        lines.append(f"Routing cost basis: {summary.get('routing_cost_basis')}")
        lines.append(
            f"Totals: assignment={metrics.get('assignment_cost_total_s')} s · "
            f"model_route={metrics.get('model_route_cost_total_s')} s · "
            f"road_time={metrics.get('road_duration_total_s')} s · "
            f"road_dist={metrics.get('road_distance_total_m')} m · "
            f"feasible={metrics.get('feasible')}"
        )
        lines.append(f"Assignment metadata: {payload.get('assignment_meta')}")
        lines.append("Validation:")
        for check in payload.get("validation", {}).get("checks", []):
            lines.append(f"  - {check.get('check')}: {check.get('ok')} ({check.get('detail')})")
        lines.append("Vehicles:")
        for vehicle in payload.get("vehicle_routes", []):
            lines.append(
                f"  Van {vehicle.get('vehicle')}: stops={vehicle.get('ordered_stops')} "
                f"model={vehicle.get('model_route_cost_s')} s road={vehicle.get('road_duration_s')} s "
                f"dist={vehicle.get('road_distance_m')} m feasible={vehicle.get('feasible')}"
            )
            lines.append(f"    start={vehicle.get('start_lonlat')} end={vehicle.get('end_lonlat')}")
            for leg in vehicle.get("leg_trace", []):
                lines.append(
                    f"    leg {leg.get('leg')}: {leg.get('from_type')} {leg.get('from_id')} -> "
                    f"{leg.get('to_type')} {leg.get('to_id')} = {leg.get('cost_s')} s"
                )
        lines.append("Stops:")
        for stop in payload.get("stop_details", []):
            lines.append(
                f"  stop {stop.get('stop_idx')}: van={stop.get('assigned_vehicle')} "
                f"pos={stop.get('route_position')} assignment={stop.get('assignment_stage_cost_s')} s "
                f"in={stop.get('model_inbound_cost_s')} s out={stop.get('model_outbound_cost_s')} s "
                f"prev={stop.get('previous_stop')} next={stop.get('next_stop')}"
            )
        lines.append("")
    integrity_rows = report.get("benchmark_integrity_checks", [])
    if integrity_rows:
        lines.append("BENCHMARK INTEGRITY CHECKS")
        for row in integrity_rows:
            lines.append(str(row))
        lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────
#  FRAGMENTATION / TERRITORY QUALITY METRICS
# ──────────────────────────────────────────────────────
def compute_territory_metrics(pts_n, labels, K):
    """
    Returns a dict of territory quality metrics beyond balance:
      avg_intra      — avg distance from each stop to its cluster centroid (compactness)
      hull_overlap   — fraction of cluster-pair bounding boxes that overlap (fragmentation)
      max_spread     — largest cluster's avg intra-dist (worst-case compactness)
      isolation      — avg min distance between cluster centroids (separation)
    """
    centroids = []
    for k in range(K):
        m = pts_n[labels == k]
        centroids.append(m.mean(0) if len(m) > 0 else np.array([0.5, 0.5]))
    centroids = np.array(centroids)

    # Avg intra-cluster distance
    intra_per_k = []
    for k in range(K):
        m = pts_n[labels == k]
        if len(m) > 0:
            intra_per_k.append(np.linalg.norm(m - centroids[k], axis=1).mean())
    avg_intra  = float(np.mean(intra_per_k)) if intra_per_k else 0.0
    max_spread = float(np.max(intra_per_k))  if intra_per_k else 0.0

    # Bounding-box overlap fraction
    bboxes = []
    for k in range(K):
        m = pts_n[labels == k]
        if len(m) >= 2:
            bboxes.append((m[:,0].min(), m[:,1].min(), m[:,0].max(), m[:,1].max()))
        else:
            bboxes.append(None)

    overlap_count = 0; pairs = 0
    for i in range(K):
        for j in range(i+1, K):
            if bboxes[i] is None or bboxes[j] is None: continue
            pairs += 1
            ax0,ay0,ax1,ay1 = bboxes[i]
            bx0,by0,bx1,by1 = bboxes[j]
            if ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0:
                overlap_count += 1
    hull_overlap = overlap_count / max(pairs, 1)

    # Average minimum separation between centroids
    if len(centroids) >= 2:
        from scipy.spatial import distance_matrix as sdm
        cdist = sdm(centroids, centroids)
        np.fill_diagonal(cdist, np.inf)
        isolation = float(cdist.min(axis=1).mean())
    else:
        isolation = 0.0

    return {
        "avg_intra":    round(avg_intra,   4),
        "max_spread":   round(max_spread,  4),
        "hull_overlap": round(hull_overlap, 3),
        "isolation":    round(isolation,   4),
    }


# ──────────────────────────────────────────────────────
#  ROUTE METRIC HELPERS
# ──────────────────────────────────────────────────────

def build_route_metrics(route_dist: dict, route_dur: dict, K: int, osrm_used: bool) -> dict:
    """
    Aggregate per-van OSRM route results into fleet-level metrics.
    route_dist : {k: dist_m}  — road distance per van in meters  (0 if OSRM failed)
    route_dur  : {k: dur_s}   — road duration per van in seconds (0 if OSRM failed)
    Returns dict with total/avg time and distance.
    """
    total_dist_m   = sum(route_dist.get(k, 0.0) for k in range(K))
    total_dur_s    = sum(route_dur.get(k, 0.0)  for k in range(K))
    active_vans = sum(1 for k in range(K) if route_dist.get(k, 0) > 0)
    avg_denom   = max(active_vans, 1)  # avoid div by zero

    return {
        "total_dist_m":       round(total_dist_m, 1),
        "total_dist_km":      round(total_dist_m / 1000, 3),
        "total_dur_s":        round(total_dur_s, 1),
        "total_dur_min":      round(total_dur_s / 60, 2),
        "avg_dist_km":        round(total_dist_m / 1000 / avg_denom, 3) if active_vans > 0 else 0.0,
        "avg_dur_min":        round(total_dur_s / 60 / avg_denom, 2) if active_vans > 0 else 0.0,
        "route_dist":         {k: round(route_dist.get(k,0), 1) for k in range(K)},
        "route_dur":          {k: round(route_dur.get(k,0), 1)  for k in range(K)},
        "osrm_route_metrics": osrm_used,
    }


def build_fuel_metrics(total_dist_km: float, liters_per_100km: float,
                        fuel_price: float, co2_per_liter: float) -> dict:
    """
    Estimate fuel consumption, cost, and CO₂ from total road distance.
    liters_per_100km : fuel efficiency (e.g. 8.5 L/100km for a van)
    fuel_price       : price per liter in local currency
    co2_per_liter    : kg CO₂ per liter burned (diesel ≈ 2.68, petrol ≈ 2.31)
    """
    fuel_liters = total_dist_km * liters_per_100km / 100.0
    fuel_cost   = fuel_liters * fuel_price
    co2_kg      = fuel_liters * co2_per_liter
    return {
        "fuel_liters": round(fuel_liters, 2),
        "fuel_cost":   round(fuel_cost, 2),
        "co2_kg":      round(co2_kg, 2),
    }





# ──────────────────────────────────────────────────────
#  ANALYTICS  (full metrics: workload + territory + route + fuel)
# ──────────────────────────────────────────────────────
def analytics(pts_n, labels, centers, name, elapsed_s, osrm_used,
              route_metrics=None, fuel_metrics=None):
    """
    Build the complete analytics dict for one pipeline run.
    route_metrics : output of build_route_metrics() — or None if not yet computed
    fuel_metrics  : output of build_fuel_metrics()  — or None
    """
    K     = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=K)
    msz   = sizes.mean()
    cv    = sizes.std() / msz if msz > 0 else 0
    terr  = compute_territory_metrics(pts_n, labels, K)

    d = {
        # identity
        "name":    name,
        "osrm":    osrm_used,
        # workload
        "sizes":   sizes.tolist(),
        "n_stops": int(sizes.sum()),
        "min":     int(sizes.min()),
        "max":     int(sizes.max()),
        "mean":    round(float(msz), 1),
        "cv":      round(float(cv), 4),
        # runtime
        "time_ms": round(elapsed_s * 1000, 1),
        # territory quality
        **terr,
    }

    # Route metrics (OSRM-based when available)
    if route_metrics:
        d.update({
            "total_dist_m":   route_metrics["total_dist_m"],
            "total_dist_km":  route_metrics["total_dist_km"],
            "total_dur_s":    route_metrics["total_dur_s"],
            "total_dur_min":  route_metrics["total_dur_min"],
            "avg_dist_km":    route_metrics["avg_dist_km"],
            "avg_dur_min":    route_metrics["avg_dur_min"],
            "osrm_route":     route_metrics["osrm_route_metrics"],
            # back-compat alias
            "tour_s":         route_metrics["total_dur_s"],
        })
    # Fuel metrics
    if fuel_metrics:
        d.update({
            "fuel_liters": fuel_metrics["fuel_liters"],
            "fuel_cost":   fuel_metrics["fuel_cost"],
            "co2_kg":      fuel_metrics["co2_kg"],
        })
    return d


# ──────────────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────────────
for k, v in [
    ("ll_pts", None), ("bbox", None), ("osm_tags", []),
    ("cost_matrix", None), ("dist_matrix", None), ("osrm_used", False),
    ("result_a", None), ("result_b", None),
    ("result_c", None), ("result_d", None),
    ("gif_a", None), ("gif_b", None),
    ("gif_c", None), ("gif_d", None),
    ("route_start_policy_used", "seeded"), ("route_end_policy_used", "open"),
    ("debug_report", None), ("debug_report_text", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

TILE_SOURCES = {
    "Dark (CartoDB)":  ctx.providers.CartoDB.DarkMatter,
    "Street (OSM)":    ctx.providers.OpenStreetMap.Mapnik,
    "Light (CartoDB)": ctx.providers.CartoDB.Positron,
}

# ──────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚐 Van Planner")
    st.divider()

    city  = st.selectbox("City", list(CITIES.keys()), index=0,
                          label_visibility="collapsed")
    poi_k = st.selectbox("POI Type", list(POI_FILTERS.keys()), index=0,
                          label_visibility="collapsed")
    st.divider()

    st.markdown('<p style="font-family:monospace;font-size:.7rem;color:#64748b;'
                'text-transform:uppercase;letter-spacing:.1em;margin-bottom:.2rem">'
                'Pickup Points</p>', unsafe_allow_html=True)
    n_points = st.number_input("points", min_value=1, max_value=1000, value=80,
                                step=10, label_visibility="collapsed")

    st.markdown('<p style="font-family:monospace;font-size:.7rem;color:#64748b;'
                'text-transform:uppercase;letter-spacing:.1em;margin-bottom:.2rem;'
                'margin-top:.6rem">Available Vans</p>', unsafe_allow_html=True)
    n_vans = st.number_input("vans", min_value=1, max_value=50, value=4, step=1,
                              label_visibility="collapsed")
    st.divider()

    col1, col2 = st.columns(2)
    fetch_btn = col1.button("📡 Fetch POIs", width="stretch", type="primary")
    run_btn   = col2.button("🚀 Run", width="stretch", type="primary",
                             disabled=st.session_state.ll_pts is None)

    if st.session_state.ll_pts is not None:
        tag = ('🛣 OSRM active' if st.session_state.osrm_used
               else '📐 Euclidean fallback')
        color = '#064e3b' if st.session_state.osrm_used else '#7f1d1d'
        border = '#10b981' if st.session_state.osrm_used else '#ef4444'
        text_c = '#6ee7b7' if st.session_state.osrm_used else '#fca5a5'
        st.markdown(
            f'<div style="text-align:center;margin-top:.3rem">'
            f'<span style="background:{color};border:1px solid {border};color:{text_c};'
            f'padding:3px 10px;border-radius:12px;font-family:monospace;font-size:.68rem">'
            f'{tag}</span></div>', unsafe_allow_html=True)

    st.divider()
    with st.expander("⚙️ Advanced Settings", expanded=False):
        tile_label   = st.selectbox("Map Style", list(TILE_SOURCES.keys()), index=0)
        cap_pct      = st.slider("Capacity Buffer %", 100, 200, 130,
                                  help="Max stops per van = ⌈(N/K) × buffer/100⌉")
        pm_iters     = st.slider("P-Median Iterations", 10, 80, 30, step=5)
        c_time_limit = st.slider("Pipeline C OR-Tools (s)", 4, 30, 8, step=2,
                                  help="Time budget for per-cluster path optimization")
        d_time_limit = st.slider("Pipeline D OR-Tools (s)", 8, 60, 15, step=5,
                                  help="Time budget for full CVRP. More = better routes.")
        st.markdown("**Shared Route Model**")
        route_start_policy = st.selectbox(
            "Route: Start policy",
            ["seeded", "centroid", "random"],
            index=0,
            help="Shared across A/B/C/D. seeded=common geographic van starts, "
                 "centroid=one shared depot, random=stress-test dispersed starts")
        route_end_policy = st.selectbox(
            "Route: End policy",
            ["open", "return_to_start", "return_to_depot"],
            index=0,
            help="Shared across A/B/C/D. open=end at last stop, "
                 "return_to_start=go back to van origin, "
                 "return_to_depot=finish at the shared centroid depot")
        demand_per_stop = st.number_input(
            "Demand per stop",
            value=1.0, step=0.5, min_value=0.5, max_value=10.0,
            help="Uniform demand used by every pipeline for capacity validation")
        st.markdown("**Fuel / Cost Assumptions**")
        fuel_lpk    = st.number_input("Fuel efficiency (L/100 km)", value=8.5, step=0.5,
                                       min_value=1.0, max_value=30.0,
                                       help="Typical delivery van: 8–12 L/100 km")
        fuel_price  = st.number_input("Fuel price (per liter, local currency)", value=2.0,
                                       step=0.1, min_value=0.1, max_value=20.0)
        co2_per_l   = st.number_input("CO₂ per liter (kg)", value=2.68, step=0.1,
                                       min_value=1.0, max_value=4.0,
                                       help="Diesel ≈ 2.68 kg/L · Petrol ≈ 2.31 kg/L")
        st.markdown("**Audit / Export**")
        enable_debug_trace = st.checkbox(
            "Detailed benchmark trace",
            value=True,
            help="Capture a full calculation log and enable JSON/TXT export after the run")
        include_full_matrices = st.checkbox(
            "Include full matrices in export",
            value=False,
            help="Adds full customer/start/end matrices and assignment matrices to the debug export. "
                 "Can be large for big point sets.")
        anim_frames  = st.slider("Animation Frames", 40, 200, 80, 20)
        anim_fps     = st.slider("Animation FPS", 10, 30, 20)
        trail_len    = st.slider("Trail Length", 5, 40, 16)

    try:    tile_src = TILE_SOURCES[tile_label]
    except: tile_src = ctx.providers.CartoDB.DarkMatter


# ──────────────────────────────────────────────────────
#  FETCH BUTTON
# ──────────────────────────────────────────────────────
if fetch_btn:
    bbox = CITIES[city]
    with st.spinner("Querying OpenStreetMap…"):
        ll, tags = fetch_osm(bbox, POI_FILTERS[poi_k], max_pts=n_points)
    if ll is None or len(ll) < max(n_vans + 1, 3):
        st.error("Not enough POIs returned. Try a different city or POI type.")
    else:
        prog = st.progress(0, text="Building road-network distance matrix via OSRM…")
        dur_mat, dist_mat, osrm_used = osrm_distance_matrix(ll)
        prog.progress(100, text="✓ Distance matrix ready"); prog.empty()
        for k, v in [
            ("ll_pts", ll), ("bbox", bbox), ("osm_tags", tags),
            ("cost_matrix", dur_mat), ("dist_matrix", dist_mat),
            ("osrm_used", osrm_used),
            ("result_a", None), ("result_b", None),
            ("result_c", None), ("result_d", None),
            ("gif_a", None), ("gif_b", None),
            ("gif_c", None), ("gif_d", None),
            ("debug_report", None), ("debug_report_text", None),
        ]:
            st.session_state[k] = v
        dist_note = " + distances" if dist_mat is not None else " (durations only)"
        st.success(
            f"✓ {len(ll)} POIs · "
            f"{'🛣 OSRM road matrix' + dist_note if osrm_used else '📐 Euclidean fallback'}")


# ──────────────────────────────────────────────────────
#  RUN BUTTON
# ──────────────────────────────────────────────────────
if run_btn and st.session_state.ll_pts is not None:
    ll        = st.session_state.ll_pts
    bbox      = st.session_state.bbox
    mat       = st.session_state.cost_matrix
    K         = n_vans
    osrm_used = st.session_state.osrm_used

    # Safe defaults without bare expressions that Streamlit would render.
    cap_pct = int(locals().get("cap_pct", 130))
    pm_iters = int(locals().get("pm_iters", 30))
    c_time_limit = int(locals().get("c_time_limit", 8))
    d_time_limit = int(locals().get("d_time_limit", 15))
    anim_frames = int(locals().get("anim_frames", 80))
    anim_fps = int(locals().get("anim_fps", 20))
    trail_len = int(locals().get("trail_len", 16))
    tile_src = locals().get("tile_src", ctx.providers.CartoDB.DarkMatter)

    pts_bbox = (ll[:,0].min(), ll[:,1].min(), ll[:,0].max(), ll[:,1].max())
    pts_n    = norm(ll, pts_bbox)
    dist_mat = st.session_state.dist_matrix   # may be None if OSRM didn't return distances

    # Safe defaults for fuel/depot settings
    fuel_lpk = float(locals().get("fuel_lpk", 8.5))
    fuel_price = float(locals().get("fuel_price", 2.0))
    co2_per_l = float(locals().get("co2_per_l", 2.68))
    route_start_policy = str(locals().get("route_start_policy", "seeded"))
    route_end_policy = str(locals().get("route_end_policy", "open"))
    demand_per_stop = float(locals().get("demand_per_stop", 1.0))
    enable_debug_trace = bool(locals().get("enable_debug_trace", True))
    include_full_matrices = bool(locals().get("include_full_matrices", False))

    def _fuel(dist_km):
        return build_fuel_metrics(dist_km, fuel_lpk, fuel_price, co2_per_l)

    route_model = build_shared_route_model(
        ll, K, cap_pct, mat,
        start_policy=route_start_policy,
        end_policy=route_end_policy,
        demand_per_stop=demand_per_stop,
    )

    def _finalize_pipeline(letter, title, ana_name, labels, centers, ordered_global,
                           route_model_cost_per_vehicle, assignment_meta,
                           summary, elapsed_s, route_sequences=None):
        if route_sequences is None:
            route_sequences = build_route_sequences_from_orders(ll, ordered_global, route_model)
        geoms, rdist, rdur, exact_routes = build_route_geometries(
            route_sequences, close_loop=False)
        rm = build_route_metrics(
            rdist, rdur, K,
            osrm_used and exact_routes and route_model["terminal_costs_exact"])
        ana = analytics(
            pts_n, labels, centers, ana_name, elapsed_s,
            osrm_used=osrm_used,
            route_metrics=rm,
            fuel_metrics=_fuel(rm["total_dist_km"]),
        )
        validation = validate_pipeline_result(labels, ordered_global, route_sequences, route_model)
        route_logs = build_route_logs(
            letter, labels, ordered_global, rdist, rdur,
            assignment_meta["assignment_cost_per_vehicle"],
            route_model_cost_per_vehicle, route_model, validation)
        ana.update({
            "assignment_cost_total_s": round(float(assignment_meta["assignment_cost_total"]), 2),
            "routing_cost_total_s": round(float(sum(route_model_cost_per_vehicle.values())), 2),
            "assignment_cost_basis": assignment_meta["assignment_cost_basis"],
            "capacity_enforced": bool(assignment_meta.get("capacity_enforced", True)),
            "capacity_units": route_model["capacity_units"],
            "demand_per_stop": route_model["demand_per_stop"],
            "route_start_policy": route_model["start_policy"],
            "route_end_policy": route_model["end_policy"],
            "roundtrip": route_model["roundtrip"],
            "feasible": validation["feasible"],
            "validation_checks": validation["checks"],
            "summary": summary,
            "van_log": route_logs,
            "integrity_row": build_integrity_row(letter, summary, route_model, validation),
        })
        fig = make_map(
            ll, labels, centers, bbox, title,
            road_geoms=geoms, tile_src=tile_src, route_model=route_model)
        trace_ctx = {
            "route_sequences": route_sequences,
            "route_dist": rdist,
            "route_dur": rdur,
            "validation": validation,
            "assignment_meta": assignment_meta,
            "route_model_cost_per_vehicle": route_model_cost_per_vehicle,
        }
        return fig, ana, geoms, trace_ctx

    prog = st.progress(0, text="Pipeline A — Min-Cost Flow…")

    # ── A: MCF + OSRM matrix path heuristic ────────────────────────────────
    t0 = time.perf_counter()
    La, Ca_n, meta_a = run_mcf(ll, K, cap_pct, mat, return_meta=True)
    La, Ca_n, meta_a, _map_a = realign_labels_to_shared_starts(La, Ca_n, route_model, meta_a)
    prog.progress(8, text="Pipeline A — fixed-start path sequencing…")
    ordered_a, path_cost_a = {}, {}
    for k in range(K):
        stop_indices = list(np.where(La == k)[0])
        order, cost = heuristic_sequence_route(stop_indices, k, route_model)
        ordered_a[k] = order
        path_cost_a[k] = cost
    ta = time.perf_counter() - t0
    sum_a = build_pipeline_summary(
        "Min-Cost Flow assignment",
        "OSRM duration-matrix path heuristic",
        "staged",
        meta_a["assignment_cost_basis"],
        route_model,
        "hard capacity in assignment, then validated on the realized route",
        "cost-first baseline",
        road_network_usage="assignment uses the OSRM customer matrix; routing uses the shared start/end path model on OSRM routes",
    )
    prog.progress(18, text="Rendering Pipeline A…")
    fig_a, ana_a, geoms_a, trace_a = _finalize_pipeline(
        "A", "A — Min-Cost Flow → OSRM path heuristic",
        "A: MCF + OSRM path", La, Ca_n, ordered_a, path_cost_a, meta_a, sum_a, ta)

    # ── B: P-Median + OSRM matrix path heuristic ───────────────────────────
    prog.progress(22, text="Pipeline B — P-Median territory assignment…")
    t0 = time.perf_counter()
    Lb, Cb_n, meta_b = run_pmedian(ll, K, cap_pct, pm_iters, mat, return_meta=True)
    Lb, Cb_n, meta_b, _map_b = realign_labels_to_shared_starts(Lb, Cb_n, route_model, meta_b)
    prog.progress(30, text="Pipeline B — fixed-start path sequencing…")
    ordered_b, path_cost_b = {}, {}
    for k in range(K):
        stop_indices = list(np.where(Lb == k)[0])
        order, cost = heuristic_sequence_route(stop_indices, k, route_model)
        ordered_b[k] = order
        path_cost_b[k] = cost
    tb = time.perf_counter() - t0
    sum_b = build_pipeline_summary(
        "Capacitated P-Median territory assignment",
        "OSRM duration-matrix path heuristic",
        "staged",
        meta_b["assignment_cost_basis"],
        route_model,
        "hard capacity in assignment, then validated on the realized route",
        "territory benchmark",
        road_network_usage="assignment uses the OSRM customer matrix; routing uses the shared start/end path model on OSRM routes",
    )
    prog.progress(40, text="Rendering Pipeline B…")
    fig_b, ana_b, geoms_b, trace_b = _finalize_pipeline(
        "B", "B — P-Median → OSRM path heuristic",
        "B: P-Median + OSRM path", Lb, Cb_n, ordered_b, path_cost_b, meta_b, sum_b, tb)

    # ── C: same assignment as A + OR-Tools path sequencing ─────────────────
    prog.progress(44, text=f"Pipeline C — OR-Tools path sequencing ({c_time_limit}s)…")
    t0 = time.perf_counter()
    Lc, Cc_n = La.copy(), Ca_n.copy()
    meta_c = {
        "assignment_cost_total": meta_a["assignment_cost_total"],
        "assignment_cost_per_vehicle": dict(meta_a["assignment_cost_per_vehicle"]),
        "assignment_cost_basis": meta_a["assignment_cost_basis"],
        "capacity_enforced": True,
    }
    Lc, Cc_n, ordered_c, path_cost_c = run_pipeline_c(
        Lc, Cc_n, route_model, time_limit_s=c_time_limit)
    tc = time.perf_counter() - t0
    sum_c = build_pipeline_summary(
        "Min-Cost Flow assignment (identical to A)",
        "OR-Tools single-vehicle path per assigned territory",
        "staged",
        meta_c["assignment_cost_basis"],
        route_model,
        "hard capacity in assignment, then validated on the realized route",
        "local sequencing improvement",
        road_network_usage="assignment uses the OSRM customer matrix; sequencing optimizes the shared start/end path and is rendered on OSRM routes",
    )
    prog.progress(63, text="Rendering Pipeline C…")
    fig_c, ana_c, geoms_c, trace_c = _finalize_pipeline(
        "C", "C — MCF + OR-Tools fixed-start path",
        "C: MCF + OR-Tools path", Lc, Cc_n, ordered_c, path_cost_c, meta_c, sum_c, tc)

    # ── D: Full OR-Tools CVRP ──────────────────────────────────────────────
    prog.progress(67, text=f"Pipeline D — Full OR-Tools CVRP ({d_time_limit}s)…")
    t0 = time.perf_counter()
    Ld, Cd_n, ordered_d, route_ll_d, meta_d = run_pipeline_d(
        ll, route_model, time_limit_s=d_time_limit)
    td = time.perf_counter() - t0
    path_cost_d = {
        k: compute_vehicle_path_cost(ordered_d.get(k, []), k, route_model)
        for k in range(K)
    }
    meta_assign_d = {
        "assignment_cost_total": float(sum(path_cost_d.values())),
        "assignment_cost_per_vehicle": path_cost_d,
        "assignment_cost_basis": "Joint OR-Tools route objective with explicit starts/ends",
        "capacity_enforced": True,
    }
    sum_d = build_pipeline_summary(
        "Integrated OR-Tools CVRP assignment+routing",
        "Integrated OR-Tools CVRP",
        "joint",
        meta_assign_d["assignment_cost_basis"],
        route_model,
        "hard capacity dimension inside the solver and validated on the realized route",
        "integrated full benchmark",
        road_network_usage="the joint solver uses the OSRM customer matrix plus explicit OSRM start/end legs inside one CVRP objective",
    )
    prog.progress(87, text="Rendering Pipeline D…")
    fig_d, ana_d, geoms_d, trace_d = _finalize_pipeline(
        "D",
        f"D — Full OR-Tools CVRP · {route_start_policy} starts · {route_end_policy.replace('_', ' ')}",
        "D: Full OR-Tools CVRP", Ld, Cd_n, ordered_d, path_cost_d,
        meta_assign_d, sum_d, td, route_sequences=route_ll_d)

    prog.progress(92, text="Rendering animations…")
    gif_a = make_gif(ll, La, Ca_n, bbox, geoms_a, tile_src, route_model, anim_frames, anim_fps, trail_len)
    gif_b = make_gif(ll, Lb, Cb_n, bbox, geoms_b, tile_src, route_model, anim_frames, anim_fps, trail_len)
    gif_c = make_gif(ll, Lc, Cc_n, bbox, geoms_c, tile_src, route_model, anim_frames, anim_fps, trail_len)
    gif_d = make_gif(ll, Ld, Cd_n, bbox, geoms_d, tile_src, route_model, anim_frames, anim_fps, trail_len)

    prog.progress(100, text="Done."); prog.empty()

    st.session_state.result_a = (fig_a, ana_a, La, Ca_n, geoms_a)
    st.session_state.result_b = (fig_b, ana_b, Lb, Cb_n, geoms_b)
    st.session_state.result_c = (fig_c, ana_c, Lc, Cc_n, geoms_c)
    st.session_state.result_d = (fig_d, ana_d, Ld, Cd_n, geoms_d)
    st.session_state.gif_a = gif_a; st.session_state.gif_b = gif_b
    st.session_state.gif_c = gif_c; st.session_state.gif_d = gif_d
    st.session_state.route_start_policy_used = route_start_policy
    st.session_state.route_end_policy_used = route_end_policy
    if enable_debug_trace:
        pipelines_payload = {
            "A": build_pipeline_debug_payload(
                "A", La, Ca_n, ordered_a, trace_a["route_sequences"],
                trace_a["route_dist"], trace_a["route_dur"], path_cost_a,
                meta_a, route_model, trace_a["validation"], sum_a, ana_a,
                include_full_matrices=include_full_matrices),
            "B": build_pipeline_debug_payload(
                "B", Lb, Cb_n, ordered_b, trace_b["route_sequences"],
                trace_b["route_dist"], trace_b["route_dur"], path_cost_b,
                meta_b, route_model, trace_b["validation"], sum_b, ana_b,
                include_full_matrices=include_full_matrices),
            "C": build_pipeline_debug_payload(
                "C", Lc, Cc_n, ordered_c, trace_c["route_sequences"],
                trace_c["route_dist"], trace_c["route_dur"], path_cost_c,
                meta_c, route_model, trace_c["validation"], sum_c, ana_c,
                include_full_matrices=include_full_matrices),
            "D": build_pipeline_debug_payload(
                "D", Ld, Cd_n, ordered_d, trace_d["route_sequences"],
                trace_d["route_dist"], trace_d["route_dur"], path_cost_d,
                meta_assign_d, route_model, trace_d["validation"], sum_d, ana_d,
                include_full_matrices=include_full_matrices),
        }
        debug_report = build_benchmark_debug_report(
            {
                "city": city,
                "poi_type": poi_k,
                "n_points_requested": int(n_points),
                "n_points_loaded": int(len(ll)),
                "n_vans": int(K),
                "cap_pct": int(cap_pct),
                "start_policy": route_start_policy,
                "end_policy": route_end_policy,
                "demand_per_stop": float(demand_per_stop),
                "pm_iters": int(pm_iters),
                "c_time_limit_s": int(c_time_limit),
                "d_time_limit_s": int(d_time_limit),
                "osrm_used": bool(osrm_used),
                "distance_matrix_available": bool(dist_mat is not None),
            },
            route_model,
            pipelines_payload,
            dist_matrix=dist_mat,
            include_full_matrices=include_full_matrices,
        )
        debug_report["benchmark_integrity_checks"] = _json_ready([
            ana_a.get("integrity_row", {}),
            ana_b.get("integrity_row", {}),
            ana_c.get("integrity_row", {}),
            ana_d.get("integrity_row", {}),
        ])
        st.session_state.debug_report = debug_report
        st.session_state.debug_report_text = render_debug_report_text(debug_report)
    else:
        st.session_state.debug_report = None
        st.session_state.debug_report_text = None


# ──────────────────────────────────────────────────────
#  DISPLAY
# ──────────────────────────────────────────────────────
st.markdown(
    '<div class="page-title"><span class="page-title-icon">🚐</span>'
    '<span>Van Territory Planner</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#64748b;font-family:monospace;font-size:.75rem;margin-top:-.4rem">'
    '4 pipelines · MCF · P-Median · OR-Tools path · Full CVRP · OSRM street routing</p>',
    unsafe_allow_html=True)

if st.session_state.ll_pts is None:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0a0f1e"); ax.set_facecolor("#0d1526")
    ax.text(0.5, 0.5, "← Set pickup points & vans, then Fetch POIs",
            ha="center", va="center", fontsize=13, color="#334155",
            fontfamily="monospace", style="italic")
    ax.set_axis_off()
    st.pyplot(fig, width="stretch"); plt.close()

elif st.session_state.result_a is None:
    ll = st.session_state.ll_pts; bbox = st.session_state.bbox
    tile_src = locals().get("tile_src", ctx.providers.CartoDB.DarkMatter)
    n_vans = int(locals().get("n_vans", 4))
    with st.spinner("Loading map…"):
        fig = make_map(ll, np.zeros(len(ll), dtype=int), ll[:1].copy(),
                       bbox, f"{len(ll)} POIs · {n_vans} vans · press Run →",
                       tile_src=tile_src)
    st.pyplot(fig, width="stretch"); plt.close()

else:
    fig_a, ana_a, La, Ca_n, geoms_a = st.session_state.result_a
    fig_b, ana_b, Lb, Cb_n, geoms_b = st.session_state.result_b
    fig_c, ana_c, Lc, Cc_n, geoms_c = st.session_state.result_c
    fig_d, ana_d, Ld, Cd_n, geoms_d = st.session_state.result_d
    pts_n = norm(st.session_state.ll_pts,
                 (st.session_state.ll_pts[:,0].min(), st.session_state.ll_pts[:,1].min(),
                  st.session_state.ll_pts[:,0].max(), st.session_state.ll_pts[:,1].max()))

    # ── pipeline cards ──────────────────────────────────────────────────────
    ca, cb, cc, cd = st.columns(4, gap="small")
    CARDS = [
        (ca, "#3b82f6", "A — Min-Cost Flow",
         "COST-FIRST BASELINE",
         "MCF finds the globally optimal van-to-stop assignment. "
         "A fixed-start path heuristic then sequences each territory on the OSRM matrix.<br>"
         "<b>Role:</b> baseline. Cheapest assignment, fast."),
        (cb, "#f59e0b", "B — P-Median",
         "TERRITORY-FIRST BASELINE",
         "Finds K geographic territory anchors minimising distance to zone centres. "
         "A fixed-start path heuristic then sequences each territory on the OSRM matrix.<br>"
         "<b>Role:</b> territory design benchmark. Shows whether compact zones matter."),
        (cc, "#10b981", "C — MCF + OR-Tools Path",
         "SAME ZONES AS A, BETTER ORDERING",
         "MCF assignment (identical to A) + OR-Tools path routing under the same "
         "shared start/end assumptions. Answers: was A weak because of stop ordering?<br>"
         "<b>Role:</b> isolates routing quality from assignment quality."),
        (cd, "#8b5cf6", "D — Full OR-Tools CVRP",
         "INTEGRATED SINGLE-STAGE BENCHMARK",
         "OR-Tools decides assignment AND route order simultaneously with capacity "
         "constraints. The real production-shaped free benchmark.<br>"
         "<b>Role:</b> main serious benchmark. Closest to modern VRP solvers.<br>"
         "<b style='color:#facc15'>Note:</b> D now shares the same start/end model as "
         "A/B/C, so route totals are apples-to-apples."),
    ]
    for col, color, title, sub, desc in CARDS:
        col.markdown(f"""<div style="background:#0d1526;border:2px solid {color}33;
            border-radius:8px;padding:.8rem;height:100%">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.82rem;
                 font-weight:600;color:{color};margin-bottom:.2rem">{title}</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.62rem;
                 color:{color}99;margin-bottom:.5rem">{sub}</div>
            <div style="font-size:.72rem;color:#94a3b8;line-height:1.55">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── per-pipeline result columns ─────────────────────────────────────────
    PIPELINES = [
        ("A", fig_a, ana_a, st.session_state.gif_a, "#3b82f6"),
        ("B", fig_b, ana_b, st.session_state.gif_b, "#f59e0b"),
        ("C", fig_c, ana_c, st.session_state.gif_c, "#10b981"),
        ("D", fig_d, ana_d, st.session_state.gif_d, "#8b5cf6"),
    ]
    cols4 = st.columns(4, gap="small")

    def show_pipeline(col, letter, fig, ana, gif_path, color):
        with col:
            st.markdown(
                f'<p style="font-family:monospace;font-size:.78rem;color:{color};'
                f'font-weight:600;margin-bottom:.3rem">Pipeline {letter}</p>',
                unsafe_allow_html=True)
            mt, at, st2 = st.tabs(["📍 MAP", "🚗 ANIM", "📊 STATS"])
            with mt:
                st.pyplot(fig, width="stretch"); plt.close()
            with at:
                if gif_path and os.path.exists(gif_path):
                    with open(gif_path, "rb") as f:
                        gb = f.read()
                    st.image(gb, width="stretch")
                    st.download_button(f"⬇️ GIF {letter}", data=gb,
                                       file_name=f"dispatch_{letter.lower()}.gif",
                                       mime="image/gif", width="stretch")
            with st2:
                cv  = ana["cv"]
                cls = ("metric-good" if cv < 0.15 else
                       "metric-warn" if cv < 0.35 else "metric-bad")
                def mc(label, val, c=""):
                    st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {c}">{val}</div>
                    </div>""", unsafe_allow_html=True)

                # ── Workload ─────────────────────────────────────────────
                mc("Balance CV ↓", f"{cv:.3f}", cls)
                mc("Stops: min/avg/max",
                   f"{ana['min']} / {ana['mean']:.0f} / {ana['max']}")
                mc("Exec Time", f"{ana['time_ms']:.0f} ms")
                mc("Assignment Cost", f"{ana.get('assignment_cost_total_s', 0):.1f} s")
                mc("Model Route Cost", f"{ana.get('routing_cost_total_s', 0):.1f} s")
                mc("Feasible", "YES" if ana.get("feasible") else "NO",
                   "metric-good" if ana.get("feasible") else "metric-bad")

                # ── Route metrics ─────────────────────────────────────────
                st.markdown('<p style="font-family:monospace;font-size:.62rem;color:#475569;'
                    'text-transform:uppercase;letter-spacing:.08em;margin:.4rem 0 .25rem">'
                    'Route Metrics</p>', unsafe_allow_html=True)
                if ana.get("osrm_route") and ana.get("total_dur_s", 0) > 0:
                    mc("Total Drive Time",
                       f"{ana['total_dur_min']:.1f} min ({ana['total_dur_s']:.0f} s)")
                    mc("Total Distance",
                       f"{ana['total_dist_km']:.2f} km ({ana['total_dist_m']:.0f} m)")
                    mc("Avg per Van",
                       f"{ana['avg_dur_min']:.1f} min · {ana['avg_dist_km']:.2f} km")
                elif ana.get("total_dur_s", 0) > 0:
                    mc("Total Drive Time", f"{ana['total_dur_min']:.1f} min (approx)", "metric-warn")
                    mc("Total Distance", f"{ana['total_dist_km']:.2f} km (approx)", "metric-warn")
                    mc("Avg per Van", f"{ana['avg_dur_min']:.1f} min · {ana['avg_dist_km']:.2f} km", "metric-warn")
                else:
                    mc("Total Route", "OSRM unavailable", "metric-bad")

                # ── Fuel / CO₂ ───────────────────────────────────────────
                if ana.get("fuel_liters", 0) > 0:
                    st.markdown('<p style="font-family:monospace;font-size:.62rem;color:#475569;'
                        'text-transform:uppercase;letter-spacing:.08em;margin:.4rem 0 .25rem">'
                        'Fuel & Emissions</p>', unsafe_allow_html=True)
                    mc("Fuel Used", f"{ana['fuel_liters']:.1f} L")
                    mc("Fuel Cost", f"{ana['fuel_cost']:.2f}")
                    mc("CO₂ Emitted", f"{ana['co2_kg']:.1f} kg")

                # ── Territory quality ─────────────────────────────────────
                st.markdown('<p style="font-family:monospace;font-size:.62rem;color:#475569;'
                    'text-transform:uppercase;letter-spacing:.08em;margin:.4rem 0 .25rem">'
                    'Territory Quality</p>', unsafe_allow_html=True)
                mc("Avg Intra-Dist ↓", f"{ana['avg_intra']:.3f}")
                mc("Zone Overlap ↓",
                   f"{ana['hull_overlap']*100:.0f}%",
                   "metric-good" if ana['hull_overlap'] < 0.4 else
                   "metric-warn" if ana['hull_overlap'] < 0.7 else "metric-bad")
                mc("Isolation ↑", f"{ana['isolation']:.3f}")
                st.pyplot(size_chart(ana["sizes"]), width="stretch"); plt.close()

                # ── Assumptions / audit block ────────────────────────────────
                summ = ana.get("summary", {})
                st.markdown('<p style="font-family:monospace;font-size:.62rem;color:#475569;'
                    'text-transform:uppercase;letter-spacing:.08em;margin:.4rem 0 .25rem">'
                    'Assumptions</p>', unsafe_allow_html=True)
                st.markdown(f"""<div class="insight-box">
                <b>Assignment:</b> {summ.get('assignment_method', 'n/a')}<br>
                <b>Sequencing:</b> {summ.get('sequencing_method', 'n/a')}<br>
                <b>Benchmark mode:</b> {summ.get('integration_mode', 'n/a')}<br>
                <b>Route model:</b> {summ.get('route_model', 'n/a')}<br>
                <b>Road-network usage:</b> {summ.get('road_network_usage', 'n/a')}<br>
                <b>Assignment cost basis:</b> {summ.get('assignment_cost_basis', 'n/a')}<br>
                <b>Routing cost basis:</b> {summ.get('routing_cost_basis', 'n/a')}<br>
                <b>Capacity:</b> {summ.get('capacity_treatment', 'n/a')}<br>
                <b>Use case:</b> {summ.get('intended_use', 'n/a')}
                </div>""", unsafe_allow_html=True)

                # ── Validation output example ────────────────────────────────
                st.markdown('<p style="font-family:monospace;font-size:.62rem;color:#475569;'
                    'text-transform:uppercase;letter-spacing:.08em;margin:.4rem 0 .25rem">'
                    'Validation</p>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(ana.get("validation_checks", [])),
                             width="stretch", hide_index=True)

                # ── Per-van log ──────────────────────────────────────────────
                st.markdown('<p style="font-family:monospace;font-size:.62rem;color:#475569;'
                    'text-transform:uppercase;letter-spacing:.08em;margin:.4rem 0 .25rem">'
                    'Per-Van Log</p>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(ana.get("van_log", [])),
                             width="stretch", hide_index=True)

    for col, (letter, fig, ana, gif, color) in zip(cols4, PIPELINES):
        show_pipeline(col, letter, fig, ana, gif, color)

    # ── scoreboard ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚖️ Head-to-Head Scoreboard")

    COLORS = {"A": "#3b82f6", "B": "#f59e0b", "C": "#10b981", "D": "#8b5cf6"}
    DIM = "#334155"

    scoreboard_metrics = [
        ("Balance CV ↓",      "cv",           True,  "{:.3f}"),
        ("Min Stops ↑",       "min",          False, "{:.0f}"),
        ("Max Stops ↓",       "max",          True,  "{:.0f}"),
        ("Drive Time ↓ (min)","total_dur_min",True,  "{:.1f}"),
        ("Distance ↓ (km)",   "total_dist_km",True,  "{:.2f}"),
        ("Fuel Used ↓ (L)",   "fuel_liters",  True,  "{:.1f}"),
        ("CO₂ ↓ (kg)",        "co2_kg",       True,  "{:.1f}"),
        ("Avg Intra ↓",       "avg_intra",    True,  "{:.3f}"),
        ("Zone Overlap ↓",    "hull_overlap", True,  "{:.2f}"),
        ("Exec Time ↓ (ms)",  "time_ms",      True,  "{:.0f}"),
    ]
    all_ana = [ana_a, ana_b, ana_c, ana_d]
    all_letters = ["A", "B", "C", "D"]

    scols = st.columns(len(scoreboard_metrics))
    for col, (label, key, lower, fmt) in zip(scols, scoreboard_metrics):
        vals = [a.get(key, 0) for a in all_ana]
        best = min(vals) if lower else max(vals)
        winner = all_letters[vals.index(best)]
        rows = ""
        for letter, val in zip(all_letters, vals):
            c = COLORS[letter] if val == best else DIM
            rows += f'<div style="font-family:monospace;font-size:.85rem;color:{c}">{letter}: {fmt.format(val)}</div>'
        col.markdown(f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div style="margin-top:.3rem">{rows}</div>
        <div style="font-family:monospace;font-size:.62rem;
             color:{COLORS[winner]};margin-top:.3rem">🏆 Pipeline {winner}</div>
        </div>""", unsafe_allow_html=True)

    # ── benchmark integrity checks ──────────────────────────────────────────
    st.divider()
    st.markdown("### Benchmark Integrity Checks")
    integrity_df = pd.DataFrame([
        ana_a.get("integrity_row", {}),
        ana_b.get("integrity_row", {}),
        ana_c.get("integrity_row", {}),
        ana_d.get("integrity_row", {}),
    ])
    st.dataframe(integrity_df, width="stretch", hide_index=True)

    # ── detailed trace export ──────────────────────────────────────────────
    if st.session_state.debug_report is not None:
        st.divider()
        st.markdown("### Detailed Benchmark Trace")
        debug_json = json.dumps(st.session_state.debug_report, indent=2, ensure_ascii=False)
        debug_text = st.session_state.debug_report_text or ""
        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇️ Download Trace JSON",
            data=debug_json,
            file_name="benchmark_trace.json",
            mime="application/json",
            width="stretch",
        )
        d2.download_button(
            "⬇️ Download Trace TXT",
            data=debug_text,
            file_name="benchmark_trace.txt",
            mime="text/plain",
            width="stretch",
        )
        with st.expander("Trace Preview", expanded=False):
            st.markdown(
                '<p style="font-family:monospace;font-size:.68rem;color:#94a3b8">'
                'The exported trace includes route-model inputs, matrix stats, assignment metadata, '
                'per-stop details, per-van leg-by-leg calculations, validations, and final pipeline metrics.'
                '</p>',
                unsafe_allow_html=True,
            )
            st.code(debug_text[:12000] + ("\n\n... truncated in preview ..." if len(debug_text) > 12000 else ""),
                    language="text")

    # ── metrics legend ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📖 Metrics Legend")

    legend = [
        ("⚖️", "Balance CV",
         "Standard deviation ÷ mean of cluster sizes = how evenly stops are split. "
         "<b style='color:#4ade80'>&lt;0.15</b> = well balanced · "
         "<b style='color:#facc15'>0.15–0.35</b> = moderate · "
         "<b style='color:#f87171'>&gt;0.35</b> = skewed. "
         "A skewed fleet means some drivers are overloaded while others are idle."),

        ("🔢", "Min / Avg / Max Stops",
         "Smallest, mean, and largest stop count across all vans. "
         "Ideally all three close to N÷K. A large gap means one driver finishes "
         "at noon while another works until midnight — direct labour cost implication."),

        ("⏱️", "Total Drive Time",
         "Sum of all van driving times for one complete dispatch loop, in minutes (and seconds). "
         "Computed from OSRM /route/ response when active — real road network time. "
         "Falls back to cost-matrix sum if OSRM geometry is unavailable. "
         "Directly proportional to driver wages and vehicle running costs."),

        ("📏", "Total Distance",
         "Sum of all van driving distances in km and meters. "
         "Sourced from OSRM /route/ distance field when active — actual road distance, "
         "not straight-line. Used to compute fuel consumption and CO₂. "
         "<b>Note:</b> OSRM public server may not always return distances — "
         "the app falls back gracefully without crashing."),

        ("⛽", "Fuel Used / Cost / CO₂",
         "Estimated from total road distance using the assumptions in Advanced Settings. "
         "<code>fuel_L = dist_km × L/100km ÷ 100</code> · "
         "<code>cost = fuel_L × price/L</code> · "
         "<code>CO₂ = fuel_L × kg/L</code>. "
         "Defaults: 8.5 L/100 km · €2.00/L · 2.68 kg CO₂/L (diesel). "
         "Adjust in ⚙️ Advanced Settings. These are estimates, not exact values."),

        ("📐", "Avg Intra-Dist",
         "Average distance from each stop to its cluster centroid in normalised space. "
         "Measures geographic <b>compactness</b>. "
         "Compact zones mean less zigzagging even before route optimisation."),

        ("🗺️", "Zone Overlap %",
         "Fraction of cluster-pair bounding boxes that overlap. "
         "High overlap = vans cover the same streets — operationally messy and confusing. "
         "<b style='color:#4ade80'>&lt;40%</b> = clean territory · "
         "<b style='color:#facc15'>40–70%</b> = some overlap · "
         "<b style='color:#f87171'>&gt;70%</b> = fragmented."),

        ("↔️", "Isolation",
         "Average minimum distance between cluster centroids. "
         "Higher = zones are well-separated with clear geographic boundaries. "
         "Low isolation = zones bleed into each other."),

        ("⏱", "Exec Time",
         "Wall-clock ms. A and B are fast deterministic algorithms. "
         "C and D use OR-Tools with a configurable time budget — "
         "they often exit early once the solver converges. "
         "In production these run on dedicated infrastructure in parallel; "
         "wall time matters less than solution quality."),

        ("🏗️", "Shared Route Model",
         "All four pipelines now use the same start policy, end policy, demand model, "
         "and capacity definition.<br>"
         "<b>seeded</b> = common geographic van starts for every pipeline.<br>"
         "<b>centroid</b> = one shared depot for every pipeline.<br>"
         "<b>random</b> = shared random starts for robustness testing.<br>"
         "<b>open</b> = finish at the last stop.<br>"
         "<b>return_to_start</b> = finish where that van started.<br>"
         "<b>return_to_depot</b> = finish at the shared centroid depot."),
    ]

    for icon, title, desc in legend:
        st.markdown(f"""<div style="background:#0d1526;border:1px solid #1e293b;
            border-radius:8px;padding:.75rem 1rem;margin-bottom:.45rem;
            display:flex;gap:.9rem;align-items:flex-start">
          <div style="font-size:1.2rem;line-height:1;margin-top:.1rem">{icon}</div>
          <div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.77rem;
                 font-weight:600;color:#e2e8f0;margin-bottom:.25rem">{title}</div>
            <div style="font-size:.74rem;color:#94a3b8;line-height:1.6">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── status ───────────────────────────────────────────────────────────────
    st.divider()
    osrm = st.session_state.osrm_used
    dist_available = st.session_state.dist_matrix is not None
    st.markdown(f"""<div class="insight-box">
    {'<b style="color:#4ade80">🛣 OSRM active</b>' if osrm else '<b style="color:#f87171">📐 OSRM unavailable — Euclidean fallback</b>'}
    {" · distances available" if dist_available else " · durations only (distances estimated from duration×speed)" if osrm else ""}
    &nbsp;·&nbsp; <b style="color:#a78bfa">🔧 OR-Tools</b>:
    C = per-cluster OR-Tools path (same zones as A, better ordering) ·
    Shared route model = {st.session_state.get('route_start_policy_used','seeded')} starts · {st.session_state.get('route_end_policy_used','open').replace('_', ' ')}.<br>
    <b>Approximations in use:</b> fuel metrics assume constant speed/consumption ·
    zone overlap uses bounding-box proxy (not exact hull intersection) ·
    open-route mode models return arcs as zero-cost (approximation).<br>
    <b>Production gap:</b> replace OSRM public API with self-hosted instance or
    Google Maps Distance Matrix for real-time traffic.
    </div>""", unsafe_allow_html=True)
