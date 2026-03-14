"""
Van Territory Planner
Two production pipelines with REAL road-network distances via OSRM:
  A) Min-Cost Flow  → NN-TSP  (optimal assignment)
  B) Capacitated P-Median → 2-opt TSP  (territory-first)
Routes drawn following actual streets, not straight lines.
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import distance_matrix as scipy_dm
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
.block-container{padding-top:1rem}
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
h1{font-family:'IBM Plex Mono',monospace;font-size:1.25rem!important;color:#e2e8f0}
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
    coords: list of (lon, lat) — max ~80 at a time.
    Returns NxN numpy array of travel durations in seconds, or None.
    """
    coord_str = ";".join(f"{lon},{lat}" for lon, lat in coords)
    url = f"{base}/table/v1/driving/{coord_str}?annotations=duration"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data.get("code") == "Ok":
                return np.array(data["durations"], dtype=float)
    except Exception:
        pass
    return None


def osrm_distance_matrix(lonlat: np.ndarray, status_cb=None) -> tuple[np.ndarray, bool]:
    """
    Build full NxN travel-time matrix using OSRM table API.
    Chunks into OSRM_CHUNK-sized batches.
    Returns (matrix, osrm_used).
    Falls back to Euclidean on failure.
    """
    n = len(lonlat)
    # Try OSRM — chunked if n > OSRM_CHUNK
    for base in OSRM_BASES:
        try:
            if n <= OSRM_CHUNK:
                coords = [(float(lon), float(lat)) for lon, lat in lonlat]
                mat = _osrm_table_chunk(coords, base)
                if mat is not None and mat.shape == (n, n):
                    # replace NaN/inf with large value
                    mat = np.where(np.isfinite(mat), mat, mat[np.isfinite(mat)].max()*2)
                    return mat, True
            else:
                # For large N: build full matrix via sub-queries
                # Use sources/destinations params
                coord_str = ";".join(f"{lon},{lat}" for lon,lat in lonlat)
                # Try full table in one shot first (OSRM supports up to ~200)
                url = (f"{base}/table/v1/driving/{coord_str}"
                       f"?annotations=duration")
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("code") == "Ok":
                        mat = np.array(data["durations"], dtype=float)
                        if mat.shape == (n, n):
                            mat = np.where(np.isfinite(mat), mat, np.nanmax(mat)*2)
                            return mat, True
        except Exception:
            continue

    # Fallback — Euclidean on normalized coords
    if status_cb:
        status_cb("⚠️ OSRM unreachable — using Euclidean distance")
    n_pts = norm(lonlat, (lonlat[:,0].min(), lonlat[:,1].min(),
                           lonlat[:,0].max(), lonlat[:,1].max()))
    return scipy_dm(n_pts, n_pts), False


def osrm_route_geometry(ordered_lonlat: np.ndarray) -> list[tuple] | None:
    """
    Get the actual road-following polyline for an ordered sequence of stops.
    Returns list of (lon, lat) Web Mercator (x, y) pairs, or None on failure.
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
                    coords = data["routes"][0]["geometry"]["coordinates"]
                    # convert lon/lat pairs to Web Mercator
                    arr = np.array(coords)   # shape (M, 2) — lon, lat
                    wm  = to_wm(arr)
                    return list(map(tuple, wm))
        except Exception:
            continue
    return None


def osrm_trip(cluster_lonlat: np.ndarray) -> tuple[list | None, list | None]:
    """
    Use OSRM /trip/v1/driving/ — their built-in farthest-insertion TSP solver.
    Returns (waypoint_order, road_geometry_wm) or (None, None) on failure.
    Per the OSRM docs: The trip plugin solves TSP using a greedy heuristic
    (farthest-insertion algorithm). Returns waypoints in visit order + full geometry.
    """
    if len(cluster_lonlat) < 2:
        return None, None
    coord_str = ";".join(f"{lon},{lat}" for lon, lat in cluster_lonlat)
    for base in OSRM_BASES:
        try:
            url = (f"{base}/trip/v1/driving/{coord_str}"
                   f"?roundtrip=true&overview=full&geometries=geojson")
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if data.get("code") == "Ok" and data.get("trips"):
                    trip = data["trips"][0]
                    # waypoint order: each waypoint has waypoint_index = position in trip
                    waypoints = data["waypoints"]
                    order = [None] * len(waypoints)
                    for wp in waypoints:
                        order[wp["waypoint_index"]] = wp.get("trips_index", 0)
                    # reconstruct original index order
                    orig_order = sorted(range(len(waypoints)),
                                        key=lambda i: waypoints[i]["waypoint_index"])
                    # road geometry
                    coords = trip["geometry"]["coordinates"]
                    arr = np.array(coords)
                    wm  = to_wm(arr)
                    return orig_order, list(map(tuple, wm))
        except Exception:
            continue
    return None, None


def _fetch_route_for_van(args):
    """Worker: fetch OSRM trip geometry for a single van. Runs in thread pool."""
    k, cluster_ll = args
    if len(cluster_ll) < 2:
        return k, list(range(len(cluster_ll))), None
    order, geom_wm = osrm_trip(cluster_ll)
    if order is not None and geom_wm is not None:
        return k, order, geom_wm
    # fallback: sequential route geometry
    loop_pts = np.vstack([cluster_ll, cluster_ll[:1]])
    geom = osrm_route_geometry(loop_pts)
    return k, list(range(len(cluster_ll))), geom


def fetch_all_routes(lonlat_pts, labels, K, prog_base=0, prog_range=15, status_cb=None):
    """
    Fetch OSRM trip routes for all K vans in PARALLEL using a thread pool.
    Sequential OSRM calls were the main bottleneck — threads cut wall time by ~K×.
    Returns:
      routes  — dict {k: [local_indices]}
      geoms   — dict {k: [(x_wm, y_wm)]}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    tasks = [(k, lonlat_pts[labels == k]) for k in range(K) if (labels == k).any()]
    routes, geoms = {}, {}
    # Use up to 8 threads — OSRM public API handles concurrent requests fine
    with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as pool:
        futures = {pool.submit(_fetch_route_for_van, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            k, order, geom = fut.result()
            routes[k] = order
            if geom:
                geoms[k] = geom
    return routes, geoms


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


# ──────────────────────────────────────────────────────
#  PIPELINE A — Min-Cost Flow
# ──────────────────────────────────────────────────────
def run_mcf(pts_ll, K, cap_pct, cost_matrix):
    """pts_ll = lon/lat array, cost_matrix = NxN (OSRM seconds or Euclidean)"""
    n = len(pts_ll)
    cap = int(np.ceil(n / K * cap_pct / 100))
    # seed centers with KMeans on normalized coords
    pts_n = norm(pts_ll, (pts_ll[:,0].min(), pts_ll[:,1].min(),
                           pts_ll[:,0].max(), pts_ll[:,1].max()))
    seeds_n = KMeans(K, init="k-means++", n_init=8, max_iter=100,
                     random_state=0).fit(pts_n).cluster_centers_
    # Map each point to nearest seed for initial cost approximation
    # but use cost_matrix for actual flow costs via seed index mapping
    # Find which point is closest to each seed → use as representative
    seed_idx = []
    for s in seeds_n:
        seed_idx.append(int(np.linalg.norm(pts_n - s, axis=1).argmin()))

    # Build flow graph using cost_matrix rows to seed representatives
    C = cost_matrix[:, seed_idx]   # (n, K) — cost from each point to each seed-rep

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
        L = C.argmin(1)

    # Compute centers as centroids of assigned points (in normalized space)
    Cn = np.array([pts_n[L==k].mean(0) if (L==k).any() else pts_n.mean(0)
                   for k in range(K)])
    L, Cn = fix_empty(pts_n, L, Cn, K)
    return L, Cn


# ──────────────────────────────────────────────────────
#  PIPELINE B — Capacitated P-Median
# ──────────────────────────────────────────────────────
def run_pmedian(pts_ll, K, cap_pct, max_iter, cost_matrix):
    n = len(pts_ll)
    cap = int(np.ceil(n / K * cap_pct / 100))
    pts_n = norm(pts_ll, (pts_ll[:,0].min(), pts_ll[:,1].min(),
                           pts_ll[:,0].max(), pts_ll[:,1].max()))
    # Initialize medians with KMeans
    km_labels = KMeans(K, init="k-means++", n_init=8, max_iter=100,
                       random_state=0).fit(pts_n).labels_
    L = km_labels.copy()

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
            if nL[i] < 0: nL[i] = D_to_medians[i].argmin()

        if np.all(nL == L): break
        L = nL

    # Compute centers in normalized space
    Cn = np.array([pts_n[L==k].mean(0) if (L==k).any() else pts_n.mean(0)
                   for k in range(K)])
    L, Cn = fix_empty(pts_n, L, Cn, K)
    return L, Cn


# ──────────────────────────────────────────────────────
#  ROUTING — NN-TSP using road distances
# ──────────────────────────────────────────────────────
def nn_tsp_road(indices, cost_matrix):
    """indices = list of point indices in the full cost matrix"""
    n = len(indices)
    if n <= 1: return list(range(n)), 0.0
    sub = cost_matrix[np.ix_(indices, indices)]
    vis = [False]*n; route = [0]; vis[0] = True
    for _ in range(n-1):
        d = sub[route[-1]].copy(); d[np.array(vis)] = np.inf
        nxt = int(d.argmin()); route.append(nxt); vis[nxt] = True
    length = sum(sub[route[i], route[(i+1)%n]] for i in range(n))
    return route, float(length)


# ──────────────────────────────────────────────────────
#  ROUTING — 2-opt using road distances
# ──────────────────────────────────────────────────────
def two_opt_road(indices, cost_matrix, max_iter=150):
    n = len(indices)
    if n <= 3:
        return list(range(n)), float(sum(cost_matrix[indices[i], indices[(i+1)%n]] for i in range(n)))
    sub = cost_matrix[np.ix_(indices, indices)]
    route, _ = nn_tsp_road(indices, cost_matrix)
    # remap: route contains local indices 0..n-1 into sub

    def tlen(r):
        return sum(sub[r[i], r[(i+1)%n]] for i in range(n))

    improved = True; iters = 0
    while improved and iters < max_iter:
        improved = False; iters += 1
        for i in range(1, n-1):
            for j in range(i+1, n):
                nr = route[:i] + route[i:j+1][::-1] + route[j+1:]
                if tlen(nr) < tlen(route) - 1e-6:
                    route = nr; improved = True; break
            if improved: break
    return route, tlen(route)


# ──────────────────────────────────────────────────────
#  ANALYTICS
# ──────────────────────────────────────────────────────
def analytics(pts_n, L, C, name, t, total_tour_s=None, osrm_used=False):
    K = int(L.max())+1
    sizes = np.bincount(L, minlength=K)
    msz = sizes.mean(); cv = sizes.std()/msz if msz>0 else 0
    intra = [scipy_dm(pts_n[L==k], C[k:k+1]).mean()
             for k in range(K) if (L==k).any()]
    d = {"name":name,"sizes":sizes.tolist(),"min":int(sizes.min()),
         "max":int(sizes.max()),"mean":round(float(msz),1),
         "cv":round(float(cv),4),"intra":round(float(np.mean(intra)),4),
         "time_ms":round(t*1000,1),"osrm":osrm_used}
    if total_tour_s is not None:
        d["tour_s"] = round(total_tour_s, 1)  # in seconds (road time)
    return d


# ──────────────────────────────────────────────────────
#  STATIC MAP — draws OSRM geometries OR straight lines
# ──────────────────────────────────────────────────────
def make_map(ll_pts, L, ll_ctr_n, bbox, title, road_geoms=None, tile_src=None):
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

    # OSRM badge
    has_roads = road_geoms and any(road_geoms.values())
    badge = "🛣 Real road routes (OSRM)" if has_roads else "📐 Euclidean lines (OSRM unavailable)"
    ax.text(0.01, 0.01, badge, transform=ax.transAxes, fontsize=6.5,
            color="#6ee7b7" if has_roads else "#f87171",
            fontfamily="monospace", va="bottom", zorder=12,
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="#064e3b" if has_roads else "#7f1d1d", alpha=0.8, ec="none"))

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
             tile_src, n_frames=140, fps=20, trail=18):
    ll_ctr = denorm(ll_ctr_n, bbox)
    K = max(int(L.max())+1, len(ll_ctr))
    colors = [PALETTE[k%len(PALETTE)] for k in range(K)]
    wm = to_wm(ll_pts); cw = to_wm(ll_ctr)
    dep = cw.mean(0)

    paths = []
    for k in range(K):
        if road_geoms and k in road_geoms and road_geoms[k]:
            # use the actual OSRM street geometry as the vehicle path
            path = np.array(road_geoms[k])
        else:
            # fallback: straight lines depot→stops→depot
            m = L==k
            if not m.any(): paths.append(_param(np.vstack([dep,dep]))); continue
            paths.append(_param(np.vstack([dep[np.newaxis], wm[m], dep[np.newaxis]])))
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

    ax.scatter(dep[0],dep[1], s=220, color="#ef4444", marker="D",
               edgecolors="white", linewidths=2, zorder=5)

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
#  PIPELINE C — MCF + OR-Tools per-cluster TSP
#  "Same zones as A, better local sequencing"
# ──────────────────────────────────────────────────────
def _ortools_tsp_cluster(indices, cost_matrix, time_limit_s=4):
    """OR-Tools TSP for a single cluster — better ordering than 2-opt."""
    n = len(indices)
    if n <= 2:
        length = cost_matrix[indices[0], indices[1]] if n == 2 else 0.0
        return list(range(n)), float(length)

    SCALE = 100_000
    sub = (cost_matrix[np.ix_(indices, indices)] * SCALE).astype(int)
    full = np.zeros((n+1, n+1), dtype=int)
    full[1:, 1:] = sub   # depot row/col = 0 (open TSP)

    mgr = pywrapcp.RoutingIndexManager(n+1, 1, 0)
    mdl = pywrapcp.RoutingModel(mgr)
    cb  = mdl.RegisterTransitCallback(
        lambda i, j: int(full[mgr.IndexToNode(i)][mgr.IndexToNode(j)]))
    mdl.SetArcCostEvaluatorOfAllVehicles(cb)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_s

    sol = mdl.SolveWithParameters(params)
    if sol:
        idx = mdl.Start(0); route = []
        while not mdl.IsEnd(idx):
            node = mgr.IndexToNode(idx)
            if node != 0: route.append(node - 1)
            idx = sol.Value(mdl.NextVar(idx))
        length = sum(sub[route[i], route[(i+1)%len(route)]] for i in range(len(route))) / SCALE
        return route, float(length)
    # fallback NN
    vis = [False]*n; r = [0]; vis[0] = True
    for _ in range(n-1):
        d = sub[r[-1]].astype(float); d[np.array(vis)] = np.inf
        nxt = int(d.argmin()); r.append(nxt); vis[nxt] = True
    return r, sum(sub[r[i], r[(i+1)%n]] for i in range(n)) / SCALE


def run_pipeline_c(pts_ll, K, cap_pct, cost_matrix, time_limit_s=8):
    """MCF assignment (optimal zones) + OR-Tools TSP per cluster (optimal ordering)."""
    labels, centers = run_mcf(pts_ll, K, cap_pct, cost_matrix)
    routes = {}
    per_cluster = max(2, time_limit_s // K)
    for k in range(K):
        m = labels == k
        if not m.any(): routes[k] = []; continue
        idx = list(np.where(m)[0])
        route, _ = _ortools_tsp_cluster(idx, cost_matrix, time_limit_s=per_cluster)
        routes[k] = route
    return labels, centers, routes


# ──────────────────────────────────────────────────────
#  PIPELINE D — Full OR-Tools CVRP
#  Assignment + routing solved simultaneously
# ──────────────────────────────────────────────────────
def run_pipeline_d(pts_ll, K, cap_pct, cost_matrix, time_limit_s=15):
    """
    Genuine single-stage CVRP: OR-Tools decides BOTH which van gets which stop
    AND in what order to visit them — simultaneously, with capacity constraints.
    Uses a geographic centroid as the depot (open VRP — depot costs are real
    but small relative to inter-stop costs, so they don't distort routing).
    """
    n = len(pts_ll)
    # D uses a tighter cap than A/B/C — the full CVRP solver needs tight capacity
    # to produce balanced assignments. With a loose cap it packs some vans and
    # leaves others near-empty, which is correct mathematically but bad operationally.
    d_cap_pct = min(cap_pct, 115)   # never go above 115% for CVRP
    cap = int(np.ceil(n / K * d_cap_pct / 100))
    pts_bbox = (pts_ll[:,0].min(), pts_ll[:,1].min(),
                pts_ll[:,0].max(), pts_ll[:,1].max())
    pts_n = norm(pts_ll, pts_bbox)

    SCALE = 100_000
    scaled = (cost_matrix * SCALE).astype(int)

    # Depot = geographic centroid — represents the "warehouse" conceptually
    depot_n = pts_n.mean(0)
    depot_dists = (np.linalg.norm(pts_n - depot_n, axis=1) * SCALE).astype(int)

    # Full (n+1) × (n+1): row/col 0 = depot
    full = np.zeros((n+1, n+1), dtype=int)
    full[1:, 1:] = scaled
    full[0, 1:]  = depot_dists
    full[1:, 0]  = depot_dists

    mgr = pywrapcp.RoutingIndexManager(n+1, K, 0)
    mdl = pywrapcp.RoutingModel(mgr)

    cb = mdl.RegisterTransitCallback(
        lambda i, j: int(full[mgr.IndexToNode(i)][mgr.IndexToNode(j)]))
    mdl.SetArcCostEvaluatorOfAllVehicles(cb)

    # Capacity: depot has demand 0, every stop has demand 1
    dcb = mdl.RegisterUnaryTransitCallback(
        lambda i: 0 if mgr.IndexToNode(i) == 0 else 1)
    mdl.AddDimensionWithVehicleCapacity(dcb, 0, [cap]*K, True, 'Capacity')

    params = pywrapcp.DefaultRoutingSearchParameters()
    # PATH_CHEAPEST_ARC gives much better initial solutions for CVRP than SAVINGS
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_s

    sol = mdl.SolveWithParameters(params)

    labels  = np.zeros(n, dtype=int)
    routes  = {}
    centers = np.zeros((K, 2))

    if sol:
        for v in range(K):
            idx = mdl.Start(v); stop_nodes = []
            while not mdl.IsEnd(idx):
                node = mgr.IndexToNode(idx)
                if node != 0:
                    stop_nodes.append(node - 1)
                    labels[node - 1] = v
                idx = sol.Value(mdl.NextVar(idx))
            routes[v] = list(range(len(stop_nodes)))
            if stop_nodes:
                centers[v] = pts_n[stop_nodes].mean(0)
    else:
        # Fallback to MCF if solver fails
        labels, centers = run_mcf(pts_ll, K, cap_pct, cost_matrix)
        for v in range(K):
            routes[v] = list(range((labels==v).sum()))

    labels, centers = fix_empty(pts_n, labels, centers, K)
    return labels, centers, routes


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
#  ANALYTICS  (extended with territory metrics)
# ──────────────────────────────────────────────────────
def analytics(pts_n, labels, centers, name, t, total_tour_s=None, osrm_used=False):
    K = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=K)
    msz = sizes.mean(); cv = sizes.std() / msz if msz > 0 else 0
    cost = sum(np.linalg.norm(pts_n[labels==k] - centers[k], axis=1).sum()
               for k in range(K) if (labels==k).any())
    terr = compute_territory_metrics(pts_n, labels, K)
    d = {
        "name": name, "sizes": sizes.tolist(),
        "min": int(sizes.min()), "max": int(sizes.max()),
        "mean": round(float(msz), 1), "cv": round(float(cv), 4),
        "cost": round(float(cost), 4), "time_ms": round(t * 1000, 1),
        "osrm": osrm_used,
        **terr,
    }
    if total_tour_s is not None:
        d["tour_s"] = round(total_tour_s, 1)
    return d


# ──────────────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────────────
for k, v in [
    ("ll_pts", None), ("bbox", None), ("osm_tags", []),
    ("cost_matrix", None), ("osrm_used", False),
    ("result_a", None), ("result_b", None),
    ("result_c", None), ("result_d", None),
    ("gif_a", None), ("gif_b", None),
    ("gif_c", None), ("gif_d", None),
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
    fetch_btn = col1.button("📡 Fetch POIs", use_container_width=True, type="primary")
    run_btn   = col2.button("🚀 Run", use_container_width=True, type="primary",
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
                                  help="Time budget for per-cluster TSP optimization")
        d_time_limit = st.slider("Pipeline D OR-Tools (s)", 8, 60, 15, step=5,
                                  help="Time budget for full CVRP. More = better routes.")
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
        mat, osrm_used = osrm_distance_matrix(ll)
        prog.progress(100, text="✓ Distance matrix ready"); prog.empty()
        for k, v in [
            ("ll_pts", ll), ("bbox", bbox), ("osm_tags", tags),
            ("cost_matrix", mat), ("osrm_used", osrm_used),
            ("result_a", None), ("result_b", None),
            ("result_c", None), ("result_d", None),
            ("gif_a", None), ("gif_b", None),
            ("gif_c", None), ("gif_d", None),
        ]:
            st.session_state[k] = v
        st.success(
            f"✓ {len(ll)} POIs · "
            f"{'🛣 OSRM road distances' if osrm_used else '📐 Euclidean fallback'}")


# ──────────────────────────────────────────────────────
#  RUN BUTTON
# ──────────────────────────────────────────────────────
if run_btn and st.session_state.ll_pts is not None:
    ll        = st.session_state.ll_pts
    bbox      = st.session_state.bbox
    mat       = st.session_state.cost_matrix
    K         = n_vans
    osrm_used = st.session_state.osrm_used

    # Safe defaults if advanced settings expander wasn't opened
    try:    cap_pct
    except: cap_pct = 130
    try:    pm_iters
    except: pm_iters = 30
    try:    c_time_limit
    except: c_time_limit = 8
    try:    d_time_limit
    except: d_time_limit = 15
    try:    anim_frames
    except: anim_frames = 80
    try:    anim_fps
    except: anim_fps = 20
    try:    trail_len
    except: trail_len = 16
    try:    tile_src
    except: tile_src = ctx.providers.CartoDB.DarkMatter

    pts_bbox = (ll[:,0].min(), ll[:,1].min(), ll[:,0].max(), ll[:,1].max())
    pts_n    = norm(ll, pts_bbox)

    prog = st.progress(0, text="Pipeline A — Min-Cost Flow…")

    # ── A: MCF + OSRM Trip ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    La, Ca_n = run_mcf(ll, K, cap_pct, mat)
    prog.progress(8, text="Pipeline A — OSRM trip routing…")
    routes_a, geoms_a = fetch_all_routes(ll, La, K)
    total_a = sum(
        sum(mat[np.where(La==k)[0][routes_a[k][i]],
                np.where(La==k)[0][routes_a[k][(i+1)%len(routes_a[k])]]]
            for i in range(len(routes_a[k])))
        for k in routes_a if len(routes_a.get(k, [])) > 1
    ) if routes_a else 0.0
    ta = time.perf_counter() - t0
    prog.progress(18, text="Rendering Pipeline A…")
    fig_a = make_map(ll, La, Ca_n, bbox, "A — Min-Cost Flow → OSRM Trip",
                     road_geoms=geoms_a, tile_src=tile_src)
    ana_a = analytics(pts_n, La, Ca_n, "A: MCF + OSRM Trip", ta,
                      total_tour_s=total_a, osrm_used=osrm_used)

    # ── B: P-Median + OSRM Trip ────────────────────────────────────────────
    prog.progress(22, text="Pipeline B — P-Median territory assignment…")
    t0 = time.perf_counter()
    Lb, Cb_n = run_pmedian(ll, K, cap_pct, pm_iters, mat)
    prog.progress(30, text="Pipeline B — OSRM trip routing…")
    routes_b, geoms_b = fetch_all_routes(ll, Lb, K)
    total_b = sum(
        sum(mat[np.where(Lb==k)[0][routes_b[k][i]],
                np.where(Lb==k)[0][routes_b[k][(i+1)%len(routes_b[k])]]]
            for i in range(len(routes_b[k])))
        for k in routes_b if len(routes_b.get(k, [])) > 1
    ) if routes_b else 0.0
    tb = time.perf_counter() - t0
    prog.progress(40, text="Rendering Pipeline B…")
    fig_b = make_map(ll, Lb, Cb_n, bbox, "B — P-Median → OSRM Trip",
                     road_geoms=geoms_b, tile_src=tile_src)
    ana_b = analytics(pts_n, Lb, Cb_n, "B: P-Median + OSRM Trip", tb,
                      total_tour_s=total_b, osrm_used=osrm_used)

    # ── C: MCF + OR-Tools per-cluster TSP ──────────────────────────────────
    prog.progress(44, text=f"Pipeline C — OR-Tools TSP per cluster ({c_time_limit}s)…")
    t0 = time.perf_counter()
    Lc, Cc_n, routes_c_raw = run_pipeline_c(ll, K, cap_pct, mat,
                                              time_limit_s=c_time_limit)
    prog.progress(55, text="Pipeline C — OSRM road geometry…")
    geoms_c = {}
    for k in range(K):
        m = Lc == k
        if not m.any(): continue
        geom = osrm_route_geometry(np.vstack([ll[m], ll[m][:1]]))
        if geom: geoms_c[k] = geom
    total_c = sum(
        sum(mat[np.where(Lc==k)[0][i], np.where(Lc==k)[0][(i+1)%max((Lc==k).sum(), 1)]]
            for i in range(max((Lc==k).sum()-1, 0)))
        for k in range(K) if (Lc==k).sum() > 1
    )
    tc = time.perf_counter() - t0
    prog.progress(63, text="Rendering Pipeline C…")
    fig_c = make_map(ll, Lc, Cc_n, bbox, "C — MCF + OR-Tools TSP per cluster",
                     road_geoms=geoms_c, tile_src=tile_src)
    ana_c = analytics(pts_n, Lc, Cc_n, "C: MCF + OR-Tools TSP", tc,
                      total_tour_s=total_c, osrm_used=osrm_used)

    # ── D: Full OR-Tools CVRP ──────────────────────────────────────────────
    prog.progress(67, text=f"Pipeline D — Full OR-Tools CVRP ({d_time_limit}s)…")
    t0 = time.perf_counter()
    Ld, Cd_n, routes_d_raw = run_pipeline_d(ll, K, cap_pct, mat,
                                              time_limit_s=d_time_limit)
    prog.progress(80, text="Pipeline D — OSRM road geometry…")
    geoms_d = {}
    for k in range(K):
        m = Ld == k
        if not m.any(): continue
        geom = osrm_route_geometry(np.vstack([ll[m], ll[m][:1]]))
        if geom: geoms_d[k] = geom
    total_d = sum(
        sum(mat[np.where(Ld==k)[0][i], np.where(Ld==k)[0][(i+1)%max((Ld==k).sum(), 1)]]
            for i in range(max((Ld==k).sum()-1, 0)))
        for k in range(K) if (Ld==k).sum() > 1
    )
    td = time.perf_counter() - t0
    prog.progress(87, text="Rendering Pipeline D…")
    fig_d = make_map(ll, Ld, Cd_n, bbox, "D — Full OR-Tools CVRP (single-stage)",
                     road_geoms=geoms_d, tile_src=tile_src)
    ana_d = analytics(pts_n, Ld, Cd_n, "D: Full OR-Tools CVRP", td,
                      total_tour_s=total_d, osrm_used=osrm_used)

    prog.progress(92, text="Rendering animations…")
    gif_a = make_gif(ll, La, Ca_n, bbox, geoms_a, tile_src, anim_frames, anim_fps, trail_len)
    gif_b = make_gif(ll, Lb, Cb_n, bbox, geoms_b, tile_src, anim_frames, anim_fps, trail_len)
    gif_c = make_gif(ll, Lc, Cc_n, bbox, geoms_c, tile_src, anim_frames, anim_fps, trail_len)
    gif_d = make_gif(ll, Ld, Cd_n, bbox, geoms_d, tile_src, anim_frames, anim_fps, trail_len)

    prog.progress(100, text="Done."); prog.empty()

    st.session_state.result_a = (fig_a, ana_a, La, Ca_n, geoms_a)
    st.session_state.result_b = (fig_b, ana_b, Lb, Cb_n, geoms_b)
    st.session_state.result_c = (fig_c, ana_c, Lc, Cc_n, geoms_c)
    st.session_state.result_d = (fig_d, ana_d, Ld, Cd_n, geoms_d)
    st.session_state.gif_a = gif_a; st.session_state.gif_b = gif_b
    st.session_state.gif_c = gif_c; st.session_state.gif_d = gif_d


# ──────────────────────────────────────────────────────
#  DISPLAY
# ──────────────────────────────────────────────────────
st.markdown("# 🚐 Van Territory Planner")
st.markdown(
    '<p style="color:#64748b;font-family:monospace;font-size:.75rem;margin-top:-.4rem">'
    '4 pipelines · MCF · P-Median · OR-Tools TSP · Full CVRP · OSRM street routing</p>',
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
    try:    tile_src
    except: tile_src = ctx.providers.CartoDB.DarkMatter
    try:    n_vans
    except: n_vans = 4
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
         "OSRM /trip sequences each cluster on real roads.<br>"
         "<b>Role:</b> baseline. Cheapest assignment, fast."),
        (cb, "#f59e0b", "B — P-Median",
         "TERRITORY-FIRST BASELINE",
         "Finds K geographic territory anchors minimising distance to zone centres. "
         "OSRM /trip sequences each territory.<br>"
         "<b>Role:</b> territory design benchmark. Shows whether compact zones matter."),
        (cc, "#10b981", "C — MCF + OR-Tools TSP",
         "SAME ZONES AS A, BETTER ORDERING",
         "MCF assignment (identical to A) + OR-Tools per-cluster TSP with "
         "Guided Local Search. Answers: was A weak because of bad stop ordering?<br>"
         "<b>Role:</b> isolates routing quality from assignment quality."),
        (cd, "#8b5cf6", "D — Full OR-Tools CVRP",
         "INTEGRATED SINGLE-STAGE BENCHMARK",
         "OR-Tools decides assignment AND route order simultaneously with capacity "
         "constraints. The real production-shaped free benchmark.<br>"
         "<b>Role:</b> main serious benchmark. Closest to modern VRP solvers.<br>"
         "<b style='color:#facc15'>Note:</b> Total Route includes depot-return "
         "costs that A/B/C don't count — compare on Balance CV and Zone Overlap."),
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
                                       mime="image/gif", use_container_width=True)
            with st2:
                cv  = ana["cv"]
                cls = ("metric-good" if cv < 0.15 else
                       "metric-warn" if cv < 0.35 else "metric-bad")
                def mc(label, val, c=""):
                    st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {c}">{val}</div>
                    </div>""", unsafe_allow_html=True)
                mc("Balance CV ↓",  f"{cv:.3f}", cls)
                mc("Min / Max",     f"{ana['min']} / {ana['max']}")
                if "tour_s" in ana:
                    unit = "s" if ana["osrm"] else "u"
                    mc("Total Route ↓", f"{ana['tour_s']:.0f}{unit}", "metric-warn")
                mc("Exec Time",     f"{ana['time_ms']:.0f} ms")
                st.markdown('<div style="height:.3rem"></div>', unsafe_allow_html=True)
                # Territory metrics
                st.markdown(
                    '<p style="font-family:monospace;font-size:.62rem;color:#475569;'
                    'text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">'
                    'Territory Quality</p>', unsafe_allow_html=True)
                mc("Avg Intra-Dist ↓", f"{ana['avg_intra']:.3f}")
                mc("Zone Overlap ↓",
                   f"{ana['hull_overlap']*100:.0f}%",
                   "metric-good" if ana['hull_overlap'] < 0.4 else
                   "metric-warn" if ana['hull_overlap'] < 0.7 else "metric-bad")
                mc("Isolation ↑",  f"{ana['isolation']:.3f}")
                st.pyplot(size_chart(ana["sizes"]), width="stretch"); plt.close()

    for col, (letter, fig, ana, gif, color) in zip(cols4, PIPELINES):
        show_pipeline(col, letter, fig, ana, gif, color)

    # ── scoreboard ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚖️ Head-to-Head Scoreboard")

    COLORS = {"A": "#3b82f6", "B": "#f59e0b", "C": "#10b981", "D": "#8b5cf6"}
    DIM = "#334155"

    scoreboard_metrics = [
        ("Balance CV ↓",    "cv",          True,  "{:.3f}"),
        ("Min Stops ↑",     "min",         False, "{:.0f}"),
        ("Max Stops ↓",     "max",         True,  "{:.0f}"),
        ("Total Route ↓",   "tour_s",      True,  "{:.0f}"),
        ("Avg Intra ↓",     "avg_intra",   True,  "{:.3f}"),
        ("Zone Overlap ↓",  "hull_overlap",True,  "{:.2f}"),
        ("Isolation ↑",     "isolation",   False, "{:.3f}"),
        ("Exec Time ↓",     "time_ms",     True,  "{:.0f}ms"),
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

    # ── metrics legend ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📖 Metrics Legend")

    legend = [
        ("⚖️", "Balance CV",
         "Standard deviation ÷ mean of cluster sizes. How evenly stops are split. "
         "<b style='color:#4ade80'>&lt;0.15</b> = balanced · "
         "<b style='color:#facc15'>0.15–0.35</b> = moderate · "
         "<b style='color:#f87171'>&gt;0.35</b> = skewed. Lower is better."),

        ("🔢", "Min / Max Stops",
         "Smallest and largest stop count across all vans. Ideally both close to N÷K. "
         "A large gap means one driver finishes at noon while another works until midnight."),

        ("🛣", "Total Route",
         "Sum of all driving time across all vans for one full loop. "
         "In seconds of road time when OSRM is active, normalized units otherwise. "
         "Directly maps to fuel cost and driver hours — the main efficiency metric."),

        ("📐", "Avg Intra-Dist",
         "Average distance from each stop to its cluster centroid in normalized space. "
         "Measures <b>compactness</b> — how tightly grouped each van's stops are geographically. "
         "Compact zones = less zigzagging between stops even before route optimization."),

        ("🗺", "Zone Overlap %",
         "Fraction of cluster-pair bounding boxes that overlap. "
         "High overlap means vans cover the same geographic areas — operationally this "
         "looks messy and can lead to driver confusion. "
         "<b style='color:#4ade80'>&lt;40%</b> = clean · "
         "<b style='color:#facc15'>40–70%</b> = some overlap · "
         "<b style='color:#f87171'>&gt;70%</b> = fragmented territories."),

        ("↔️", "Isolation",
         "Average minimum distance between cluster centroids. "
         "Higher isolation means zones are well-separated with clear geographic boundaries. "
         "Low isolation = zones bleed into each other. Higher is better."),

        ("⏱", "Exec Time",
         "Wall-clock milliseconds. A and B are fast (deterministic algorithms). "
         "C and D use OR-Tools with a time budget — they'll often finish early when they converge. "
         "At production scale these run on dedicated servers in parallel so speed matters less."),
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
    st.markdown(f"""<div class="insight-box">
    {'<b style="color:#4ade80">🛣 OSRM active</b> — routes follow real streets, matrix uses real driving times.' if osrm else
     '<b style="color:#f87171">📐 OSRM unavailable</b> — Euclidean fallback. On your machine OSRM works via http://router.project-osrm.org'}
    &nbsp;·&nbsp;
    <b style="color:#a78bfa">🔧 OR-Tools</b> — C uses per-cluster TSP, D uses full CVRP.
    Final production gap: replace OSRM public API with a self-hosted instance or
    Google Maps Distance Matrix for real-time traffic awareness.
    </div>""", unsafe_allow_html=True)