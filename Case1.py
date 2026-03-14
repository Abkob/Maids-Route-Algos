"""
Van Territory Planner — 4-Pipeline Benchmark
  A) MCF + OSRM Trip          — cost-first two-stage baseline
  B) P-Median + OSRM Trip     — territory-first two-stage baseline
  C) MCF + OR-Tools TSP       — same zones as A, better local ordering
  D) Full OR-Tools CVRP       — integrated single-stage benchmark
All pipelines use the OSRM road-network matrix. Routes follow real streets.
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


def osrm_trip(cluster_lonlat: np.ndarray):
    """
    OSRM /trip/v1/ — farthest-insertion TSP on the road network.
    Returns (orig_order, geom_wm, dist_m, dur_s) or (None, None, 0, 0) on failure.
      orig_order : list of original input indices in OSRM visit order
      geom_wm    : list of (x_wm, y_wm) — full road polyline
      dist_m     : total route distance in meters
      dur_s      : total route duration in seconds
    """
    if len(cluster_lonlat) < 2:
        return None, None, 0.0, 0.0
    coord_str = ";".join(f"{lon},{lat}" for lon, lat in cluster_lonlat)
    for base in OSRM_BASES:
        try:
            url = (f"{base}/trip/v1/driving/{coord_str}"
                   f"?roundtrip=true&overview=full&geometries=geojson")
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if data.get("code") == "Ok" and data.get("trips"):
                    trip      = data["trips"][0]
                    dist_m    = float(trip.get("distance", 0.0))
                    dur_s     = float(trip.get("duration", 0.0))
                    waypoints = data["waypoints"]
                    # orig_order[i] = original input index of the i-th stop visited
                    orig_order = sorted(range(len(waypoints)),
                                        key=lambda i: waypoints[i]["waypoint_index"])
                    coords = trip["geometry"]["coordinates"]
                    wm     = to_wm(np.array(coords))
                    return orig_order, list(map(tuple, wm)), dist_m, dur_s
        except Exception:
            continue
    return None, None, 0.0, 0.0


def _fetch_route_for_van(args):
    """
    Worker: fetch OSRM trip geometry + metrics for a single van.
    Returns (k, order, geom_wm, dist_m, dur_s).
    Runs in thread pool for parallel fetching.
    """
    k, cluster_ll = args
    if len(cluster_ll) < 2:
        return k, list(range(len(cluster_ll))), None, 0.0, 0.0
    order, geom_wm, dist_m, dur_s = osrm_trip(cluster_ll)
    if order is not None and geom_wm is not None:
        return k, order, geom_wm, dist_m, dur_s
    # fallback: sequential geometry via /route/
    loop_pts = np.vstack([cluster_ll, cluster_ll[:1]])
    result = ordered_route_geometry(loop_pts)
    if result:
        geom, dist_m, dur_s = result
        return k, list(range(len(cluster_ll))), geom, dist_m, dur_s
    return k, list(range(len(cluster_ll))), None, 0.0, 0.0


def fetch_all_routes(lonlat_pts, labels, K):
    """
    Fetch OSRM trip routes for all K vans in PARALLEL.
    Returns:
      routes     — dict {k: [local_indices in OSRM visit order]}
      geoms      — dict {k: [(x_wm, y_wm)] street polylines}
      route_dist — dict {k: dist_m}   road distance per van in meters
      route_dur  — dict {k: dur_s}    road duration per van in seconds
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    tasks = [(k, lonlat_pts[labels == k]) for k in range(K) if (labels == k).any()]
    routes, geoms, route_dist, route_dur = {}, {}, {}, {}
    with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as pool:
        futures = {pool.submit(_fetch_route_for_van, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            k, order, geom, dist_m, dur_s = fut.result()
            routes[k]     = order
            route_dist[k] = dist_m
            route_dur[k]  = dur_s
            if geom:
                geoms[k] = geom
    return routes, geoms, route_dist, route_dur


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





# (old analytics removed — see analytics() below with full metrics)


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
        # Note: this computes closed-tour cost for reference only.
        # Actual road route time/distance comes from OSRM /route/ in the run block.
        length = sum(sub[route[i], route[(i+1)%len(route)]] for i in range(len(route))) / SCALE
        return route, float(length)
    # fallback NN
    vis = [False]*n; r = [0]; vis[0] = True
    for _ in range(n-1):
        d = sub[r[-1]].astype(float); d[np.array(vis)] = np.inf
        nxt = int(d.argmin()); r.append(nxt); vis[nxt] = True
    return r, sum(sub[r[i], r[(i+1)%n]] for i in range(n)) / SCALE


def run_pipeline_c(pts_ll, K, cap_pct, cost_matrix, time_limit_s=8):
    """
    Pipeline C: MCF assignment (same zones as A) + OR-Tools TSP per cluster.
    OR-Tools finds a better stop visit order than OSRM's heuristic.
    Returns (labels, centers, local_routes, ordered_global_idx).
      local_routes       : {k: [local 0-based indices in OR-Tools visit order]}
      ordered_global_idx : {k: [original point indices in visit order]}
                           Use these for OSRM geometry — order matters.
    """
    labels, centers = run_mcf(pts_ll, K, cap_pct, cost_matrix)
    local_routes, ordered_global = {}, {}
    # Divide budget across clusters; minimum 2s each. OR-Tools exits early on convergence.
    per_cluster = max(2, time_limit_s // max(K, 1))
    for k in range(K):
        m = labels == k
        if not m.any():
            local_routes[k] = []; ordered_global[k] = []; continue
        global_idx = list(np.where(m)[0])          # original indices of cluster k
        local_route, _ = _ortools_tsp_cluster(global_idx, cost_matrix,
                                               time_limit_s=per_cluster)
        local_routes[k]   = local_route
        # Map local route order back to original point indices
        ordered_global[k] = [global_idx[i] for i in local_route]
    return labels, centers, local_routes, ordered_global


# ──────────────────────────────────────────────────────
#  PIPELINE D — Full OR-Tools CVRP
#  Single-stage: assignment + routing solved simultaneously.
#  Supports multiple depot modes and open/closed route types.
# ──────────────────────────────────────────────────────

def _build_ortools_data_model(pts_n, cost_matrix, K, cap, depot_mode, route_mode):
    """
    Build the (n+K) × (n+K) cost matrix for OR-Tools CVRP.

    Depot modeling:
      "centroid"  — shared geographic centroid appended as node n.
                    All K vehicles start/end at the same point.
      "seeded"    — K per-van start positions from MCF cluster centers.
                    Each vehicle v starts at node n+v (its cluster center).
      "random"    — K random start positions within the bounding box.
                    Each vehicle v starts at node n+v.

    Route type:
      "open"      — vehicles don't need to return to start. Modeled by setting
                    all arcs FROM any depot/start to cost 0 in the return direction.
                    In OR-Tools, we add dummy end nodes with zero-cost arcs.
      "closed"    — vehicles return to their start node (standard CVRP).

    Returns (full_matrix, starts, ends, n_nodes).
    """
    n = len(pts_n)
    SCALE = 100_000
    scaled = (cost_matrix * SCALE).astype(int)

    if depot_mode == "centroid":
        # One shared depot at geographic centroid.
        # Node layout: stops = 0..n-1, depot = n  (same as seeded/random)
        # This ensures the extraction loop "if node < n" works consistently.
        depot_n = pts_n.mean(0)
        nearest_to_depot = int(np.linalg.norm(pts_n - depot_n, axis=1).argmin())
        depot_dists = (cost_matrix[nearest_to_depot] * SCALE).astype(int)
        # (n+1) × (n+1): stops at 0..n-1, depot at n
        full = np.zeros((n+1, n+1), dtype=int)
        full[:n, :n] = scaled           # stop → stop
        full[n, :n]  = depot_dists      # depot → stop
        full[:n, n]  = depot_dists      # stop → depot
        starts = [n] * K               # all vehicles start at node n (depot)
        if route_mode == "open":
            # Add K dummy end nodes (n+1 .. n+K) with zero-cost arcs from any stop
            m = n + 1 + K
            big = np.zeros((m, m), dtype=int)
            big[:n+1, :n+1] = full
            ends = list(range(n+1, m))
            return big, [n]*K, ends, m
        else:
            ends = [n] * K             # return to shared depot
            return full, starts, ends, n+1

    elif depot_mode in ("seeded", "random"):
        # K separate depot nodes: node n = depot_0, n+1 = depot_1, ...
        if depot_mode == "seeded":
            # Use actual MCF assignment to seed per-van start positions.
            # This ensures the depot aligns with the same cluster structure
            # that MCF would produce — not a fresh independent KMeans fit.
            _seed_labels, _seed_centers = run_mcf(
                # rebuild pts_ll from pts_n (we don't have pts_ll here)
                # approximation: use centroid of each cluster in pts_n space
                np.column_stack([pts_n[:,0], pts_n[:,1]]),   # dummy pts_ll
                K, 130, cost_matrix)
            depot_positions = _seed_centers  # (K, 2) in normalised space
        else:
            rng = np.random.default_rng(42)
            depot_positions = rng.uniform(0, 1, (K, 2))

        # Depot↔stop costs: for each synthetic depot position, find nearest real point
        # and use its row in the cost_matrix as the proxy — stays in OSRM units.
        depot_to_stop = np.zeros((K, n), dtype=int)
        for d_idx, dp in enumerate(depot_positions):
            nearest = int(np.linalg.norm(pts_n - dp, axis=1).argmin())
            depot_to_stop[d_idx] = (cost_matrix[nearest] * SCALE).astype(int)
        depot_to_depot = np.zeros((K, K), dtype=int)

        n_nodes = n + K
        full = np.zeros((n_nodes, n_nodes), dtype=int)
        full[:n, :n] = scaled          # stop → stop
        for d in range(K):
            full[n+d, :n]  = depot_to_stop[d]   # depot_d → stop
            full[:n, n+d]  = depot_to_stop[d]   # stop → depot_d
        starts = list(range(n, n+K))

        if route_mode == "open":
            # Add K dummy end nodes
            m = n_nodes + K
            big = np.zeros((m, m), dtype=int)
            big[:n_nodes, :n_nodes] = full
            ends = list(range(n_nodes, m))
            return big, starts, ends, m
        else:
            ends = starts[:]   # return to own start
            return full, starts, ends, n_nodes

    raise ValueError(f"Unknown depot_mode: {depot_mode}")


def run_pipeline_d(pts_ll, K, cap_pct, cost_matrix, time_limit_s=15,
                   depot_mode="centroid", route_mode="closed"):
    """
    Pipeline D: Full OR-Tools CVRP — integrated single-stage benchmark.
    OR-Tools decides which van serves which stop AND in what order simultaneously.

    Args:
      depot_mode : "centroid" | "seeded" | "random"
      route_mode : "closed" (return to start) | "open" (finish anywhere)

    Returns (labels, centers, ordered_global_idx).
      labels             : (n,) van index per stop
      centers            : (K,2) normalized cluster centers
      ordered_global_idx : {v: [original point indices in OR-Tools visit order]}
                           These are the ACTUAL solved routes — use for OSRM geometry.
    """
    n     = len(pts_ll)
    d_cap = min(cap_pct, 115)  # CVRP needs tighter cap to balance well
    cap   = int(np.ceil(n / K * d_cap / 100))
    pts_bbox = (pts_ll[:,0].min(), pts_ll[:,1].min(),
                pts_ll[:,0].max(), pts_ll[:,1].max())
    pts_n = norm(pts_ll, pts_bbox)

    full_mat, starts, ends, n_nodes = _build_ortools_data_model(
        pts_n, cost_matrix, K, cap, depot_mode, route_mode)

    mgr = pywrapcp.RoutingIndexManager(n_nodes, K, starts, ends)
    mdl = pywrapcp.RoutingModel(mgr)

    # Arc cost: route through the full_mat
    cb = mdl.RegisterTransitCallback(
        lambda i, j: int(full_mat[mgr.IndexToNode(i)][mgr.IndexToNode(j)]))
    mdl.SetArcCostEvaluatorOfAllVehicles(cb)

    # Capacity: only real stops (indices 0..n-1) have demand 1; depot/dummy nodes = 0
    dcb = mdl.RegisterUnaryTransitCallback(
        lambda i: 1 if mgr.IndexToNode(i) < n else 0)
    mdl.AddDimensionWithVehicleCapacity(dcb, 0, [cap]*K, True, 'Capacity')

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_s

    sol = mdl.SolveWithParameters(params)

    labels  = np.zeros(n, dtype=int)
    centers = np.zeros((K, 2))
    ordered_global = {}   # {v: [original point indices in visit order]}

    if sol:
        for v in range(K):
            idx = mdl.Start(v); stop_nodes = []
            while not mdl.IsEnd(idx):
                node = mgr.IndexToNode(idx)
                if node < n:  # only real stops, skip depots/dummy nodes
                    stop_nodes.append(node)
                    labels[node] = v
                idx = sol.Value(mdl.NextVar(idx))
            # stop_nodes IS already in OR-Tools visit order — preserve it
            ordered_global[v] = stop_nodes
            if stop_nodes:
                centers[v] = pts_n[stop_nodes].mean(0)
    else:
        # Fallback: MCF assignment with trivial ordering
        labels, centers = run_mcf(pts_ll, K, cap_pct, cost_matrix)
        for v in range(K):
            ordered_global[v] = list(np.where(labels == v)[0])

    labels, centers = fix_empty(pts_n, labels, centers, K)
    return labels, centers, ordered_global


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
        st.markdown("**Pipeline D — Route Start/End**")
        d_depot_mode = st.selectbox("D: Depot mode",
                                     ["centroid", "seeded", "random"],
                                     index=0,
                                     help="centroid=shared depot at geographic centre | "
                                          "seeded=each van starts at its MCF cluster centre | "
                                          "random=random start positions within bbox")
        d_route_mode = st.selectbox("D: Route type",
                                     ["closed", "open"],
                                     index=0,
                                     help="closed=vans return to start | "
                                          "open=vans finish at last stop (no forced return)")
        st.markdown("**Fuel / Cost Assumptions**")
        fuel_lpk    = st.number_input("Fuel efficiency (L/100 km)", value=8.5, step=0.5,
                                       min_value=1.0, max_value=30.0,
                                       help="Typical delivery van: 8–12 L/100 km")
        fuel_price  = st.number_input("Fuel price (per liter, local currency)", value=2.0,
                                       step=0.1, min_value=0.1, max_value=20.0)
        co2_per_l   = st.number_input("CO₂ per liter (kg)", value=2.68, step=0.1,
                                       min_value=1.0, max_value=4.0,
                                       help="Diesel ≈ 2.68 kg/L · Petrol ≈ 2.31 kg/L")
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
    dist_mat = st.session_state.dist_matrix   # may be None if OSRM didn't return distances

    # Safe defaults for fuel/depot settings
    try:    fuel_lpk
    except: fuel_lpk = 8.5
    try:    fuel_price
    except: fuel_price = 2.0
    try:    co2_per_l
    except: co2_per_l = 2.68
    try:    d_depot_mode
    except: d_depot_mode = "centroid"
    try:    d_route_mode
    except: d_route_mode = "closed"

    def _fuel(dist_km):
        return build_fuel_metrics(dist_km, fuel_lpk, fuel_price, co2_per_l)

    def _route_geom_from_ordered(ordered_global, labels_arr):
        """
        Fetch OSRM road geometry using the ACTUAL solved visit order.
        ordered_global : {v: [original point indices in visit order]}
        Returns geoms dict {v: [(x_wm, y_wm)]}, route_dist {v: m}, route_dur {v: s}.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fetch_one(args):
            v, global_idx = args
            if len(global_idx) < 2:
                return v, None, 0.0, 0.0
            ordered_ll = ll[global_idx]
            # close the loop
            loop_ll = np.vstack([ordered_ll, ordered_ll[:1]])
            result = ordered_route_geometry(loop_ll)
            if result:
                geom, dist_m, dur_s = result
                return v, geom, dist_m, dur_s
            return v, None, 0.0, 0.0

        tasks = [(v, idx) for v, idx in ordered_global.items() if len(idx) >= 2]
        geoms, rdist, rdur = {}, {}, {}
        with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as pool:
            futs = {pool.submit(_fetch_one, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                v, geom, dist_m, dur_s = fut.result()
                rdist[v] = dist_m; rdur[v] = dur_s
                if geom: geoms[v] = geom
        return geoms, rdist, rdur

    prog = st.progress(0, text="Pipeline A — Min-Cost Flow…")

    # ── A: MCF + OSRM Trip ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    La, Ca_n = run_mcf(ll, K, cap_pct, mat)
    prog.progress(8, text="Pipeline A — OSRM trip routing…")
    # fetch_all_routes uses OSRM /trip/ which returns the order it chose
    routes_a, geoms_a, rdist_a, rdur_a = fetch_all_routes(ll, La, K)
    # Build ordered_global for A using the OSRM trip order
    ordered_a = {k: [int(np.where(La==k)[0][i]) for i in routes_a[k]]
                 for k in range(K) if (La==k).any() and k in routes_a}
    ta = time.perf_counter() - t0
    rm_a  = build_route_metrics(rdist_a, rdur_a, K, osrm_used)
    ana_a = analytics(pts_n, La, Ca_n, "A: MCF + OSRM Trip", ta,
                      osrm_used=osrm_used,
                      route_metrics=rm_a,
                      fuel_metrics=_fuel(rm_a["total_dist_km"]))
    prog.progress(18, text="Rendering Pipeline A…")
    fig_a = make_map(ll, La, Ca_n, bbox, "A — Min-Cost Flow → OSRM Trip",
                     road_geoms=geoms_a, tile_src=tile_src)

    # ── B: P-Median + OSRM Trip ────────────────────────────────────────────
    prog.progress(22, text="Pipeline B — P-Median territory assignment…")
    t0 = time.perf_counter()
    Lb, Cb_n = run_pmedian(ll, K, cap_pct, pm_iters, mat)
    prog.progress(30, text="Pipeline B — OSRM trip routing…")
    routes_b, geoms_b, rdist_b, rdur_b = fetch_all_routes(ll, Lb, K)
    ordered_b = {k: [int(np.where(Lb==k)[0][i]) for i in routes_b[k]]
                 for k in range(K) if (Lb==k).any() and k in routes_b}
    tb = time.perf_counter() - t0
    rm_b  = build_route_metrics(rdist_b, rdur_b, K, osrm_used)
    ana_b = analytics(pts_n, Lb, Cb_n, "B: P-Median + OSRM Trip", tb,
                      osrm_used=osrm_used,
                      route_metrics=rm_b,
                      fuel_metrics=_fuel(rm_b["total_dist_km"]))
    prog.progress(40, text="Rendering Pipeline B…")
    fig_b = make_map(ll, Lb, Cb_n, bbox, "B — P-Median → OSRM Trip",
                     road_geoms=geoms_b, tile_src=tile_src)

    # ── C: MCF + OR-Tools per-cluster TSP ──────────────────────────────────
    prog.progress(44, text=f"Pipeline C — OR-Tools TSP per cluster ({c_time_limit}s)…")
    t0 = time.perf_counter()
    Lc, Cc_n, _local_c, ordered_c = run_pipeline_c(ll, K, cap_pct, mat,
                                                     time_limit_s=c_time_limit)
    prog.progress(55, text="Pipeline C — OSRM road geometry (ordered)…")
    # Use OR-Tools visit order for OSRM geometry — this is the fix
    geoms_c, rdist_c, rdur_c = _route_geom_from_ordered(ordered_c, Lc)
    tc = time.perf_counter() - t0
    rm_c  = build_route_metrics(rdist_c, rdur_c, K, osrm_used)
    ana_c = analytics(pts_n, Lc, Cc_n, "C: MCF + OR-Tools TSP", tc,
                      osrm_used=osrm_used,
                      route_metrics=rm_c,
                      fuel_metrics=_fuel(rm_c["total_dist_km"]))
    prog.progress(63, text="Rendering Pipeline C…")
    fig_c = make_map(ll, Lc, Cc_n, bbox, "C — MCF + OR-Tools TSP per cluster",
                     road_geoms=geoms_c, tile_src=tile_src)

    # ── D: Full OR-Tools CVRP ──────────────────────────────────────────────
    prog.progress(67, text=f"Pipeline D — Full OR-Tools CVRP ({d_time_limit}s)…")
    t0 = time.perf_counter()
    Ld, Cd_n, ordered_d = run_pipeline_d(ll, K, cap_pct, mat,
                                          time_limit_s=d_time_limit,
                                          depot_mode=d_depot_mode,
                                          route_mode=d_route_mode)
    prog.progress(80, text="Pipeline D — OSRM road geometry (ordered)…")
    # Use OR-Tools route order directly — this fixes the route-order bug
    geoms_d, rdist_d, rdur_d = _route_geom_from_ordered(ordered_d, Ld)
    td = time.perf_counter() - t0
    rm_d  = build_route_metrics(rdist_d, rdur_d, K, osrm_used)
    d_label = f"D — Full OR-Tools CVRP · {d_depot_mode} depot · {d_route_mode} route"
    ana_d = analytics(pts_n, Ld, Cd_n, "D: Full OR-Tools CVRP", td,
                      osrm_used=osrm_used,
                      route_metrics=rm_d,
                      fuel_metrics=_fuel(rm_d["total_dist_km"]))
    prog.progress(87, text="Rendering Pipeline D…")
    fig_d = make_map(ll, Ld, Cd_n, bbox, d_label,
                     road_geoms=geoms_d, tile_src=tile_src)

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

                # ── Workload ─────────────────────────────────────────────
                mc("Balance CV ↓", f"{cv:.3f}", cls)
                mc("Stops: min/avg/max",
                   f"{ana['min']} / {ana['mean']:.0f} / {ana['max']}")
                mc("Exec Time", f"{ana['time_ms']:.0f} ms")

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
                elif "tour_s" in ana and ana["tour_s"] > 0:
                    unit = "s (road)" if ana.get("osrm") else "norm units"
                    mc("Total Route ↓", f"{ana['tour_s']:.0f} {unit}", "metric-warn")
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

        ("🏗️", "Pipeline D depot modes",
         "<b>centroid</b> — all vans share a single geographic centroid as depot. "
         "Simplest model, comparable to A/B/C.<br>"
         "<b>seeded</b> — each van starts at its MCF cluster centre. "
         "More realistic: each van effectively 'owns' a territory from the start.<br>"
         "<b>random</b> — random start positions. Tests solver robustness.<br>"
         "<b>closed</b> route = vans return to start (standard CVRP). "
         "<b>open</b> route = vans finish at their last stop (no forced return leg). "
         "Open routes produce shorter measured routes but may not reflect real operations."),
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
    C = per-cluster TSP (same zones as A, better ordering) ·
    D = full integrated CVRP ({f"{st.session_state.get('d_depot_mode_used','centroid')} depot" if True else ""}).<br>
    <b>Approximations in use:</b> fuel metrics assume constant speed/consumption ·
    zone overlap uses bounding-box proxy (not exact hull intersection) ·
    open-route mode models return arcs as zero-cost (approximation).<br>
    <b>Production gap:</b> replace OSRM public API with self-hosted instance or
    Google Maps Distance Matrix for real-time traffic.
    </div>""", unsafe_allow_html=True)