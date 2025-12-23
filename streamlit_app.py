import streamlit as st
import pandas as pd
from collections import defaultdict, deque
from io import BytesIO
import hashlib

from streamlit_vis_network import streamlit_vis_network


# ============================================================
# Core Routing (Version A logic, refactored to use DataFrames)
# ============================================================

class CableNetwork:
    def __init__(self):
        self.graph = defaultdict(list)
        self.noise_levels = {}
        self.endpoints = defaultdict(list)
        self.endpoint_exposed = {}

    @staticmethod
    def _infer_noise_from_name(run_name: str):
        name = str(run_name or "")
        if "T6" in name:
            return {1}
        if "T4" in name:
            return {2}
        if "T3" in name:
            return {3}
        return set()

    @staticmethod
    def _parse_noise_levels(val, run_name=None):
        if pd.isna(val) or str(val).strip() == "":
            return CableNetwork._infer_noise_from_name(run_name)

        if isinstance(val, (int, float)) and not pd.isna(val):
            return {int(val)}

        s = str(val).strip()
        parts = [p.strip() for p in s.split(",")]
        out = set()
        for p in parts:
            if p:
                try:
                    out.add(int(float(p)))
                except ValueError:
                    pass
        return out

    def add_tray_or_conduit(self, name, noise_levels):
        levels = set(noise_levels) if isinstance(noise_levels, (set, list, tuple)) else {int(noise_levels)}
        self.noise_levels[name] = levels
        if name not in self.graph:
            self.graph[name] = []

    def connect(self, from_node, to_node):
        if from_node in self.noise_levels and to_node in self.noise_levels:
            self.graph[from_node].append((to_node, self.noise_levels[to_node]))
            self.graph[to_node].append((from_node, self.noise_levels[from_node]))

    def register_endpoint(self, device_name, tray_or_conduits, exposed=None):
        device_name = str(device_name).strip().lstrip("+")
        for item in str(tray_or_conduits).split(","):
            it = item.strip()
            if it:
                self.endpoints[device_name].append(it)
        if exposed is not None:
            self.endpoint_exposed[device_name] = bool(exposed)

    def find_route(self, starts, ends, allowed_noise_level):
        visited = set()
        queue = deque([[start] for start in starts])
        end_set = set(ends)

        while queue:
            path = queue.popleft()
            node = path[-1]
            if node in end_set:
                return path
            if node not in visited:
                visited.add(node)
                for neighbor, neighbor_levels in self.graph[node]:
                    if neighbor not in visited and allowed_noise_level in (neighbor_levels or set()):
                        queue.append(path + [neighbor])
        return None

    def build_from_dfs(self, trays_df: pd.DataFrame, connections_df: pd.DataFrame, endpoints_df: pd.DataFrame):
        for _, row in trays_df.iterrows():
            run_name = row.get("RunName", None)
            if pd.isna(run_name) or str(run_name).strip() == "":
                continue
            run_name = str(run_name).strip()
            levels = self._parse_noise_levels(row.get("Noise Level", None), run_name=run_name)
            if levels:
                self.add_tray_or_conduit(run_name, levels)

        for _, row in connections_df.iterrows():
            a = row.get("From", None)
            b = row.get("To", None)
            if pd.isna(a) or pd.isna(b):
                continue
            self.connect(str(a).strip(), str(b).strip())

        for _, row in endpoints_df.iterrows():
            device = row.get("device/panel", "")
            trays = row.get("tray/conduit(s)", "")
            exposed_flag = False
            try:
                val = row.iloc[2] if len(row) >= 3 else ""
            except Exception:
                val = ""
            if str(val).strip().lower() == "yes":
                exposed_flag = True
            self.register_endpoint(device, trays, exposed=exposed_flag)

    def route_cables_df(self, cables_df: pd.DataFrame) -> pd.DataFrame:
        cables = []
        seen = set()

        for _, row in cables_df.iterrows():
            sort = None
            if "Sort" in row and not pd.isna(row["Sort"]):
                try:
                    sort = int(row["Sort"])
                except Exception:
                    sort = None

            name = str(row.get("Cable number", ""))
            start = str(row.get("equipfrom", "")).strip().lstrip("+")
            end = str(row.get("equipto", "")).strip().lstrip("+")

            nl = row.get("Noise Level", None)
            if pd.isna(nl):
                continue
            try:
                noise_level = int(float(nl))
            except Exception:
                continue

            key = sort if sort is not None else name
            if key in seen:
                continue
            seen.add(key)

            cables.append({
                "sort": sort,
                "name": name,
                "start": start,
                "end": end,
                "noise_level": noise_level
            })

        results = []
        for cable in cables:
            nl = cable["noise_level"]

            start_trays = [t for t in self.endpoints.get(cable["start"], [])
                           if nl in self.noise_levels.get(t, set())]
            end_trays = [t for t in self.endpoints.get(cable["end"], [])
                         if nl in self.noise_levels.get(t, set())]

            if not start_trays and not end_trays:
                path_result = "Error: No endpoints with matching noise level (start & end)"
            elif not start_trays:
                path_result = "Error: No start endpoint with matching noise level"
            elif not end_trays:
                path_result = "Error: No end endpoint with matching noise level"
            else:
                path = self.find_route(start_trays, set(end_trays), nl)
                if path:
                    suffix = ""
                    if nl == 1:
                        suffix = "(T6)"
                    elif nl == 2:
                        suffix = "(T4)"

                    def format_node(node):
                        node = str(node)
                        if "LT" in node:
                            return f"{node}{suffix}"
                        if "CND" in node:
                            return node
                        return node

                    parts = [format_node(p) for p in path]

                    FROM_exposed = self.endpoint_exposed.get(cable["start"], False)
                    TO_exposed = self.endpoint_exposed.get(cable["end"], False)

                    if FROM_exposed:
                        parts.insert(0, "EXPOSED CONDUIT ROUTE")
                    if TO_exposed:
                        parts.append("EXPOSED CONDUIT ROUTE")

                    path_result = ",".join(parts)
                else:
                    path_result = "No valid route"

            results.append({
                "Sort": cable["sort"],
                "Cable number": cable["name"],
                "equipfrom": cable["start"],
                "equipto": cable["end"],
                "Noise Level": nl,
                "Via": path_result
            })

        return pd.DataFrame(results)


# ============================================================
# Excel I/O helpers
# ============================================================

REQUIRED_SHEETS = ["Tray", "Connections", "Endpoints", "Cables(input)"]

def load_excel_to_dfs(file_bytes: bytes) -> dict:
    xls = pd.ExcelFile(BytesIO(file_bytes))
    dfs = {}
    for sh in REQUIRED_SHEETS:
        if sh not in xls.sheet_names:
            raise ValueError(f"Missing required sheet: {sh}")
        dfs[sh] = pd.read_excel(xls, sh)
    return dfs

def write_updated_workbook_bytes(dfs: dict) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sh, df in dfs.items():
            df.to_excel(writer, sheet_name=sh, index=False)
    out.seek(0)
    return out.getvalue()

def write_routed_workbook_bytes(dfs: dict, routes_df: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sh, df in dfs.items():
            df.to_excel(writer, sheet_name=sh, index=False)

        out_df = routes_df.copy()
        if "Cable number" in out_df.columns:
            out_df = out_df.rename(columns={"Cable number": "Cable Number"})
        out_df = out_df[["Sort", "Cable Number", "equipfrom", "equipto", "Noise Level", "Via"]]
        out_df = out_df.where(out_df.notna(), "")
        for c in out_df.columns:
            out_df[c] = out_df[c].astype(str)

        out_df.to_excel(writer, sheet_name="CableRoutes(output)", index=False)

        from openpyxl.styles import numbers
        wb = writer.book
        ws = wb["CableRoutes(output)"]
        header_map = {cell.value: idx for idx, cell in enumerate(ws[1], start=1)}
        for col_name in ["Cable Number", "equipfrom", "equipto", "Via"]:
            if col_name in header_map:
                cidx = header_map[col_name]
                for r in range(2, ws.max_row + 1):
                    cell = ws.cell(row=r, column=cidx)
                    cell.value = "" if cell.value is None else str(cell.value)
                    cell.number_format = numbers.FORMAT_TEXT

    out.seek(0)
    return out.getvalue()

def validate_dfs(dfs: dict) -> list[str]:
    errors = []
    expected_cols = {
        "Tray": {"RunName"},
        "Connections": {"From", "To"},
        "Endpoints": {"device/panel", "tray/conduit(s)"},
        "Cables(input)": {"Cable number", "equipfrom", "equipto", "Noise Level"},
    }
    for sh, cols in expected_cols.items():
        missing = cols - set(dfs[sh].columns)
        if missing:
            errors.append(f"{sh}: missing columns {sorted(missing)}")
    return errors


# ============================================================
# Graph helpers
# ============================================================

def ensure_xy_columns(tray_df: pd.DataFrame) -> pd.DataFrame:
    tray_df = tray_df.copy()
    if "X" not in tray_df.columns:
        tray_df["X"] = pd.NA
    if "Y" not in tray_df.columns:
        tray_df["Y"] = pd.NA
    return tray_df

def noise_title(run_name: str, noise_val) -> str:
    lv = CableNetwork._parse_noise_levels(noise_val, run_name=run_name)
    if not lv:
        return "Noise Levels: N/A"
    return "Noise Levels: " + ",".join(str(x) for x in sorted(lv))

def infer_type_from_name(name: str) -> str:
    s = str(name).upper()
    if "CND" in s:
        return "Conduit"
    if "LT" in s:
        return "Tray"
    return "Node"

def build_adjacency(connections_df: pd.DataFrame) -> dict[str, set[str]]:
    adj = defaultdict(set)
    if connections_df is None or connections_df.empty:
        return adj
    if "From" not in connections_df.columns or "To" not in connections_df.columns:
        return adj
    for _, r in connections_df.iterrows():
        a = r.get("From", None)
        b = r.get("To", None)
        if pd.isna(a) or pd.isna(b):
            continue
        a = str(a).strip()
        b = str(b).strip()
        if not a or not b:
            continue
        adj[a].add(b)
        adj[b].add(a)
    return adj

def neighborhood_nodes(adj: dict[str, set[str]], start: str, depth: int) -> set[str]:
    start = str(start).strip()
    if not start:
        return set()
    seen = {start}
    frontier = {start}
    for _ in range(max(0, int(depth))):
        nxt = set()
        for u in frontier:
            nxt |= set(adj.get(u, set()))
        nxt -= seen
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    return seen

def build_vis_nodes_edges(tray_df: pd.DataFrame, connections_df: pd.DataFrame, focus_node: str | None = None, focus_depth: int = 2):
    tray_df = ensure_xy_columns(tray_df)

    focus_set = None
    if focus_node:
        adj = build_adjacency(connections_df)
        focus_set = neighborhood_nodes(adj, focus_node, focus_depth)

    nodes = []
    for _, r in tray_df.iterrows():
        rn = r.get("RunName", None)
        if pd.isna(rn) or str(rn).strip() == "":
            continue
        rn = str(rn).strip()

        if focus_set is not None and rn not in focus_set:
            continue

        is_focus = (focus_node is not None and rn == str(focus_node).strip())

        node = {
            "id": rn,
            "label": rn,
            "title": noise_title(rn, r.get("Noise Level", "")),
            "shape": "dot",
            "size": 18,
        }

        if is_focus:
            node["color"] = {"background": "#FFD700", "border": "#000000"}

        x = r.get("X", pd.NA)
        y = r.get("Y", pd.NA)
        try:
            x = None if pd.isna(x) else float(x)
        except Exception:
            x = None
        try:
            y = None if pd.isna(y) else float(y)
        except Exception:
            y = None

        if x is not None and y is not None:
            node["x"] = x
            node["y"] = y

        nodes.append(node)

    node_ids = {n["id"] for n in nodes}

    edges = []
    seen = set()
    if connections_df is not None and not connections_df.empty and "From" in connections_df.columns and "To" in connections_df.columns:
        for _, r in connections_df.iterrows():
            a = r.get("From", None)
            b = r.get("To", None)
            if pd.isna(a) or pd.isna(b):
                continue
            a = str(a).strip()
            b = str(b).strip()
            if not a or not b:
                continue
            if a not in node_ids or b not in node_ids:
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)

            # IMPORTANT:
            # Some builds of streamlit_vis_network return selected edges as [from,to] instead of "id".
            # We still set id (for stability), but selection parsing must handle list-form too.
            edges.append({
                "id": f"{a}|||{b}",
                "from": a,
                "to": b,
                "title": f"{a} ↔ {b}",
            })

    return nodes, edges

def apply_positions_to_tray(tray_df: pd.DataFrame, positions: dict) -> pd.DataFrame:
    tray_df = ensure_xy_columns(tray_df)
    if not positions:
        return tray_df

    tray_df = tray_df.copy()
    rn_series = tray_df["RunName"].astype(str).str.strip()

    for nid, xy in positions.items():
        x = y = None
        if isinstance(xy, dict):
            x = xy.get("x")
            y = xy.get("y")
        elif isinstance(xy, (list, tuple)) and len(xy) >= 2:
            x, y = xy[0], xy[1]

        try:
            x = float(x)
            y = float(y)
        except Exception:
            continue

        mask = rn_series == str(nid).strip()
        if mask.any():
            tray_df.loc[mask, "X"] = x
            tray_df.loc[mask, "Y"] = y

    return tray_df


# ============================================================
# DataFrame mutation helpers
# ============================================================

def df_delete_node(tray_df, connections_df, endpoints_df, node: str):
    node = str(node).strip()
    tray_df2 = tray_df.copy()
    tray_df2 = tray_df2[tray_df2["RunName"].astype(str).str.strip() != node].reset_index(drop=True)

    con2 = connections_df.copy()
    if "From" in con2.columns and "To" in con2.columns:
        con2 = con2[
            (con2["From"].astype(str).str.strip() != node) &
            (con2["To"].astype(str).str.strip() != node)
        ].reset_index(drop=True)

    ep2 = endpoints_df.copy()
    if "tray/conduit(s)" in ep2.columns:
        def remove_token(s):
            parts = [p.strip() for p in str(s).split(",") if p.strip()]
            parts = [p for p in parts if p != node]
            return ",".join(parts)
        ep2["tray/conduit(s)"] = ep2["tray/conduit(s)"].apply(remove_token)

    return tray_df2, con2, ep2

def df_delete_edge(connections_df, a: str, b: str):
    aa = str(a).strip()
    bb = str(b).strip()
    con = connections_df.copy()

    def is_match(r):
        x = str(r.get("From", "")).strip()
        y = str(r.get("To", "")).strip()
        return set([x, y]) == set([aa, bb])

    if ("From" in con.columns) and ("To" in con.columns):
        mask = con.apply(is_match, axis=1)
        con = con[~mask].reset_index(drop=True)
    return con

def df_rename_node(tray_df, connections_df, endpoints_df, old: str, new: str):
    old = str(old).strip()
    new = str(new).strip()
    if not old or not new or old == new:
        return tray_df, connections_df, endpoints_df

    t = tray_df.copy()
    mask = t["RunName"].astype(str).str.strip() == old
    if mask.any():
        t.loc[mask, "RunName"] = new

    c = connections_df.copy()
    if "From" in c.columns:
        c.loc[c["From"].astype(str).str.strip() == old, "From"] = new
    if "To" in c.columns:
        c.loc[c["To"].astype(str).str.strip() == old, "To"] = new

    e = endpoints_df.copy()
    if "tray/conduit(s)" in e.columns:
        def rename_token(s):
            parts = [p.strip() for p in str(s).split(",") if p.strip()]
            parts = [new if p == old else p for p in parts]
            return ",".join(parts)
        e["tray/conduit(s)"] = e["tray/conduit(s)"].apply(rename_token)

    return t, c, e

def df_duplicate_node(tray_df, source_name: str, new_name: str):
    source_name = str(source_name).strip()
    new_name = str(new_name).strip()
    if not source_name or not new_name:
        return tray_df
    df = ensure_xy_columns(tray_df.copy())

    if (df["RunName"].astype(str).str.strip() == new_name).any():
        return df

    src_mask = df["RunName"].astype(str).str.strip() == source_name
    if not src_mask.any():
        return df

    src_row = df[src_mask].iloc[0].to_dict()
    src_row["RunName"] = new_name

    try:
        if not pd.isna(src_row.get("X")):
            src_row["X"] = float(src_row["X"]) + 50.0
        if not pd.isna(src_row.get("Y")):
            src_row["Y"] = float(src_row["Y"]) + 50.0
    except Exception:
        pass

    df = pd.concat([df, pd.DataFrame([src_row])], ignore_index=True)
    return df


# ============================================================
# Selection parsing helper (FIX FOR EDGE DELETE)
# ============================================================

def parse_selected_edge(sel_edge) -> tuple[str | None, str | None, str]:
    """
    streamlit_vis_network edge selection is inconsistent across versions:
      - Sometimes: "A|||B" (string id)
      - Sometimes: ["A","B"] (list/tuple endpoints)  <-- what you're seeing
      - Sometimes: {"from": "...", "to": "..."} (dict)
    This normalizes to (a, b, display_text).
    """
    if sel_edge is None:
        return None, None, "None"

    # dict with from/to
    if isinstance(sel_edge, dict):
        a = str(sel_edge.get("from", "")).strip()
        b = str(sel_edge.get("to", "")).strip()
        if a and b:
            return a, b, f"{a} ↔ {b}"
        # maybe has id
        sid = sel_edge.get("id")
        if isinstance(sid, str) and "|||" in sid:
            x, y = sid.split("|||", 1)
            return x.strip(), y.strip(), f"{x.strip()} ↔ {y.strip()}"
        return None, None, str(sel_edge)

    # list/tuple like ["A","B"]
    if isinstance(sel_edge, (list, tuple)):
        if len(sel_edge) >= 2:
            a = str(sel_edge[0]).strip()
            b = str(sel_edge[1]).strip()
            if a and b:
                return a, b, f"{a} ↔ {b}"
        return None, None, str(sel_edge)

    # string
    if isinstance(sel_edge, str):
        s = sel_edge.strip()
        if "|||" in s:
            x, y = s.split("|||", 1)
            return x.strip(), y.strip(), f"{x.strip()} ↔ {y.strip()}"
        return None, None, s

    return None, None, str(sel_edge)


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Cable Routing Webapp (Streamlit)", layout="wide")
st.title("Cable Routing Webapp (Streamlit)")

st.sidebar.header("Workbook")

st.session_state.setdefault("uploader_key_v", 0)
uploader_key = f"uploader_{st.session_state.uploader_key_v}"
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], key=uploader_key)

st.session_state.setdefault("tray_df", None)
st.session_state.setdefault("connections_df", None)
st.session_state.setdefault("endpoints_df", None)
st.session_state.setdefault("cables_df", None)
st.session_state.setdefault("routes_df", None)
st.session_state.setdefault("upload_hash", None)

st.session_state.setdefault("sel_nodes", [])
st.session_state.setdefault("sel_edges", [])

st.session_state.setdefault("focus_node", None)
st.session_state.setdefault("focus_depth", 2)

GRAPH_HEIGHT = 740

st.markdown(
    """
    <style>
      div[data-testid="stVerticalBlockBorderWrapper"]{
        border-radius: 10px !important;
      }
      /* Stretch component iframes inside our wrapper (helps fullscreen width) */
      .graphwrap, .graphwrap > div { width: 100% !important; }
      .graphwrap iframe { width: 100% !important; min-width: 100% !important; display: block !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("Clear workbook", key="clear_workbook_btn"):
    st.session_state.tray_df = None
    st.session_state.connections_df = None
    st.session_state.endpoints_df = None
    st.session_state.cables_df = None
    st.session_state.routes_df = None
    st.session_state.upload_hash = None
    st.session_state.sel_nodes = []
    st.session_state.sel_edges = []
    st.session_state.focus_node = None
    st.session_state.uploader_key_v += 1
    st.rerun()

if uploaded is not None:
    try:
        file_bytes = uploaded.getvalue()
        this_hash = hashlib.md5(file_bytes).hexdigest()
        if st.session_state.upload_hash != this_hash:
            loaded = load_excel_to_dfs(file_bytes)
            st.session_state.tray_df = ensure_xy_columns(loaded["Tray"])
            st.session_state.connections_df = loaded["Connections"]
            st.session_state.endpoints_df = loaded["Endpoints"]
            st.session_state.cables_df = loaded["Cables(input)"]
            st.session_state.routes_df = None
            st.session_state.upload_hash = this_hash
            st.session_state.sel_nodes = []
            st.session_state.sel_edges = []
            st.session_state.focus_node = None
            st.sidebar.success("Workbook loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to load workbook: {e}")
        st.stop()

if st.session_state.tray_df is None:
    st.info("Upload an Excel workbook to begin.")
    st.stop()

st.session_state.tray_df = ensure_xy_columns(st.session_state.tray_df)

dfs = {
    "Tray": st.session_state.tray_df,
    "Connections": st.session_state.connections_df,
    "Endpoints": st.session_state.endpoints_df,
    "Cables(input)": st.session_state.cables_df,
}

val_errors = validate_dfs(dfs)
if val_errors:
    st.warning("Workbook validation warnings:")
    for e in val_errors:
        st.write(f"- {e}")

tab1, tab2, tab3, tab4, tabG, tab5 = st.tabs(
    ["Tray", "Connections", "Endpoints", "Cables(input)", "Graph Editor", "Routing Output"]
)

with tab1:
    st.subheader("Tray")
    st.caption("X and Y store node positions (saved manually from Graph Editor).")
    st.session_state.tray_df = st.data_editor(
        st.session_state.tray_df,
        num_rows="dynamic",
        width="stretch",
        key="tray_editor",
    )

with tab2:
    st.subheader("Connections")
    st.session_state.connections_df = st.data_editor(
        st.session_state.connections_df,
        num_rows="dynamic",
        width="stretch",
        key="connections_editor",
    )

with tab3:
    st.subheader("Endpoints")
    st.caption("Column C (3rd column) is interpreted as exposed yes/no (case-insensitive).")
    st.session_state.endpoints_df = st.data_editor(
        st.session_state.endpoints_df,
        num_rows="dynamic",
        width="stretch",
        key="endpoints_editor",
    )

with tab4:
    st.subheader("Cables(input)")
    st.session_state.cables_df = st.data_editor(
        st.session_state.cables_df,
        num_rows="dynamic",
        width="stretch",
        key="cables_editor",
    )

with tabG:
    st.subheader("Graph Editor")
    st.caption("Select a node/edge to edit it on the right. Use Search to focus/highlight a node.")

    graph_col, tools_col = st.columns([3, 1], gap="large")

    node_names = (
        st.session_state.tray_df["RunName"]
        .dropna()
        .astype(str)
        .map(str.strip)
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    node_names = sorted(node_names, key=lambda s: s.lower())

    with graph_col:
        graph_box = st.container(border=True)
        with graph_box:
            st.markdown('<div class="graphwrap">', unsafe_allow_html=True)

            nodes, edges = build_vis_nodes_edges(
                st.session_state.tray_df,
                st.session_state.connections_df,
                focus_node=st.session_state.focus_node,
                focus_depth=int(st.session_state.focus_depth),
            )

            options = {
                "physics": {"enabled": False},
                "interaction": {
                    "dragNodes": True,
                    "dragView": True,
                    "zoomView": True,
                    "hover": True,
                    "multiselect": False,
                    "selectConnectedEdges": True,
                },
                "nodes": {
                    "shape": "dot",
                    "size": 18,
                    "font": {"vadjust": 24},
                },
                "edges": {"smooth": False},
            }

            # NOTE: no fixed width so it expands to container
            selection = streamlit_vis_network(
                nodes,
                edges,
                height=GRAPH_HEIGHT,
                options=options,
                key="vis_network_graph",
            )

            st.markdown("</div>", unsafe_allow_html=True)

        if selection:
            try:
                sel_nodes, sel_edges, _pos = selection
            except Exception:
                sel_nodes, sel_edges = [], []
            st.session_state.sel_nodes = sel_nodes or []
            st.session_state.sel_edges = sel_edges or []
        else:
            st.session_state.sel_nodes = []
            st.session_state.sel_edges = []

    with tools_col:
        tools_box = st.container(height=GRAPH_HEIGHT, border=True)
        with tools_box:
            # ====================================================
            # Selection (ALWAYS TOP)
            # ====================================================
            st.markdown("### Selection")

            sel_nodes = st.session_state.sel_nodes or []
            sel_edges = st.session_state.sel_edges or []

            if sel_nodes:
                node_id = str(sel_nodes[0]).strip()
                tdf = st.session_state.tray_df
                row = tdf[tdf["RunName"].astype(str).str.strip() == node_id]
                noise_val = ""
                x_val = y_val = None
                if not row.empty:
                    noise_val = row.iloc[0].get("Noise Level", "")
                    x_val = row.iloc[0].get("X", pd.NA)
                    y_val = row.iloc[0].get("Y", pd.NA)

                st.markdown("**Selected Node**")
                st.write(f"**Name:** {node_id}")
                st.write(f"**Type:** {infer_type_from_name(node_id)}")
                st.write(f"**Noise Level:** {noise_val if str(noise_val).strip() else 'N/A'}")
                st.write(f"**Size:** 18")
                st.write(f"**X:** {'' if pd.isna(x_val) else x_val}")
                st.write(f"**Y:** {'' if pd.isna(y_val) else y_val}")

                st.divider()

                if st.button("Delete node", key="sel_delete_node", width="stretch"):
                    t, c, e = df_delete_node(
                        st.session_state.tray_df,
                        st.session_state.connections_df,
                        st.session_state.endpoints_df,
                        node_id
                    )
                    st.session_state.tray_df = ensure_xy_columns(t)
                    st.session_state.connections_df = c
                    st.session_state.endpoints_df = e
                    st.session_state.routes_df = None
                    if st.session_state.focus_node == node_id:
                        st.session_state.focus_node = None
                    st.session_state.sel_nodes = []
                    st.session_state.sel_edges = []
                    st.success("Node deleted.")
                    st.rerun()

                st.markdown("**Rename**")
                new_name = st.text_input("New name", value=node_id, key="sel_rename_node_new")
                if st.button("Rename node", key="sel_rename_node_btn", width="stretch"):
                    if new_name.strip() and new_name.strip() != node_id:
                        if (st.session_state.tray_df["RunName"].astype(str).str.strip() == new_name.strip()).any():
                            st.error("That name already exists.")
                        else:
                            t, c, e = df_rename_node(
                                st.session_state.tray_df,
                                st.session_state.connections_df,
                                st.session_state.endpoints_df,
                                node_id,
                                new_name.strip()
                            )
                            st.session_state.tray_df = ensure_xy_columns(t)
                            st.session_state.connections_df = c
                            st.session_state.endpoints_df = e
                            st.session_state.routes_df = None
                            if st.session_state.focus_node == node_id:
                                st.session_state.focus_node = new_name.strip()
                            st.session_state.sel_nodes = [new_name.strip()]
                            st.session_state.sel_edges = []
                            st.success("Node renamed.")
                            st.rerun()
                    else:
                        st.warning("Enter a different non-empty name.")

                st.markdown("**Duplicate**")
                dup_name = st.text_input("Duplicate as", value=f"{node_id}_COPY", key="sel_dup_node_new")
                if st.button("Duplicate node", key="sel_dup_node_btn", width="stretch"):
                    if dup_name.strip():
                        if (st.session_state.tray_df["RunName"].astype(str).str.strip() == dup_name.strip()).any():
                            st.error("That duplicate name already exists.")
                        else:
                            st.session_state.tray_df = df_duplicate_node(st.session_state.tray_df, node_id, dup_name.strip())
                            st.session_state.routes_df = None
                            st.success("Node duplicated.")
                            st.rerun()
                    else:
                        st.warning("Enter a name for the duplicate.")

            elif sel_edges:
                raw_edge = sel_edges[0]
                a, b, disp = parse_selected_edge(raw_edge)

                st.markdown("**Selected Connection**")
                st.write(f"**Connection:** {disp}")

                if not (a and b):
                    st.caption("Could not parse endpoints for this connection (unexpected format).")
                    st.code(str(raw_edge))
                else:
                    st.write(f"**From:** {a}")
                    st.write(f"**To:** {b}")

                    st.divider()

                    if st.button("Delete connection", key="sel_delete_edge", width="stretch"):
                        before = len(st.session_state.connections_df) if st.session_state.connections_df is not None else 0
                        st.session_state.connections_df = df_delete_edge(st.session_state.connections_df, a, b)
                        after = len(st.session_state.connections_df) if st.session_state.connections_df is not None else 0
                        st.session_state.routes_df = None

                        st.session_state.sel_nodes = []
                        st.session_state.sel_edges = []

                        if before == after:
                            st.info("No matching connection was found to delete (already removed?).")
                        else:
                            st.success("Connection deleted.")
                        st.rerun()

                    st.caption("This deletes the row from the Connections sheet (undirected match).")

            else:
                st.info("Select a node or connection in the graph to edit it here.")

            st.divider()

            # ====================================================
            # Search / Focus
            # ====================================================
            st.markdown("### Search / Focus")

            st.session_state.focus_depth = st.slider(
                "Neighborhood depth",
                min_value=1,
                max_value=6,
                value=int(st.session_state.focus_depth),
                step=1,
                help="When you focus a node, the graph shows this node + neighbors up to N hops.",
            )

            focus_pick = st.selectbox(
                "Find a node (type to search)",
                options=[""] + node_names,
                index=0 if not st.session_state.focus_node else (
                    node_names.index(st.session_state.focus_node) + 1
                    if st.session_state.focus_node in node_names else 0
                ),
                key="focus_pick_select",
                help="Choosing a node will focus/highlight it by showing only its local neighborhood.",
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Focus", width="stretch", key="focus_btn"):
                    if focus_pick and focus_pick.strip():
                        st.session_state.focus_node = focus_pick.strip()
                        st.session_state.sel_nodes = [st.session_state.focus_node]
                        st.session_state.sel_edges = []
                        st.rerun()
                    else:
                        st.warning("Pick a node to focus.")
            with c2:
                if st.button("Clear", width="stretch", key="clear_focus_btn"):
                    st.session_state.focus_node = None
                    st.rerun()

            if st.session_state.focus_node:
                st.caption(f"Focused on: **{st.session_state.focus_node}** (showing local neighborhood)")

            st.divider()

            # ====================================================
            # Connection (manual) Add/Delete
            # ====================================================
            st.markdown("### Connection")

            conn_from = st.selectbox(
                "From (type to search)",
                options=[""] + node_names,
                index=0,
                key="conn_from",
            )
            conn_to = st.selectbox(
                "To (type to search)",
                options=[""] + node_names,
                index=0,
                key="conn_to",
            )

            b_add, b_del = st.columns(2)

            with b_add:
                if st.button("Add", width="stretch", key="add_edge_btn"):
                    a = (conn_from or "").strip()
                    b = (conn_to or "").strip()
                    if not a or not b:
                        st.warning("Choose both From and To.")
                    elif a == b:
                        st.warning("From and To must be different.")
                    else:
                        con = st.session_state.connections_df.copy()

                        def is_dup(r):
                            x = str(r.get("From", "")).strip()
                            y = str(r.get("To", "")).strip()
                            return set([x, y]) == set([a, b])

                        dup = False
                        if ("From" in con.columns) and ("To" in con.columns) and not con.empty:
                            try:
                                dup = bool(con.apply(is_dup, axis=1).any())
                            except Exception:
                                dup = False

                        if dup:
                            st.info("That connection already exists.")
                        else:
                            new_row = {col: "" for col in con.columns}
                            if "From" in con.columns:
                                new_row["From"] = a
                            if "To" in con.columns:
                                new_row["To"] = b
                            con = pd.concat([con, pd.DataFrame([new_row])], ignore_index=True)

                            st.session_state.connections_df = con
                            st.session_state.routes_df = None
                            st.success(f"Added connection: {a} ↔ {b}")
                            st.rerun()

            with b_del:
                if st.button("Delete", width="stretch", key="delete_edge_btn"):
                    a = (conn_from or "").strip()
                    b = (conn_to or "").strip()
                    if not a or not b:
                        st.warning("Choose both From and To.")
                    elif a == b:
                        st.warning("From and To must be different.")
                    else:
                        before = len(st.session_state.connections_df) if st.session_state.connections_df is not None else 0
                        st.session_state.connections_df = df_delete_edge(st.session_state.connections_df, a, b)
                        after = len(st.session_state.connections_df) if st.session_state.connections_df is not None else 0
                        st.session_state.routes_df = None

                        if before == after:
                            st.info("No matching connection was found to delete.")
                        else:
                            st.session_state.sel_edges = []
                            st.session_state.sel_nodes = []
                            st.success(f"Deleted connection: {a} ↔ {b}")
                        st.rerun()

            st.divider()

            # ====================================================
            # Delete node (autocomplete)
            # ====================================================
            st.markdown("### Delete node (autocomplete)")

            del_pick = st.selectbox(
                "Node to delete (type to search)",
                options=[""] + node_names,
                index=0,
                key="delete_node_pick",
            )
            if st.button("Delete selected node", width="stretch", key="delete_node_autofill_btn"):
                if not del_pick.strip():
                    st.warning("Choose a node to delete.")
                else:
                    node_id = del_pick.strip()
                    t, c, e = df_delete_node(
                        st.session_state.tray_df,
                        st.session_state.connections_df,
                        st.session_state.endpoints_df,
                        node_id
                    )
                    st.session_state.tray_df = ensure_xy_columns(t)
                    st.session_state.connections_df = c
                    st.session_state.endpoints_df = e
                    st.session_state.routes_df = None

                    if st.session_state.focus_node == node_id:
                        st.session_state.focus_node = None
                    if st.session_state.sel_nodes and st.session_state.sel_nodes[0] == node_id:
                        st.session_state.sel_nodes = []
                        st.session_state.sel_edges = []
                    st.success(f"Deleted node: {node_id}")
                    st.rerun()

            st.divider()

            # ====================================================
            # Layout
            # ====================================================
            st.markdown("### Layout")

            if st.button("Save current node positions to Tray (X/Y)", key="save_positions_btn", width="stretch"):
                positions = None
                selected_nodes = st.session_state.sel_nodes or []
                selected_edges = st.session_state.sel_edges or []

                if selection:
                    try:
                        _sn, _se, positions = selection
                    except Exception:
                        positions = None

                if positions and isinstance(positions, dict) and len(positions) > 0:
                    st.session_state.tray_df = apply_positions_to_tray(st.session_state.tray_df, positions)
                    st.success(f"Saved positions for {len(positions)} node(s). You can keep dragging and save again anytime.")
                    st.rerun()
                else:
                    if selected_nodes or selected_edges:
                        st.warning(
                            "Positions could not be captured because something is selected.\n\n"
                            "✅ Click an empty area of the graph to deselect (select the graph), then click **Save** again."
                        )
                    else:
                        st.warning(
                            "No positions were returned yet.\n\n"
                            "✅ Drag a node, then click an empty area of the graph once, then click **Save** again."
                        )

            if st.button("Clear saved positions (blank X/Y)", key="clear_positions_btn", width="stretch"):
                st.session_state.tray_df["X"] = pd.NA
                st.session_state.tray_df["Y"] = pd.NA
                st.session_state.routes_df = None
                st.success("Cleared Tray.X and Tray.Y.")
                st.rerun()

with tab5:
    st.subheader("Route cables")
    colA, colB, colC = st.columns([1, 1, 2])

    with colA:
        if st.button("Validate", key="validate_btn"):
            dfs_for_validate = {
                "Tray": st.session_state.tray_df,
                "Connections": st.session_state.connections_df,
                "Endpoints": st.session_state.endpoints_df,
                "Cables(input)": st.session_state.cables_df,
            }
            errs = validate_dfs(dfs_for_validate)
            if errs:
                st.error("Validation issues found:")
                for e in errs:
                    st.write(f"- {e}")
            else:
                st.success("Basic validation passed.")

    with colB:
        if st.button("Route Now", key="route_btn"):
            try:
                net = CableNetwork()
                net.build_from_dfs(
                    st.session_state.tray_df,
                    st.session_state.connections_df,
                    st.session_state.endpoints_df
                )
                routes_df = net.route_cables_df(st.session_state.cables_df)
                st.session_state.routes_df = routes_df
                st.success("Routing complete.")
            except Exception as e:
                st.error(f"Routing failed: {e}")

    routes_df = st.session_state.routes_df
    if routes_df is not None:
        st.dataframe(routes_df, width="stretch")

        dfs_for_export = {
            "Tray": st.session_state.tray_df,
            "Connections": st.session_state.connections_df,
            "Endpoints": st.session_state.endpoints_df,
            "Cables(input)": st.session_state.cables_df,
        }

        updated_bytes = write_updated_workbook_bytes(dfs_for_export)
        routed_bytes = write_routed_workbook_bytes(dfs_for_export, routes_df)

        st.download_button(
            "Download UPDATED workbook (edited sheets only)",
            data=updated_bytes,
            file_name="network_configuration_updated.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Download ROUTED workbook (includes CableRoutes(output))",
            data=routed_bytes,
            file_name="network_configuration_routed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Click **Route Now** to generate CableRoutes(output).")
