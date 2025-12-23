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

def build_vis_nodes_edges(tray_df: pd.DataFrame, connections_df: pd.DataFrame):
    tray_df = ensure_xy_columns(tray_df)

    nodes = []
    for _, r in tray_df.iterrows():
        rn = r.get("RunName", None)
        if pd.isna(rn) or str(rn).strip() == "":
            continue
        rn = str(rn).strip()

        node = {
            "id": rn,
            "label": rn,
            "title": noise_title(rn, r.get("Noise Level", "")),
            "shape": "dot",
        }

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

    edges = []
    seen = set()
    if connections_df is not None and not connections_df.empty:
        for _, r in connections_df.iterrows():
            a = r.get("From", None)
            b = r.get("To", None)
            if pd.isna(a) or pd.isna(b):
                continue
            a = str(a).strip()
            b = str(b).strip()
            if not a or not b:
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"from": a, "to": b})

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

if st.sidebar.button("Clear workbook", key="clear_workbook_btn"):
    st.session_state.tray_df = None
    st.session_state.connections_df = None
    st.session_state.endpoints_df = None
    st.session_state.cables_df = None
    st.session_state.routes_df = None
    st.session_state.upload_hash = None
    st.session_state.uploader_key_v += 1
    st.rerun()

if uploaded is not None:
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
        st.sidebar.success("Workbook loaded.")

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
    st.caption("Drag nodes freely (circles, labels below, hover shows noise levels). Save positions when you want.")

    graph_col, tools_col = st.columns([3, 1], gap="large")

    with graph_col:
        nodes, edges = build_vis_nodes_edges(st.session_state.tray_df, st.session_state.connections_df)

        options = {
            "physics": {"enabled": False},
            "interaction": {
                "dragNodes": True,
                "dragView": True,
                "zoomView": True,
                "hover": True,
            },
            "nodes": {
                "shape": "dot",
                "size": 18,
                "font": {
                    "vadjust": 24,  # label below node
                },
            },
            # Straight edges always
            "edges": {"smooth": False},
        }

        selection = streamlit_vis_network(
            nodes,
            edges,
            height=740,
            width=1200,
            options=options,
            key="vis_network_graph",
        )

    with tools_col:
        st.markdown("### Edit tools")

        st.markdown("**Layout**")
        if st.button("Save current node positions to Tray (X/Y)", key="save_positions_btn", width="stretch"):
            positions = None
            selected_nodes = []
            selected_edges = []

            if selection:
                try:
                    selected_nodes, selected_edges, positions = selection
                except Exception:
                    positions = None

            if positions and isinstance(positions, dict) and len(positions) > 0:
                st.session_state.tray_df = apply_positions_to_tray(st.session_state.tray_df, positions)
                st.success(f"Saved positions for {len(positions)} node(s). You can keep dragging and save again anytime.")
                st.rerun()
            else:
                # Helpful warning explaining the common selection issue
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

        st.divider()

        st.markdown("**Add / Update Node**")
        new_name = st.text_input("RunName", key="add_node_name")
        new_noise = st.text_input("Noise Level (e.g. 1 or 1,2)", key="add_node_noise")
        if st.button("Save Node", key="save_node_btn", width="stretch"):
            if new_name.strip():
                df = ensure_xy_columns(st.session_state.tray_df.copy())
                rn = new_name.strip()
                mask = df["RunName"].astype(str).str.strip() == rn
                if mask.any():
                    df.loc[mask, "Noise Level"] = new_noise
                else:
                    df = pd.concat(
                        [df, pd.DataFrame([{"RunName": rn, "Noise Level": new_noise, "X": pd.NA, "Y": pd.NA}])],
                        ignore_index=True
                    )
                st.session_state.tray_df = df
                st.session_state.routes_df = None
                st.rerun()
            else:
                st.error("RunName cannot be blank.")

        st.divider()

        st.markdown("**Delete Node**")
        del_name = st.text_input("Node to delete (RunName)", key="del_node_name")
        if st.button("Delete Node", key="delete_node_btn", width="stretch"):
            if del_name.strip():
                node = del_name.strip()

                df = st.session_state.tray_df
                st.session_state.tray_df = df[df["RunName"].astype(str).str.strip() != node].reset_index(drop=True)

                con = st.session_state.connections_df
                if "From" in con.columns and "To" in con.columns:
                    st.session_state.connections_df = con[
                        (con["From"].astype(str).str.strip() != node) &
                        (con["To"].astype(str).str.strip() != node)
                    ].reset_index(drop=True)

                ep = st.session_state.endpoints_df
                if "tray/conduit(s)" in ep.columns:
                    def remove_token(s):
                        parts = [p.strip() for p in str(s).split(",") if p.strip()]
                        parts = [p for p in parts if p != node]
                        return ",".join(parts)
                    ep = ep.copy()
                    ep["tray/conduit(s)"] = ep["tray/conduit(s)"].apply(remove_token)
                    st.session_state.endpoints_df = ep

                st.session_state.routes_df = None
                st.rerun()
            else:
                st.error("Enter a node name to delete.")

        st.divider()

        st.markdown("**Connections**")
        a = st.text_input("From", key="edge_from")
        b = st.text_input("To", key="edge_to")

        if st.button("Add Edge", key="add_edge_btn", width="stretch"):
            aa, bb = a.strip(), b.strip()
            if aa and bb and aa != bb:
                con = st.session_state.connections_df.copy()
                if "From" not in con.columns:
                    con["From"] = ""
                if "To" not in con.columns:
                    con["To"] = ""
                exists = set(tuple(sorted((str(r["From"]).strip(), str(r["To"]).strip())))
                             for _, r in con.dropna(subset=["From", "To"]).iterrows())
                key = tuple(sorted((aa, bb)))
                if key not in exists:
                    con = pd.concat([con, pd.DataFrame([{"From": aa, "To": bb}])], ignore_index=True)
                st.session_state.connections_df = con
                st.session_state.routes_df = None
                st.rerun()

        if st.button("Delete Edge", key="delete_edge_btn", width="stretch"):
            aa, bb = a.strip(), b.strip()
            con = st.session_state.connections_df.copy()

            def is_match(r):
                x = str(r.get("From", "")).strip()
                y = str(r.get("To", "")).strip()
                return set([x, y]) == set([aa, bb])

            if ("From" in con.columns) and ("To" in con.columns) and aa and bb:
                mask = con.apply(is_match, axis=1)
                con = con[~mask].reset_index(drop=True)
            st.session_state.connections_df = con
            st.session_state.routes_df = None
            st.rerun()

        st.info("You can keep dragging nodes after saving. Save again anytime.")

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
                net.build_from_dfs(st.session_state.tray_df, st.session_state.connections_df, st.session_state.endpoints_df)
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
