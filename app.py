import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import brentq

st.set_page_config(page_title="Freeze-Drying Design Space", page_icon="❄️", layout="wide")
st.title("Freeze-Drying Tie-Lines and Optimal Operating Trajectory")
st.caption("Conductive design-space model with fixed-shelf-temperature and fixed-product-temperature tie-line families.")

# ---------------------------- MODEL ----------------------------
def kv(ts, pc):
    return (9.211 + (0.066 * pc * 1000.0) / (1.0 + 0.002 * pc * 1000.0)) * 0.001434

def vapor_pressure(tc):
    return np.exp(-6144.96 / (tc + 273.15) + 24.01849)

def cake_resistance(ldry, r0, a1, a2):
    return r0 + (a1 * ldry) / (1.0 + a2 * ldry)

def sublimation_rate(to, pc, rp, ap):
    return ap * (vapor_pressure(to) - pc) / rp

def interface_balance(to, ts, pc, lpre, rp, p):
    remaining = max(p["lmax"] - lpre, 1e-8)
    kval = kv(ts, pc)
    tp = (kval * ts + (p["ki"] / remaining) * to) / (kval + p["ki"] / remaining)
    qin = p["av"] * kval * (ts - tp)
    qsub = p["dhs"] * p["ap"] * (vapor_pressure(to) - pc) / rp
    return qin - qsub

def interface_balance_tp(to, tp, pc, lpre, rp, p):
    remaining = max(p["lmax"] - lpre, 1e-8)
    qcond = p["av"] * p["ki"] * (tp - to) / remaining
    qsub = p["dhs"] * p["ap"] * (vapor_pressure(to) - pc) / rp
    return qcond - qsub

def shelf_balance(ts, tp, pc, qc, p):
    return p["av"] * kv(ts, pc) * (ts - tp) - qc

def build_space(p):
    ts_rows, tp_rows = [], []
    ts_fail = tp_fail = 0
    pressures = np.linspace(p["pmin"], p["pmax"], p["pcount"])
    ts_values = np.arange(p["tsmin"], p["tsmax"] + 0.5 * p["tsstep"], p["tsstep"])
    tp_values = np.arange(p["tpmin"], p["tpmax"] + 0.5 * p["tpstep"], p["tpstep"])
    fractions = np.linspace(0.0, p["maximum_dry_fraction"], p["dry_levels"])
    lpre_values = fractions * p["lmax"]

    for lpre, fraction in zip(lpre_values, fractions):
        rp = cake_resistance(lpre, p["r0"], p["a1"], p["a2"])
        for ts in ts_values:
            for pc in pressures:
                try:
                    to = brentq(interface_balance, p["to_low_ts"], p["to_high_ts"], args=(ts, pc, lpre, rp, p))
                    remaining = max(p["lmax"] - lpre, 1e-8)
                    kval = kv(ts, pc)
                    tp = (kval * ts + (p["ki"] / remaining) * to) / (kval + p["ki"] / remaining)
                    sub = sublimation_rate(to, pc, rp, p["ap"])
                    if np.isfinite(sub) and sub >= 0:
                        ts_rows.append({"family":"Ts", "dry_fraction":fraction, "lpre_cm":lpre, "Ts_C":ts, "Tp_C":tp, "Pc":pc, "To_C":to, "sublimation":sub})
                    else:
                        ts_fail += 1
                except (ValueError, FloatingPointError, ZeroDivisionError):
                    ts_fail += 1

        for tp in tp_values:
            for pc in pressures:
                try:
                    to = brentq(interface_balance_tp, p["to_low_tp"], tp - 0.1, args=(tp, pc, lpre, rp, p))
                    remaining = max(p["lmax"] - lpre, 1e-8)
                    qc = p["av"] * p["ki"] * (tp - to) / remaining
                    ts = brentq(shelf_balance, tp, tp + p["ts_search_span"], args=(tp, pc, qc, p))
                    sub = sublimation_rate(to, pc, rp, p["ap"])
                    if np.isfinite(sub) and sub >= 0:
                        tp_rows.append({"family":"Tp", "dry_fraction":fraction, "lpre_cm":lpre, "Ts_C":ts, "Tp_C":tp, "Pc":pc, "To_C":to, "sublimation":sub})
                    else:
                        tp_fail += 1
                except (ValueError, FloatingPointError, ZeroDivisionError):
                    tp_fail += 1

    ts_df = pd.DataFrame(ts_rows)
    tp_df = pd.DataFrame(tp_rows)
    optimum = []
    for fraction, lpre in zip(fractions, lpre_values):
        candidates = ts_df[np.isclose(ts_df["lpre_cm"], lpre, atol=1e-7)] if not ts_df.empty else pd.DataFrame()
        if not candidates.empty:
            candidates = candidates[candidates["Tp_C"] <= p["collapse_temperature"]]
        if candidates.empty:
            optimum.append({"dry_fraction":fraction, "lpre_cm":lpre, "optimal_Ts_C":np.nan, "optimal_Pc":np.nan, "max_sublimation":np.nan, "product_temperature_C":np.nan, "interface_temperature_C":np.nan})
        else:
            best = candidates.loc[candidates["sublimation"].idxmax()]
            optimum.append({"dry_fraction":fraction, "lpre_cm":lpre, "optimal_Ts_C":best["Ts_C"], "optimal_Pc":best["Pc"], "max_sublimation":best["sublimation"], "product_temperature_C":best["Tp_C"], "interface_temperature_C":best["To_C"]})
    return ts_df, tp_df, pd.DataFrame(optimum), ts_fail, tp_fail

def excel_bytes(ts_df, tp_df, opt_df, parameters):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        opt_df.to_excel(writer, sheet_name="Optimal trajectory", index=False)
        ts_df.to_excel(writer, sheet_name="Ts tie lines", index=False)
        tp_df.to_excel(writer, sheet_name="Tp tie lines", index=False)
        pd.DataFrame({"Parameter":list(parameters.keys()), "Value":list(parameters.values())}).to_excel(writer, sheet_name="Inputs", index=False)
    return output.getvalue()

# ---------------------------- INPUTS ----------------------------
with st.sidebar:
    st.header("Model inputs")
    fill = st.number_input("Fill volume (cm³)", value=48.0, min_value=0.001)
    d_out = st.number_input("Outer diameter (cm)", value=4.7, min_value=0.001)
    wall = st.number_input("Wall thickness (cm)", value=0.17, min_value=0.0)
    dhs = st.number_input("Sublimation enthalpy", value=680.0)
    ki_base = st.number_input("Ice conductivity base", value=0.0059, format="%.6f")
    rho = st.number_input("Ice density (g/cm³)", value=0.918, min_value=0.001)
    collapse_temperature = st.number_input("Collapse temperature (°C)", value=20.0)

    st.subheader("Cake resistance")
    r0 = st.number_input("R0", value=44.59)
    a1 = st.number_input("a1", value=1451.73)
    a2 = st.number_input("a2", value=12.93)

    with st.expander("Design-space grid"):
        pmin = st.number_input("Minimum pressure", value=0.05, min_value=0.0001, format="%.3f")
        pmax = st.number_input("Maximum pressure", value=0.50, min_value=0.0002, format="%.3f")
        pcount = st.number_input("Pressure points", value=21, min_value=3, max_value=101, step=2)
        tsmin = st.number_input("Minimum shelf temperature (°C)", value=-30.0)
        tsmax = st.number_input("Maximum shelf temperature (°C)", value=30.0)
        tsstep = st.number_input("Shelf-temperature step (°C)", value=5.0, min_value=0.1)
        tpmin = st.number_input("Minimum product temperature (°C)", value=-20.0)
        tpmax = st.number_input("Maximum product temperature (°C)", value=-10.0)
        tpstep = st.number_input("Product-temperature step (°C)", value=5.0, min_value=0.1)
        dry_levels = st.number_input("Dry-layer levels", value=10, min_value=2, max_value=30)
        maximum_dry_fraction = st.number_input("Maximum dry fraction", value=0.9, min_value=0.1, max_value=0.99, format="%.2f")

    with st.expander("Root-solver limits"):
        to_low_ts = st.number_input("Ts-family lower To bound (°C)", value=-60.0)
        to_high_ts = st.number_input("Ts-family upper To bound (°C)", value=-1.0)
        to_low_tp = st.number_input("Tp-family lower To bound (°C)", value=-40.0)
        ts_search_span = st.number_input("Shelf-temperature search span (°C)", value=80.0, min_value=1.0)

    run = st.button("Calculate design space", type="primary", use_container_width=True)

d_in = d_out - 2.0 * wall
if d_in <= 0:
    st.error("Outer diameter must be greater than twice the wall thickness.")
    st.stop()
if pmax <= pmin:
    st.error("Maximum pressure must be greater than minimum pressure.")
    st.stop()

av = np.pi * d_out**2 / 4.0
ap = np.pi * d_in**2 / 4.0
lmax = fill / ap
parameters = {"fill":fill, "d_out":d_out, "wall":wall, "d_in":d_in, "av":av, "ap":ap, "lmax":lmax, "dhs":dhs, "ki":ki_base*60.0, "rho":rho, "collapse_temperature":collapse_temperature, "r0":r0, "a1":a1, "a2":a2, "pmin":pmin, "pmax":pmax, "pcount":int(pcount), "tsmin":tsmin, "tsmax":tsmax, "tsstep":tsstep, "tpmin":tpmin, "tpmax":tpmax, "tpstep":tpstep, "dry_levels":int(dry_levels), "maximum_dry_fraction":maximum_dry_fraction, "to_low_ts":to_low_ts, "to_high_ts":to_high_ts, "to_low_tp":to_low_tp, "ts_search_span":ts_search_span}

cols = st.columns(4)
cols[0].metric("Inner diameter", f"{d_in:.3f} cm")
cols[1].metric("Product area", f"{ap:.3f} cm²")
cols[2].metric("Vial area", f"{av:.3f} cm²")
cols[3].metric("Initial cake height", f"{lmax:.3f} cm")

if run:
    with st.spinner("Solving tie-line families and optimal trajectory..."):
        try:
            ts_df, tp_df, opt_df, ts_fail, tp_fail = build_space(parameters)
            st.session_state["ds_results"] = (ts_df, tp_df, opt_df, ts_fail, tp_fail, parameters.copy())
        except Exception as exc:
            st.exception(exc)

if "ds_results" not in st.session_state:
    st.info("Set the model inputs and select **Calculate design space**.")
else:
    ts_df, tp_df, opt_df, ts_fail, tp_fail, used = st.session_state["ds_results"]
    feasible_opt = opt_df.dropna(subset=["max_sublimation"])
    m = st.columns(5)
    m[0].metric("Feasible Ts cases", f"{len(ts_df):,}")
    m[1].metric("Failed Ts cases", f"{ts_fail:,}")
    m[2].metric("Feasible Tp cases", f"{len(tp_df):,}")
    m[3].metric("Failed Tp cases", f"{tp_fail:,}")
    m[4].metric("Maximum optimal rate", f"{feasible_opt['max_sublimation'].max():.5f}" if not feasible_opt.empty else "No feasible point")

    if feasible_opt.empty:
        st.warning("No feasible optimum was found under the selected collapse-temperature constraint.")
    else:
        st.success(f"Feasible optimal conditions were found for {len(feasible_opt)} of {len(opt_df)} dry-layer levels.")

    options = opt_df[["dry_fraction", "lpre_cm"]].copy()
    options["label"] = options.apply(lambda r: f"{r['dry_fraction']*100:.0f}% dry | lpre = {r['lpre_cm']:.3f} cm", axis=1)
    selected_label = st.selectbox("Tie-line dry-layer level", options["label"].tolist())
    selected_index = options.index[options["label"] == selected_label][0]
    selected_lpre = options.loc[selected_index, "lpre_cm"]

    tab_dashboard, tab_tie, tab_opt, tab_control, tab_table, tab_download = st.tabs(["Combined dashboard", "Tie-lines", "Optimal rate", "Optimal Ts and Pc", "Results table", "Downloads"])

    def plot_tielines(ax):
        ts_part = ts_df[np.isclose(ts_df["lpre_cm"], selected_lpre)]
        tp_part = tp_df[np.isclose(tp_df["lpre_cm"], selected_lpre)]
        for ts, group in ts_part.groupby("Ts_C"):
            group = group.sort_values("Pc")
            if len(group) >= 3:
                ax.plot(group["Pc"], group["sublimation"], linewidth=2, label=f"Ts = {ts:g} °C")
        for tp, group in tp_part.groupby("Tp_C"):
            group = group.sort_values("Pc")
            if len(group) >= 3:
                ax.plot(group["Pc"], group["sublimation"], "--", linewidth=2, label=f"Tp = {tp:g} °C")
        ax.set_xlabel("Chamber pressure")
        ax.set_ylabel("Sublimation rate")
        ax.set_title(f"Tie-Lines at lpre = {selected_lpre:.3f} cm")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)

    with tab_dashboard:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plot_tielines(axes[0, 0])
        axes[0, 1].plot(opt_df["lpre_cm"], opt_df["max_sublimation"], "o-")
        axes[0, 1].set_title("Maximum Sublimation Rate")
        axes[0, 1].set_xlabel("Dry-layer thickness (cm)")
        axes[0, 1].set_ylabel("Maximum sublimation rate")
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(opt_df["lpre_cm"], opt_df["optimal_Ts_C"], "s-", color="darkred")
        axes[1, 0].set_title("Optimal Shelf Temperature")
        axes[1, 0].set_xlabel("Dry-layer thickness (cm)")
        axes[1, 0].set_ylabel("Shelf temperature (°C)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(opt_df["lpre_cm"], opt_df["optimal_Pc"], "o-", color="navy")
        axes[1, 1].set_title("Optimal Chamber Pressure")
        axes[1, 1].set_xlabel("Dry-layer thickness (cm)")
        axes[1, 1].set_ylabel("Pressure")
        axes[1, 1].grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_tie:
        fig, ax = plt.subplots(figsize=(12, 7))
        plot_tielines(ax)
        st.pyplot(fig)
        plt.close(fig)

    with tab_opt:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(opt_df["lpre_cm"], opt_df["max_sublimation"], "o-")
        ax.set_xlabel("Dry-layer thickness, lpre (cm)")
        ax.set_ylabel("Maximum sublimation rate")
        ax.set_title("Optimal Sublimation Rate")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with tab_control:
        fig, ax1 = plt.subplots(figsize=(11, 6))
        line1 = ax1.plot(opt_df["lpre_cm"], opt_df["optimal_Ts_C"], "s-", linewidth=2, color="darkred", label="Optimal Ts")
        ax1.set_xlabel("Dry-layer thickness, lpre (cm)")
        ax1.set_ylabel("Shelf temperature, Ts (°C)", color="darkred")
        ax1.tick_params(axis="y", labelcolor="darkred")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        line2 = ax2.plot(opt_df["lpre_cm"], opt_df["optimal_Pc"], "o-", linewidth=2, color="navy", label="Optimal Pc")
        ax2.set_ylabel("Chamber pressure", color="navy")
        ax2.tick_params(axis="y", labelcolor="navy")
        ax1.legend(line1 + line2, [x.get_label() for x in line1 + line2])
        ax1.set_title("Optimal Shelf Temperature and Chamber Pressure")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_table:
        display = opt_df.copy()
        display["dry_fraction_percent"] = 100 * display["dry_fraction"]
        display = display[["dry_fraction_percent", "lpre_cm", "optimal_Ts_C", "optimal_Pc", "max_sublimation", "product_temperature_C", "interface_temperature_C"]]
        st.dataframe(display, use_container_width=True, height=480)

    with tab_download:
        csv_data = opt_df.to_csv(index=False).encode("utf-8")
        xlsx_data = excel_bytes(ts_df, tp_df, opt_df, used)
        c1, c2 = st.columns(2)
        c1.download_button("Download optimal trajectory CSV", csv_data, "optimal_trajectory.csv", "text/csv", use_container_width=True)
        c2.download_button("Download complete Excel workbook", xlsx_data, "freeze_drying_design_space.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

st.divider()
st.caption("Important: verify pressure, resistance, heat-transfer, enthalpy, geometry, and time-unit bases before using the design space for process decisions.")
