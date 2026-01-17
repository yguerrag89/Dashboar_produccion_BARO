# pages/02_Produccion_Diaria.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from core.transform import build_aggs
from core.charts import chart_diario_line


st.set_page_config(page_title="Producción Diaria", page_icon="🏭", layout="wide")
st.title("🏭 Producción Diaria")


# ==============================================================================
# Carga desde session_state
# ==============================================================================
df_ok = st.session_state.get("df_ok")
aggs = st.session_state.get("aggs")

if df_ok is None:
    st.error("No encontré df_ok en session_state. Abre primero la Home (app.py).")
    st.stop()

if aggs is None:
    aggs = build_aggs(df_ok)
    st.session_state["aggs"] = aggs

base = aggs["base"].copy()

# Fechas disponibles
min_d = pd.to_datetime(base["dia"]).min()
max_d = pd.to_datetime(base["dia"]).max()

if pd.isna(min_d) or pd.isna(max_d):
    st.warning("No hay fechas válidas para construir vistas diarias.")
    st.stop()


# ==============================================================================
# Sidebar filtros
# ==============================================================================
st.sidebar.header("Filtros")

date_range = st.sidebar.date_input(
    "Rango de fechas",
    value=(min_d.date(), max_d.date()),
    min_value=min_d.date(),
    max_value=max_d.date(),
)

start = pd.Timestamp(date_range[0])
end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)  # exclusive

doc_prefixes = sorted(base["doc_prefix"].dropna().astype(str).unique().tolist())
sel_doc = st.sidebar.multiselect(
    "doc_prefix",
    options=doc_prefixes,
    default=doc_prefixes,
)

include_otros = st.sidebar.toggle(
    "Incluir OTROS",
    value=False,
    help="Por defecto se analiza PRIMERA/SEGUNDA. Activa para incluir OTROS en totales.",
)

# sensibilidad atípicos
z_thr = st.sidebar.slider(
    "Sensibilidad días atípicos (Z-score)",
    min_value=1.0,
    max_value=4.0,
    value=2.0,
    step=0.1,
)

window = st.sidebar.slider(
    "Ventana rolling (días) para baseline",
    min_value=7,
    max_value=45,
    value=14,
    step=1,
)

top_prefix_n = st.sidebar.slider(
    "Top doc_prefix",
    min_value=3,
    max_value=15,
    value=8,
    step=1,
)

# aplicar filtros
mask = (pd.to_datetime(base["dia"]) >= start) & (pd.to_datetime(base["dia"]) < end)
if sel_doc:
    mask &= base["doc_prefix"].astype(str).isin(sel_doc)

base_f = base[mask].copy()

if not include_otros:
    base_f = base_f[base_f["calidad"].isin(["PRIMERA", "SEGUNDA"])].copy()

base_f["piezas"] = pd.to_numeric(base_f["piezas"], errors="coerce").fillna(0)
base_f = base_f[base_f["piezas"] > 0].copy()

if base_f.empty:
    st.warning("Con estos filtros no quedan datos.")
    st.stop()


# ==============================================================================
# KPIs rápidos
# ==============================================================================
total_piezas = float(base_f["piezas"].sum())
primera = float(base_f.loc[base_f["calidad"] == "PRIMERA", "piezas"].sum()) if "calidad" in base_f.columns else 0.0
segunda = float(base_f.loc[base_f["calidad"] == "SEGUNDA", "piezas"].sum()) if "calidad" in base_f.columns else 0.0
total_ps = primera + segunda
pct_seg = (segunda / total_ps * 100) if total_ps > 0 else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Piezas (período)", f"{int(round(total_piezas)):,}")
c2.metric("PRIMERA", f"{int(round(primera)):,}")
c3.metric("SEGUNDA", f"{int(round(segunda)):,}")
c4.metric("% SEGUNDA", f"{pct_seg:.2f}%")
c5.metric("Días activos", f"{base_f['dia'].nunique():,}")


# ==============================================================================
# Agregados diarios sobre base filtrada
# ==============================================================================
aggs_f = build_aggs(base_f)
diario = aggs_f["diario"].copy()
diario["dia"] = pd.to_datetime(diario["dia"], errors="coerce")
diario = diario.sort_values("dia").reset_index(drop=True)

# Año seleccionado (para reutilizar chart_diario_line)
years = sorted(diario["anio"].dropna().astype(int).unique().tolist())
sel_year = st.selectbox("Año (vista diaria)", options=years, index=len(years) - 1 if years else 0)


# ==============================================================================
# 1) Producción diaria (línea)
# ==============================================================================
st.markdown("---")
st.subheader("Producción diaria (PRIMERA+SEGUNDA)")

fig = chart_diario_line(diario, int(sel_year))
st.pyplot(fig, use_container_width=True)


# ==============================================================================
# 2) % SEGUNDA diario
# ==============================================================================
st.subheader("% SEGUNDA diario")

di_y = diario[diario["anio"] == int(sel_year)].copy()
if di_y.empty:
    st.info("No hay datos diarios para ese año con los filtros.")
else:
    # Linea %segunda
    fig2, ax2 = plt.subplots(figsize=(11.5, 4.6), dpi=220)
    ax2.plot(di_y["dia"], di_y["pct_segunda"])
    ax2.set_ylabel("% Segunda (sobre PRIMERA+SEGUNDA)")
    ax2.set_title(f"% Segunda diario {sel_year}")
    ax2.set_ylim(0, max(5, float(di_y["pct_segunda"].max()) * 1.15))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax2.grid(axis="y", alpha=0.25)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)


# ==============================================================================
# 3) Días atípicos (picos/caídas)
# ==============================================================================
st.markdown("---")
st.subheader("Días atípicos (picos/caídas)")

d = di_y[["dia", "piezas_total_ps", "pct_segunda"]].copy()
d["piezas_total_ps"] = pd.to_numeric(d["piezas_total_ps"], errors="coerce").fillna(0)

# baseline rolling
d["roll_mean"] = d["piezas_total_ps"].rolling(window=window, min_periods=max(3, window // 3)).mean()
d["roll_std"] = d["piezas_total_ps"].rolling(window=window, min_periods=max(3, window // 3)).std()

# z-score
d["z"] = (d["piezas_total_ps"] - d["roll_mean"]) / d["roll_std"].replace(0, np.nan)
d["z"] = d["z"].replace([np.inf, -np.inf], np.nan).fillna(0)

d["atipico"] = np.abs(d["z"]) >= float(z_thr)
atip = d[d["atipico"]].copy().sort_values("dia", ascending=False)

if atip.empty:
    st.success("No se detectaron días atípicos con esta configuración.")
else:
    st.dataframe(
        atip[["dia", "piezas_total_ps", "pct_segunda", "z"]].rename(
            columns={"piezas_total_ps": "piezas_PS", "pct_segunda": "%_segunda"}
        ),
        use_container_width=True,
    )


# ==============================================================================
# 4) Corte por doc_prefix
# ==============================================================================
st.markdown("---")
st.subheader("Corte por doc_prefix (período filtrado)")

# total piezas por prefix + %segunda dentro de cada prefix
tmp = base_f.copy()
tmp["is_segunda"] = (tmp["calidad"] == "SEGUNDA").astype(int)
tmp["piezas_seg"] = tmp["piezas"] * tmp["is_segunda"]

pref = (
    tmp.groupby("doc_prefix", as_index=False)
    .agg(
        piezas_total=("piezas", "sum"),
        piezas_segunda=("piezas_seg", "sum"),
        dias=("dia", "nunique"),
        skus=("sku", "nunique"),
    )
    .sort_values("piezas_total", ascending=False)
)

pref["pct_segunda"] = np.where(
    pref["piezas_total"] > 0,
    pref["piezas_segunda"] / pref["piezas_total"] * 100,
    0.0,
)

st.dataframe(pref.head(top_prefix_n), use_container_width=True)

# gráfico barras horizontales top prefixes
top_pref = pref.head(top_prefix_n).iloc[::-1].copy()  # invert para barh
figp, axp = plt.subplots(figsize=(10.5, 5.2), dpi=220)
axp.barh(top_pref["doc_prefix"].astype(str), top_pref["piezas_total"])
axp.set_xlabel("Piezas")
axp.set_title(f"Top {top_prefix_n} doc_prefix por piezas (período filtrado)")
axp.grid(axis="x", alpha=0.25)
for spine in ["top", "right"]:
    axp.spines[spine].set_visible(False)
figp.tight_layout()
st.pyplot(figp, use_container_width=True)


# ==============================================================================
# 5) Distribución por hora (si existe hora_ok / fecha_dt)
# ==============================================================================
st.markdown("---")
st.subheader("Distribución por hora (validación de las ~3 entradas/día)")

if "hora_ok" in base_f.columns and "fecha_dt" in base_f.columns and base_f["hora_ok"].astype(bool).any():
    hh = base_f[base_f["hora_ok"].astype(bool)].copy()
    hh["fecha_dt"] = pd.to_datetime(hh["fecha_dt"], errors="coerce")
    hh = hh.dropna(subset=["fecha_dt"]).copy()
    hh["hora"] = hh["fecha_dt"].dt.hour

    by_h = hh.groupby("hora", as_index=False)["piezas"].sum().sort_values("hora")

    fig_h, ax_h = plt.subplots(figsize=(11.5, 4.8), dpi=220)
    ax_h.bar(by_h["hora"], by_h["piezas"])
    ax_h.set_xticks(range(0, 24, 1))
    ax_h.set_xlabel("Hora del día")
    ax_h.set_ylabel("Piezas")
    ax_h.set_title("Piezas por hora (solo registros con hora válida)")
    ax_h.grid(axis="y", alpha=0.25)
    for spine in ["top", "right"]:
        ax_h.spines[spine].set_visible(False)
    fig_h.tight_layout()
    st.pyplot(fig_h, use_container_width=True)

    st.caption("Si ves 2–4 picos marcados, es consistente con el patrón de ~3 entradas al día.")
else:
    st.info("No hay horas válidas suficientes (hora_ok=False o fecha_dt vacío).")


# ==============================================================================
# Descarga de dataset filtrado
# ==============================================================================
st.markdown("---")
st.subheader("Descargar datos filtrados")

csv = base_f.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar CSV (base filtrada)",
    data=csv,
    file_name="entradas_filtradas.csv",
    mime="text/csv",
)
