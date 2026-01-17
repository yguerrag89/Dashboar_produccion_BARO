# core/charts.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _hd_fig(w: float = 12.0, h: float = 5.0, dpi: int = 220):
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    return fig, ax


def _clean_axes(ax):
    ax.grid(axis="y", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# ==============================================================================
# 1) Anual en %
# ==============================================================================

def chart_anual_pct(anual: pd.DataFrame):
    """
    anual debe tener:
      - anio
      - piezas_primera
      - piezas_segunda
      (opcional piezas_total_ps)
    """
    df = anual.copy()
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["anio"]).sort_values("anio")

    # % sobre PRIMERA+SEGUNDA
    ps = (pd.to_numeric(df["piezas_primera"], errors="coerce").fillna(0)
          + pd.to_numeric(df["piezas_segunda"], errors="coerce").fillna(0))
    pct_prim = np.where(ps > 0, df["piezas_primera"] / ps * 100, 0.0)
    pct_seg = np.where(ps > 0, df["piezas_segunda"] / ps * 100, 0.0)

    fig, ax = _hd_fig(11.5, 4.8)
    ax.bar(df["anio"].astype(int), pct_prim, label="PRIMERA")
    ax.bar(df["anio"].astype(int), pct_seg, bottom=pct_prim, label="SEGUNDA")

    ax.set_ylabel("% del total (PRIMERA+SEGUNDA)")
    ax.set_title("Mix anual por calidad (en %)")
    ax.set_ylim(0, 100)
    ax.set_xticks(df["anio"].astype(int).tolist())
    ax.legend()

    _clean_axes(ax)
    fig.tight_layout()
    return fig


# ==============================================================================
# 2) Mensual stacked (PRIMERA vs SEGUNDA)
# ==============================================================================

def chart_mensual_stacked(mensual: pd.DataFrame, year: int):
    """
    mensual debe tener:
      - mes (timestamp)
      - anio
      - mes_num
      - piezas_primera
      - piezas_segunda
    """
    df = mensual.copy()
    df = df[pd.to_numeric(df["anio"], errors="coerce") == int(year)].copy()
    df["mes"] = pd.to_datetime(df["mes"], errors="coerce")
    df = df.dropna(subset=["mes"]).sort_values("mes")

    fig, ax = _hd_fig(11.5, 5.2)
    ax.bar(df["mes"], df["piezas_primera"], label="PRIMERA")
    ax.bar(df["mes"], df["piezas_segunda"], bottom=df["piezas_primera"], label="SEGUNDA")

    ax.set_ylabel("Piezas")
    ax.set_title(f"Producción mensual {year}: PRIMERA vs SEGUNDA")
    ax.legend()

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    _clean_axes(ax)

    fig.tight_layout()
    return fig


# ==============================================================================
# 3) % Segunda mensual
# ==============================================================================

def chart_pct_segunda_mensual(mensual: pd.DataFrame, year: int):
    df = mensual.copy()
    df = df[pd.to_numeric(df["anio"], errors="coerce") == int(year)].copy()
    df["mes"] = pd.to_datetime(df["mes"], errors="coerce")
    df = df.dropna(subset=["mes"]).sort_values("mes")

    fig, ax = _hd_fig(11.5, 4.6)
    ax.plot(df["mes"], df["pct_segunda"])

    ax.set_ylabel("% Segunda (sobre PRIMERA+SEGUNDA)")
    ax.set_title(f"% Segunda mensual {year}")

    ymax = max(5.0, float(df["pct_segunda"].max()) * 1.15) if len(df) else 10.0
    ax.set_ylim(0, ymax)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    _clean_axes(ax)

    fig.tight_layout()
    return fig


# ==============================================================================
# 4) Diario line (total PRIMERA+SEGUNDA)
# ==============================================================================

def chart_diario_line(diario: pd.DataFrame, year: int):
    """
    diario debe tener:
      - dia
      - anio
      - piezas_total_ps
    """
    df = diario.copy()
    df = df[pd.to_numeric(df["anio"], errors="coerce") == int(year)].copy()
    df["dia"] = pd.to_datetime(df["dia"], errors="coerce")
    df = df.dropna(subset=["dia"]).sort_values("dia")

    fig, ax = _hd_fig(11.5, 4.8)
    ax.plot(df["dia"], df["piezas_total_ps"])

    ax.set_ylabel("Piezas (PRIMERA+SEGUNDA)")
    ax.set_title(f"Producción diaria {year}")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    _clean_axes(ax)

    fig.tight_layout()
    return fig
