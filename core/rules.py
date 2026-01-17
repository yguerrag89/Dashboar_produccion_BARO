# core/rules.py
from __future__ import annotations

import re
import unicodedata
import pandas as pd


# ==============================================================================
# Validaciones
# ==============================================================================

def require_columns(df: pd.DataFrame, cols: list[str], where: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        loc = f" ({where})" if where else ""
        raise ValueError(
            f"Faltan columnas{loc}: {missing}. "
            f"Disponibles: {list(df.columns)}"
        )


# ==============================================================================
# Normalización de texto
# ==============================================================================

def _strip_accents(s: str) -> str:
    # Quita acentos para hacer matching más robusto
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )


def normalize_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).replace("\u00A0", " ").strip()
    s = _strip_accents(s)
    s = s.upper()
    s = re.sub(r"\s+", " ", s)
    return s


# ==============================================================================
# Reglas de calidad
# ==============================================================================

FIRST_KEYWORDS = ("BISA", "IMU", "POP")   # PRIMERA
SECOND_KEYWORDS = ("SANSON",)            # SEGUNDA


def classify_calidad(producto_text: str) -> str:
    """
    Regla:
      - SEGUNDA si contiene SANSON
      - PRIMERA si contiene BISA o IMU o POP
      - OTROS si no coincide
    Nota: si llegara a aparecer SANSON y BISA juntos, gana SEGUNDA.
    """
    t = normalize_text(producto_text)

    if any(k in t for k in SECOND_KEYWORDS):
        return "SEGUNDA"
    if any(k in t for k in FIRST_KEYWORDS):
        return "PRIMERA"
    return "OTROS"


# ==============================================================================
# doc_prefix desde documento_ref
# ==============================================================================

def extract_doc_prefix(documento_ref) -> str:
    """
    Extrae un prefijo estable para agrupar (ej: STOCK, VT, BO, etc.)
    Estrategia:
      1) Normaliza texto
      2) Toma primer token (o lo que esté antes de '-')
      3) Se queda con letras iniciales (y algunos casos especiales)
    """
    s = normalize_text(documento_ref)
    if not s:
        return "NA"

    # corta antes de guion si existe
    left = s.split("-", 1)[0].strip()

    # primer token por espacio
    tok = left.split(" ", 1)[0].strip()
    if not tok:
        return "NA"

    # casos comunes
    if tok.startswith("STOCK"):
        return "STOCK"
    if tok.startswith("VT"):
        return "VT"
    if tok.startswith("BO"):
        return "BO"
    if tok.startswith("PO"):
        return "PO"
    if tok.startswith("OC"):
        return "OC"

    # letras iniciales del token (ej: VT1 -> VT)
    m = re.match(r"^([A-Z]+)", tok)
    if m:
        pref = m.group(1)
        # limita tamaño para no crear categorías raras
        return pref[:10] if pref else "OTROS_DOC"

    return "OTROS_DOC"


# ==============================================================================
# Aplicación de reglas al DF
# ==============================================================================

def apply_rules(
    df: pd.DataFrame,
    producto_col: str = "producto",
    documento_ref_col: str = "documento_ref",
) -> pd.DataFrame:
    """
    Añade:
      - producto_norm
      - calidad (PRIMERA/SEGUNDA/OTROS)
      - doc_prefix
    No modifica columnas base; devuelve copia.
    """
    out = df.copy()

    if producto_col not in out.columns:
        # fallback típico si el excel usa otro nombre
        for alt in ["comentario", "NOMBRE  DEL PRODUCTO", "NOMBRE DEL PRODUCTO", "PRODUCTO", "DESCRIPCION"]:
            if alt in out.columns:
                producto_col = alt
                break

    if documento_ref_col not in out.columns:
        for alt in ["ALMACEN Y/O PEDIDO (SERIE Y FOLIO)", "documento", "doc_ref"]:
            if alt in out.columns:
                documento_ref_col = alt
                break

    # crea columnas aunque falten (para no romper pipeline)
    if producto_col not in out.columns:
        out["producto_norm"] = ""
        out["calidad"] = "OTROS"
    else:
        out["producto_norm"] = out[producto_col].apply(normalize_text)
        out["calidad"] = out["producto_norm"].apply(classify_calidad)

    if documento_ref_col in out.columns:
        out["doc_prefix"] = out[documento_ref_col].apply(extract_doc_prefix)
    else:
        out["doc_prefix"] = "NA"

    return out
