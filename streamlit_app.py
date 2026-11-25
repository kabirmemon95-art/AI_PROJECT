"""Glass-style interactive Streamlit dashboard for customer clustering."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clustering_core import (
    FEATURES,
    evaluate_kmeans,
    fit_dbscan,
    fit_kmeans,
    make_download,
    make_profile,
    pca_projection,
    prepare_features,
)

st.set_page_config(
    page_title="Customer Intelligence | Clustering Lab",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ Visual system ------------------------------
# One carefully tuned appearance: midnight glass with restrained neon accents.
# There is intentionally no light/dark mode switch in this version.
DARK_MODE = True

st.markdown(
    '<div class="brand-lockup"><span class="brand-symbol">✧</span>'
    '<span class="brand-name">CLUSTERING LAB</span>'
    '<span class="brand-divider">/</span>'
    '<span class="brand-context">UNSUPERVISED LEARNING WORKSPACE</span></div>',
    unsafe_allow_html=True,
)

theme_vars = r"""
  color-scheme: dark;
  --app-bg: radial-gradient(ellipse at 9% 4%, rgba(24, 190, 218, .19), transparent 30%),
            radial-gradient(ellipse at 92% 9%, rgba(126, 91, 245, .21), transparent 33%),
            radial-gradient(ellipse at 78% 88%, rgba(30, 201, 169, .12), transparent 31%),
            radial-gradient(ellipse at 8% 91%, rgba(71, 105, 229, .12), transparent 34%),
            linear-gradient(135deg, #07111f 0%, #0b1728 48%, #101a31 100%);
  --ink: #eff8ff;
  --ink-soft: #c6d8e9;
  --muted: #8ea5bd;
  --glass: rgba(13, 30, 50, .68);
  --glass-strong: rgba(15, 35, 59, .88);
  --glass-line: rgba(111, 211, 239, .28);
  --glass-shadow: 0 22px 60px rgba(0, 0, 0, .36), 0 0 24px rgba(28, 190, 218, .09), inset 0 1px 0 rgba(232, 250, 255, .19);
  --sidebar-bg: linear-gradient(155deg, rgba(8, 21, 38, .96), rgba(12, 30, 51, .91));
  --metric-bg: linear-gradient(145deg, rgba(24, 53, 77, .88), rgba(12, 29, 49, .80));
  --input-bg: rgba(6, 19, 34, .86);
  --input-border: rgba(106, 204, 232, .30);
  --button-bg: linear-gradient(135deg, rgba(15, 155, 180, .28), rgba(101, 79, 205, .30));
  --button-ink: #effcff;
  --selected-bg: linear-gradient(135deg, rgba(20, 176, 198, .28), rgba(111, 87, 229, .34));
  --chart-text: #c9deef;
  --grid-line: rgba(132, 186, 220, .16);
  --hero-ink: #f4fbff;
  --hero-copy: #bdd2e5;
  --accent: #55e2f2;
  --accent-2: #a995ff;
  --glass-note-bg: rgba(15, 39, 60, .82);
  --edge-glow: rgba(215, 250, 255, .62);
  --neon-cyan: #4de4f3;
  --neon-blue: #62a8ff;
  --neon-violet: #a58aff;
  --neon-mint: #4ce0bd;
"""

GLASS_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { __THEME_VARS__ }
html, body, [data-testid="stAppViewContainer"] {
  color:var(--ink) !important;
  font-family:'DM Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important;
  background:var(--app-bg) !important;
  background-attachment:fixed !important;
  min-width:0 !important;
  transition:background .45s ease,color .35s ease;
}
[data-testid="stAppViewContainer"] {
  position:relative;
  isolation:isolate;
}
[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after {
  content:'';position:fixed;pointer-events:none;z-index:0;border-radius:999px;
  filter:blur(88px);opacity:.58;animation:aurora-float 19s ease-in-out infinite alternate;
}
[data-testid="stAppViewContainer"]::before {
  width:460px;height:460px;top:7%;right:-190px;
  background:radial-gradient(circle,rgba(62,207,238,.24),rgba(62,207,238,0) 70%);
}
[data-testid="stAppViewContainer"]::after {
  width:410px;height:410px;bottom:0;left:-175px;
  background:radial-gradient(circle,rgba(143,107,255,.24),rgba(143,107,255,0) 72%);
  animation-delay:-8s;
}
[data-testid="stHeader"] {
  background:rgba(7,17,31,.52) !important;
  border-bottom:1px solid rgba(119,207,234,.13);
  backdrop-filter:blur(25px) saturate(175%);-webkit-backdrop-filter:blur(25px) saturate(175%);
}
[data-testid="stToolbar"] { right:1rem; }
[data-testid="stAppViewContainer"] > .main {
  background:transparent !important;position:relative;z-index:1;min-width:0;
}
.block-container { max-width:1510px;padding-top:.85rem;padding-bottom:3.6rem;min-width:0; }

/* Compact brand line; the hero below carries the project name. */
.brand-lockup { display:flex;align-items:center;gap:10px;min-height:43px;color:var(--ink);margin-bottom:4px; }
.brand-symbol {
  display:inline-flex;align-items:center;justify-content:center;width:35px;height:35px;border-radius:13px;
  color:#061522;font-size:1.25rem;font-weight:900;
  background:linear-gradient(135deg,#7af2f5 0%,#4eb9f4 48%,#a38aff 100%);
  box-shadow:0 0 0 1px rgba(179,245,255,.48),0 0 20px rgba(58,211,239,.28),inset 0 1px 0 rgba(255,255,255,.82);
}
.brand-name { font:800 .92rem 'Manrope',sans-serif;letter-spacing:.13em;color:#effaff; }
.brand-divider { color:#64829d;opacity:.8; }
.brand-context { color:#8faac3;font-size:.62rem;font-weight:800;letter-spacing:.15em; }

/* Bright, easy-to-find Streamlit sidebar collapse / expand affordance. */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stHeader"] button[kind="header"] {
  min-width:44px !important;width:44px !important;min-height:44px !important;height:44px !important;
  border-radius:15px !important;border:1px solid rgba(91,226,246,.62) !important;
  color:#eafaff !important;background:linear-gradient(145deg,rgba(20,72,97,.94),rgba(37,43,90,.94)) !important;
  box-shadow:0 0 0 3px rgba(64,214,239,.10),0 0 25px rgba(45,208,237,.24),0 9px 28px rgba(0,0,0,.32),inset 0 1px 0 rgba(255,255,255,.30) !important;
  backdrop-filter:blur(24px) saturate(185%) !important;-webkit-backdrop-filter:blur(24px) saturate(185%) !important;
  transition:transform .22s ease,box-shadow .22s ease,filter .22s ease !important;z-index:1002 !important;
}
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="stHeader"] button[kind="header"]:hover {
  transform:translateY(-1px) scale(1.06);filter:brightness(1.16);
  box-shadow:0 0 0 4px rgba(64,214,239,.14),0 0 31px rgba(45,208,237,.34),0 12px 30px rgba(0,0,0,.38),inset 0 1px 0 rgba(255,255,255,.38) !important;
}
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stHeader"] button[kind="header"] svg { color:#9af5ff !important;fill:currentColor; }

/* Sidebar: darker, more legible frosted pane. */
[data-testid="stSidebar"] {
  background:var(--sidebar-bg) !important;border-right:1px solid rgba(90,204,232,.23);
  box-shadow:15px 0 48px rgba(0,0,0,.26),inset -1px 0 0 rgba(205,247,255,.06);
  backdrop-filter:blur(42px) saturate(190%);-webkit-backdrop-filter:blur(42px) saturate(190%);
}
[data-testid="stSidebar"] > div { background:transparent !important; }
[data-testid="stSidebar"] * { color:var(--ink); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:var(--muted) !important; }
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:var(--ink) !important; }
[data-testid="stSidebar"] hr { border-color:rgba(102,191,222,.18) !important; }

h1,h2,h3,h4 { font-family:'Manrope','DM Sans',sans-serif !important;letter-spacing:-.04em;color:var(--ink) !important; }
h1 { font-weight:800 !important; }
p,label,.stMarkdown { color:var(--ink-soft); }
small,.stCaption,[data-testid="stCaptionContainer"] { color:var(--muted) !important; }
::selection { background:rgba(64,220,241,.34);color:#fff; }

/* Glass metric tiles. */
[data-testid="stMetric"] {
  position:relative;overflow:hidden;background:var(--metric-bg) !important;
  border:1px solid rgba(107,213,239,.28) !important;border-radius:22px !important;padding:18px 20px !important;
  box-shadow:var(--glass-shadow) !important;backdrop-filter:blur(30px) saturate(185%);
  -webkit-backdrop-filter:blur(30px) saturate(185%);
  transition:transform .28s ease,box-shadow .28s ease,border-color .3s ease;animation:glass-rise .62s both;
}
[data-testid="stMetric"]::before {
  content:'';position:absolute;top:0;left:8%;right:8%;height:1px;
  background:linear-gradient(90deg,transparent,rgba(179,247,255,.85),rgba(153,133,255,.55),transparent);
}
[data-testid="stMetric"]::after {
  content:'';position:absolute;width:150px;height:120px;right:-90px;top:-84px;border-radius:50%;
  background:radial-gradient(circle,rgba(62,214,240,.16),transparent 70%);pointer-events:none;
}
[data-testid="stMetric"]:hover {
  transform:translateY(-3px);border-color:rgba(94,226,246,.48) !important;
  box-shadow:0 22px 48px rgba(0,0,0,.34),0 0 23px rgba(39,202,232,.13),inset 0 1px 0 rgba(235,252,255,.24) !important;
}
[data-testid="stMetricLabel"] { color:#91abc5 !important;font-size:.72rem !important;font-weight:800 !important;letter-spacing:.10em; }
[data-testid="stMetricValue"] { color:#f1faff !important;font-family:'Manrope',sans-serif;font-weight:800; }
[data-testid="stMetricDelta"] { color:#55e0bd !important; }

/* Plotly charts live inside clearly outlined glass panes. */
[data-testid="stPlotlyChart"] {
  display:block !important;width:100% !important;max-width:100% !important;min-width:0 !important;
  overflow:hidden !important;background:linear-gradient(145deg,rgba(17,40,63,.80),rgba(9,24,43,.76)) !important;
  border:1px solid rgba(91,207,237,.26) !important;border-radius:22px !important;padding:7px 7px 0 !important;
  box-sizing:border-box !important;box-shadow:0 18px 44px rgba(0,0,0,.28),0 0 20px rgba(39,194,228,.07),inset 0 1px 0 rgba(230,251,255,.13) !important;
  backdrop-filter:blur(28px) saturate(180%);-webkit-backdrop-filter:blur(28px) saturate(180%);animation:glass-rise .72s both;
}
[data-testid="stPlotlyChart"] > div,
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container,
[data-testid="stPlotlyChart"] .svg-container {
  width:100% !important;max-width:100% !important;min-width:0 !important;
}
[data-testid="stPlotlyChart"] .main-svg { max-width:100% !important; }

/* Tables, tabs, buttons and form controls. */
[data-testid="stDataFrame"],[data-testid="stTable"] {
  width:100% !important;max-width:100% !important;border:1px solid rgba(103,202,232,.24) !important;
  border-radius:17px;overflow:auto;box-shadow:0 13px 30px rgba(0,0,0,.22);background:rgba(10,25,43,.76) !important;
}
[data-testid="stTabs"] [role="tablist"] {
  display:flex;gap:7px;border:1px solid rgba(106,205,234,.24);border-radius:17px;
  background:rgba(10,26,45,.70);padding:6px;box-shadow:inset 0 1px 0 rgba(232,251,255,.12),0 12px 30px rgba(0,0,0,.20);
  backdrop-filter:blur(28px) saturate(180%);-webkit-backdrop-filter:blur(28px) saturate(180%);
}
[data-testid="stTabs"] button[role="tab"] {
  flex:0 0 auto;border:1px solid transparent;background:transparent;border-radius:12px;
  padding:10px 16px;color:#8fa8c2;font-weight:750;white-space:nowrap;transition:all .22s ease;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  background:linear-gradient(135deg,rgba(22,157,187,.23),rgba(111,83,219,.27));
  color:#8df0ff !important;border-color:rgba(111,220,241,.30);
  box-shadow:0 0 18px rgba(49,199,230,.12),inset 0 1px 0 rgba(241,253,255,.17);
}
.stButton > button,.stDownloadButton > button {
  border:1px solid rgba(106,217,239,.34) !important;border-radius:14px !important;
  background:linear-gradient(135deg,rgba(16,129,158,.32),rgba(88,70,177,.34)) !important;
  color:#effcff !important;font-weight:750 !important;
  box-shadow:0 9px 24px rgba(0,0,0,.22),0 0 16px rgba(55,195,225,.09),inset 0 1px 0 rgba(241,253,255,.19) !important;
  transition:transform .22s ease,box-shadow .22s ease,filter .22s ease !important;
}
.stButton > button:hover,.stDownloadButton > button:hover {
  transform:translateY(-2px);filter:brightness(1.12);
  box-shadow:0 13px 30px rgba(0,0,0,.28),0 0 23px rgba(56,210,237,.20),inset 0 1px 0 rgba(255,255,255,.28) !important;
}
[data-baseweb="select"] > div,[data-baseweb="input"] > div,[data-baseweb="textarea"] > div {
  background:var(--input-bg) !important;border-color:var(--input-border) !important;
  border-radius:13px !important;color:var(--ink) !important;
  box-shadow:inset 0 1px 2px rgba(0,0,0,.22),0 4px 13px rgba(0,0,0,.10);
}
[data-baseweb="select"] *,[data-baseweb="input"] input,[data-baseweb="textarea"] textarea { color:var(--ink) !important; }
[data-baseweb="popover"] [role="listbox"],[data-baseweb="menu"] {
  background:rgba(10,25,44,.97) !important;border:1px solid rgba(99,207,236,.28) !important;
  backdrop-filter:blur(25px);-webkit-backdrop-filter:blur(25px);
}
[data-testid="stRadio"] label,[data-testid="stCheckbox"] label { color:var(--ink-soft) !important; }
[data-testid="stSlider"] [role="slider"] {
  background:#59e2f1 !important;border:2px solid #e8fcff !important;
  box-shadow:0 0 0 5px rgba(69,219,239,.15),0 0 17px rgba(69,219,239,.38) !important;
}
[data-testid="stExpander"] {
  border:1px solid rgba(101,205,234,.23) !important;border-radius:17px !important;background:rgba(12,29,49,.66) !important;
  box-shadow:0 12px 28px rgba(0,0,0,.18),inset 0 1px 0 rgba(233,252,255,.12);
  backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
}
[data-testid="stAlert"] { border-radius:16px;backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px); }
hr { border-color:rgba(101,196,224,.17) !important; }

/* Hero: a luminous but restrained translucent glass sheet. */
.hero-shell {
  position:relative;isolation:isolate;overflow:hidden;padding:34px 36px 32px;
  border:1px solid rgba(117,222,244,.37);border-radius:29px;
  background:
    radial-gradient(ellipse at 86% 8%,rgba(127,100,255,.25),transparent 36%),
    radial-gradient(ellipse at 8% 100%,rgba(40,208,220,.20),transparent 40%),
    linear-gradient(126deg,rgba(16,43,66,.88),rgba(17,35,61,.76) 54%,rgba(37,31,76,.78));
  box-shadow:0 28px 74px rgba(0,0,0,.36),0 0 35px rgba(49,196,225,.10),
    inset 0 1.5px 0 rgba(233,253,255,.30),inset 0 -1px 0 rgba(133,169,255,.12);
  backdrop-filter:blur(42px) saturate(195%);-webkit-backdrop-filter:blur(42px) saturate(195%);
  margin:10px 0 24px;animation:glass-rise .76s cubic-bezier(.2,.75,.25,1) both;
}
.hero-shell:before {
  content:'';position:absolute;inset:0;z-index:-1;pointer-events:none;
  background:linear-gradient(115deg,rgba(255,255,255,.075),transparent 27%,transparent 72%,rgba(123,119,255,.09));
}
.hero-shell:after {
  content:'';position:absolute;width:450px;height:280px;right:-150px;top:-190px;z-index:-1;
  background:radial-gradient(circle,rgba(88,221,245,.24),rgba(115,99,255,.12) 42%,transparent 72%);
  filter:blur(12px);animation:aurora-float 13s ease-in-out infinite alternate;
}
.hero-kicker { position:relative;z-index:1;color:var(--accent);font-size:.68rem;font-weight:850;letter-spacing:.19em;text-transform:uppercase; }
.hero-title { position:relative;z-index:1;font:800 clamp(2rem,4vw,3.35rem)/1.04 'Manrope',sans-serif;letter-spacing:-.065em;margin:12px 0 10px;color:var(--hero-ink); }
.hero-title span {
  background:linear-gradient(95deg,#54e5ef,#65aaff 47%,#b39aff 78%,#73e4cb);
  -webkit-background-clip:text;background-clip:text;color:transparent;
  filter:drop-shadow(0 0 17px rgba(83,208,243,.20));
}
.hero-copy { max-width:780px;color:var(--hero-copy);line-height:1.68;margin:0;position:relative;z-index:1; }
.hero-chip {
  display:inline-block;margin-top:18px;margin-right:8px;border:1px solid rgba(121,221,242,.26);border-radius:99px;
  padding:7px 12px;color:#d9f6ff;font-size:.69rem;font-weight:750;
  background:linear-gradient(130deg,rgba(38,157,184,.20),rgba(111,91,191,.18));
  box-shadow:inset 0 1px 0 rgba(238,253,255,.20),0 5px 17px rgba(0,0,0,.16);
  backdrop-filter:blur(18px) saturate(180%);-webkit-backdrop-filter:blur(18px) saturate(180%);
}
.section-kicker { color:var(--accent);letter-spacing:.15em;font-size:.67rem;font-weight:850;text-transform:uppercase;margin-bottom:5px; }
.section-title { font:800 1.55rem 'Manrope',sans-serif;color:var(--ink);letter-spacing:-.045em;margin:0 0 4px; }
.section-copy { color:var(--muted);font-size:.90rem;margin:0 0 15px; }
.glass-note {
  border:1px solid rgba(100,207,235,.27);border-left:3px solid #54dfed;border-radius:15px;
  background:linear-gradient(125deg,rgba(14,46,66,.82),rgba(24,34,67,.78));padding:13px 15px;
  color:#cce3f3;font-size:.87rem;box-shadow:inset 0 1px 0 rgba(234,252,255,.13),0 9px 23px rgba(0,0,0,.18);
  backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);
}
.pill {
  display:inline-block;border:1px solid rgba(112,216,239,.33);color:#8aeeff;background:rgba(20,91,112,.27);
  border-radius:100px;padding:6px 11px;font-size:.67rem;font-weight:850;letter-spacing:.08em;
  box-shadow:inset 0 1px 0 rgba(239,253,255,.17),0 0 17px rgba(56,196,226,.10);
}
.footer-line { color:#7f9ab5;font-size:.70rem;text-align:center;padding:22px 0 3px;letter-spacing:.08em; }

@keyframes glass-rise { from { opacity:0;transform:translateY(12px); } to { opacity:1;transform:translateY(0); } }
@keyframes aurora-float {
  0% { transform:translate3d(0,0,0) scale(1); }
  48% { transform:translate3d(-20px,17px,0) scale(1.07); }
  100% { transform:translate3d(17px,-13px,0) scale(.97); }
}
@media (max-width:800px) {
  .block-container { padding:.72rem .68rem 2.5rem !important; }
  .hero-shell { padding:23px 19px;border-radius:23px;margin-top:7px; }
  .hero-title { font-size:clamp(1.9rem,8vw,2.55rem); }
  .brand-context { font-size:.52rem;letter-spacing:.09em; }
  [data-testid="stTabs"] [role="tablist"] { display:flex !important;flex-wrap:nowrap !important;overflow-x:auto !important;overflow-y:hidden !important;scrollbar-width:thin;max-width:100%; }
  [data-testid="stTabs"] button[role="tab"] { flex:0 0 auto !important;white-space:nowrap !important;padding:9px 12px !important;font-size:.81rem !important; }
  [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important;gap:.7rem !important; }
  [data-testid="stHorizontalBlock"]:not(:has(> [data-testid="column"]:nth-child(3))) > [data-testid="column"] {
    flex:1 1 100% !important;width:100% !important;min-width:100% !important;max-width:100% !important;
  }
  [data-testid="stHorizontalBlock"]:has(> [data-testid="column"]:nth-child(3)) > [data-testid="column"] {
    flex:1 1 calc(50% - .45rem) !important;width:calc(50% - .45rem) !important;
    min-width:calc(50% - .45rem) !important;max-width:calc(50% - .45rem) !important;
  }
  [data-testid="stPlotlyChart"] { width:100% !important;max-width:100% !important;min-width:0 !important;padding:3px !important;border-radius:16px !important; }
  [data-testid="stPlotlyChart"] > div,[data-testid="stPlotlyChart"] .js-plotly-plot,
  [data-testid="stPlotlyChart"] .plot-container,[data-testid="stPlotlyChart"] .svg-container {
    width:100% !important;max-width:100% !important;min-width:0 !important;
  }
  [data-testid="stDataFrame"] { max-width:100% !important;overflow-x:auto !important; }
  [data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapseButton"] button,
  [data-testid="stSidebarCollapsedControl"],[data-testid="stSidebarCollapsedControl"] button {
    min-width:42px !important;width:42px !important;min-height:42px !important;height:42px !important;
  }
}
@media (max-width:430px) {
  .block-container { padding-left:.5rem !important;padding-right:.5rem !important; }
  .brand-name { font-size:.78rem;letter-spacing:.09em; }
  .brand-context { font-size:.44rem; }
  [data-testid="stMetric"] { padding:12px 11px !important;border-radius:16px !important; }
  [data-testid="stMetricValue"] { font-size:1.3rem !important; }
  [data-testid="stPlotlyChart"] { padding:2px !important; }
  .hero-chip { margin-top:11px;margin-right:5px;padding:6px 9px;font-size:.64rem; }
}
@supports not selector(:has(*)) {
  @media (max-width:800px) {
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
      flex:1 1 100% !important;width:100% !important;min-width:100% !important;max-width:100% !important;
    }
  }
}
@media (prefers-reduced-motion:reduce) {
  *,*::before,*::after { animation-duration:.01ms !important;animation-iteration-count:1 !important;transition-duration:.01ms !important;scroll-behavior:auto !important; }
}
</style>
"""
GLASS_CSS = GLASS_CSS.replace("__THEME_VARS__", theme_vars)
st.markdown(GLASS_CSS, unsafe_allow_html=True)

def section_heading(kicker: str, title: str, copy: str = "") -> None:
    safe_kicker = str(kicker)
    safe_title = str(title)
    safe_copy = str(copy)
    st.markdown(
        f'<div class="section-kicker">{safe_kicker}</div>'
        f'<div class="section-title">{safe_title}</div>'
        f'<div class="section-copy">{safe_copy}</div>',
        unsafe_allow_html=True,
    )


def plot_layout(fig: go.Figure, height: int = 390) -> go.Figure:
    chart_text = "#d8e1f5" if DARK_MODE else "#52627d"
    title_text = "#f2f5ff" if DARK_MODE else "#263957"
    grid_line = "rgba(190,205,240,.17)" if DARK_MODE else "rgba(104,128,166,.14)"
    hover_bg = "rgba(28,35,61,.98)" if DARK_MODE else "rgba(255,255,255,.98)"
    hover_border = "#6578a8" if DARK_MODE else "#cbd8ee"
    fig.update_layout(
        height=height,
        width=None,
        autosize=True,
        margin=dict(l=12, r=12, t=54, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=chart_text, family="DM Sans, sans-serif", size=12),
        title=dict(font=dict(size=15, color=title_text)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=chart_text, size=11)),
        hoverlabel=dict(bgcolor=hover_bg, bordercolor=hover_border, font_color=title_text),
    )
    fig.update_xaxes(
        gridcolor=grid_line, zerolinecolor=grid_line, linecolor=grid_line,
        tickfont=dict(color=chart_text, size=10), title_font=dict(color=chart_text, size=11),
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor=grid_line, zerolinecolor=grid_line, linecolor=grid_line,
        tickfont=dict(color=chart_text, size=10), title_font=dict(color=chart_text, size=11),
        automargin=True,
    )
    return fig


def render_plot(fig: go.Figure) -> None:
    """Render Plotly figures at the available viewport width, including phones."""
    fig.update_layout(autosize=True, width=None)
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        config={"responsive": True, "displayModeBar": False, "scrollZoom": False},
    )


@st.cache_data(show_spinner="Connecting to the UCI repository and loading 440 customer records...")
def load_uci_dataset() -> pd.DataFrame:
    """Fetch and cache UCI dataset 292. Channel and Region remain metadata."""
    from ucimlrepo import fetch_ucirepo

    wholesale = fetch_ucirepo(id=292)
    features = wholesale.data.features.copy()
    targets = wholesale.data.targets.copy()
    return pd.concat([features, targets], axis=1)


# ------------------------------ Hero section -------------------------------
st.markdown(
    """
    <div class="hero-shell">
      <div class="hero-kicker">UNSUPERVISED LEARNING LAB · LIVE ANALYTICS</div>
      <div class="hero-title">Customer <span>Intelligence</span></div>
      <p class="hero-copy">Explore wholesale customer behaviour with two clustering methods.
      Tune K-Means, discover dense regions with DBSCAN, compare the patterns and export
      your findings — all from one interactive workspace.</p>
      <span class="hero-chip">✦ K-MEANS</span><span class="hero-chip">◉ DBSCAN</span>
      <span class="hero-chip">⌁ PCA VISUALS</span><span class="hero-chip">⇩ CSV EXPORT</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------ Sidebar: data -------------------------------
with st.sidebar:
    st.markdown('<div class="pill">CONTROL ROOM</div>', unsafe_allow_html=True)
    st.markdown("## Workspace settings")
    data_source = st.radio(
        "DATA SOURCE",
        ["Official UCI dataset", "Upload my CSV"],
        help="An uploaded CSV must include the six spending columns listed below.",
    )
    uploaded_file = None
    if data_source == "Upload my CSV":
        uploaded_file = st.file_uploader("Choose a customer CSV", type=["csv"])

# -------------------------------- Load data ---------------------------------
try:
    if data_source == "Official UCI dataset":
        df = load_uci_dataset()
        source_label = "UCI Wholesale Customers · Dataset 292"
    else:
        if uploaded_file is None:
            st.info("Upload your CSV in the sidebar to open the dashboard.")
            st.stop()
        df = pd.read_csv(BytesIO(uploaded_file.getvalue()))
        source_label = uploaded_file.name

    X, X_scaled, missing_count = prepare_features(df)
except Exception as exc:  # show actionable message in app
    st.error(f"Unable to load or prepare the dataset: {exc}")
    st.info("Check your connection for the official UCI dataset, or verify that your CSV contains Fresh, Milk, Grocery, Frozen, Detergents_Paper and Delicassen.")
    st.stop()

# ------------------------------ Sidebar: models ----------------------------
max_allowed_k = min(12, len(X) - 1)
with st.sidebar:
    st.divider()
    st.markdown("### K-MEANS PARAMETERS")
    max_k = st.slider(
        "Highest K to evaluate",
        min_value=3,
        max_value=max(3, max_allowed_k),
        value=min(10, max_allowed_k),
        step=1,
        help="The app evaluates every K from 2 up to this value.",
    )
    k_selection_mode = st.radio(
        "K selection mode",
        ["Manual selection", "Auto recommendation"],
        index=0,
        help="Manual lets you choose the number of groups. Auto uses the highest silhouette score among the tested K values.",
    )

    st.divider()
    st.markdown("### DBSCAN PARAMETERS")
    eps_value = st.slider(
        "Epsilon (eps)", 0.20, 4.00, 1.20, 0.05,
        help="Neighbourhood radius. The K-distance chart helps guide this choice.",
    )
    min_samples = st.slider(
        "Minimum samples", 3, min(20, len(X)), min(7, len(X)), 1,
        help="Minimum number of nearby records, including the point itself, needed for a core point.",
    )
    st.divider()
    st.caption("K is not restricted by the Channel labels. The spending columns are clustered independently of Channel and Region.")

# -------------------------- Compute clustering -----------------------------
with st.spinner("Testing K-Means options and fitting the selected models..."):
    k_evaluation = evaluate_kmeans(X_scaled, max_k=max_k)
    recommended_k = int(k_evaluation.loc[k_evaluation["Silhouette Score"].idxmax(), "K"])
    if k_selection_mode == "Auto recommendation":
        chosen_k = recommended_k
    else:
        chosen_k = st.session_state.get("manual_k_value", 2)
        chosen_k = min(max(2, int(chosen_k)), max_k)
    kmeans_labels, kmeans_score, kmeans_model = fit_kmeans(X_scaled, chosen_k)
    dbscan_labels, k_distances, dbscan_score, dbscan_cluster_count, dbscan_noise_count = fit_dbscan(
        X_scaled, eps=eps_value, min_samples=min_samples
    )

# PCA is a display layer only; models above use all six scaled features.
X_2d, pca_pct = pca_projection(X_scaled)
visual_data = pd.DataFrame({"PC 1": X_2d[:, 0], "PC 2": X_2d[:, 1]})
visual_data["K-Means cluster"] = [f"Cluster {int(v)}" for v in kmeans_labels]
visual_data["DBSCAN cluster"] = ["Noise (-1)" if v == -1 else f"Cluster {int(v)}" for v in dbscan_labels]
for feature in FEATURES:
    visual_data[feature] = X[feature].to_numpy()

results = df.copy()
results["KMeans_Cluster"] = kmeans_labels
results["DBSCAN_Cluster"] = dbscan_labels
k_profile = make_profile(X, kmeans_labels)
d_profile = make_profile(X, dbscan_labels)
noise_pct = dbscan_noise_count / len(df) * 100

# Keep the manual K selection valid if the tested maximum is changed.
if "manual_k_value" not in st.session_state:
    st.session_state.manual_k_value = 2
st.session_state.manual_k_value = min(
    max(2, int(st.session_state.manual_k_value)), max_k
)

# --------------------------------- Tabs -------------------------------------
tab_overview, tab_kmeans, tab_dbscan, tab_compare = st.tabs(
    ["◈ Overview", "✣ K-Means Lab", "◉ DBSCAN Lab", "⌁ Compare & Export"]
)

with tab_overview:
    section_heading("01 / DATASET SNAPSHOT", "The customer landscape", "A quick view of the records and spending signals going into the clustering models.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CUSTOMERS", f"{len(df):,}")
    m2.metric("SPENDING FEATURES", "06")
    m3.metric("MISSING VALUES FILLED", f"{missing_count:,}")
    m4.metric("DATA SOURCE", "UCI" if data_source == "Official UCI dataset" else "CSV")

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        spend_means = X.mean().sort_values(ascending=True).rename("Average annual spend").reset_index()
        spend_means.columns = ["Category", "Average annual spend"]
        fig_spend = px.bar(
            spend_means, x="Average annual spend", y="Category", orientation="h",
            title="Average spending by product category", color="Average annual spend",
            color_continuous_scale=[[0, "#70b7e8"], [0.55, "#8d87e8"], [1, "#6bc8ba"]],
        )
        fig_spend.update_coloraxes(showscale=False)
        plot_layout(fig_spend, 410)
        render_plot(fig_spend)
    with right:
        st.markdown("#### Dataset notes")
        st.markdown(f"**Source:** {source_label}")
        st.markdown("**Features used:** six annual-spending categories.")
        st.markdown("**Preparation:** `log1p` transform followed by standard scaling.")
        st.markdown("**Algorithms:** K-Means and DBSCAN.")
        st.markdown('<div class="glass-note"><b>Important:</b> UCI’s <code>Channel</code> column has two business categories (Horeca and Retail), while <code>Region</code> has three categories. Those are metadata labels — they do not limit the number of K-Means clusters. K can be selected independently.</div>', unsafe_allow_html=True)
        st.markdown(" ")
        if "Channel" in df.columns:
            channel_counts = df["Channel"].value_counts(dropna=False).sort_index()
            st.markdown("**Channel distribution (not used as clustering input)**")
            st.dataframe(channel_counts.rename_axis("Channel code").rename("Customers").to_frame(), use_container_width=True)
        if "Region" in df.columns:
            region_counts = df["Region"].value_counts(dropna=False).sort_index()
            st.markdown("**Region distribution (not used as clustering input)**")
            st.dataframe(region_counts.rename_axis("Region code").rename("Customers").to_frame(), use_container_width=True)

    with st.expander("Preview customer records"):
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)
    with st.expander("Feature statistics"):
        st.dataframe(X.describe().T.round(2), use_container_width=True)

with tab_kmeans:
    section_heading("02 / CENTROID-BASED GROUPING", "K-Means control studio", "Choose your K value and watch the group assignments and evaluation charts update.")
    if k_selection_mode == "Manual selection":
        manual_k = st.select_slider(
            "Choose the number of clusters (K)",
            options=list(range(2, max_k + 1)),
            value=int(chosen_k),
            key="manual_k_value",
            help="Every option from 2 through the selected maximum is available. K=2 is not a dataset restriction.",
        )
        if int(manual_k) != chosen_k:
            chosen_k = int(manual_k)
            kmeans_labels, kmeans_score, kmeans_model = fit_kmeans(X_scaled, chosen_k)
            results["KMeans_Cluster"] = kmeans_labels
            k_profile = make_profile(X, kmeans_labels)
            visual_data["K-Means cluster"] = [f"Cluster {int(v)}" for v in kmeans_labels]

    k_counts = pd.Series(kmeans_labels).value_counts().sort_index()
    a, b, c, d = st.columns(4)
    a.metric("SELECTED K", chosen_k)
    b.metric("SILHOUETTE SCORE", f"{kmeans_score:.4f}")
    c.metric("AUTO RECOMMENDATION", f"K = {recommended_k}")
    d.metric("LARGEST CLUSTER", f"{int(k_counts.max()):,} customers")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        fig_elbow = px.line(k_evaluation, x="K", y="Inertia", markers=True, title="Elbow method · lower is better")
        fig_elbow.add_vline(x=chosen_k, line_dash="dash", line_color="#FFB86B", annotation_text=f"K = {chosen_k}")
        plot_layout(fig_elbow, 350)
        render_plot(fig_elbow)
    with col2:
        fig_sil = px.line(k_evaluation, x="K", y="Silhouette Score", markers=True, title="Silhouette score · higher is better")
        fig_sil.add_vline(x=chosen_k, line_dash="dash", line_color="#62A8FF", annotation_text=f"K = {chosen_k}")
        plot_layout(fig_sil, 350)
        render_plot(fig_sil)

    col3, col4 = st.columns([1.2, 0.8], gap="large")
    with col3:
        fig_k_scatter = px.scatter(
            visual_data, x="PC 1", y="PC 2", color="K-Means cluster",
            hover_data=FEATURES, title="Customer map · K-Means assignments",
            color_discrete_sequence=["#4DE4F3", "#A58AFF", "#4CE0BD", "#FFBC6E", "#FF7FB2", "#62A8FF"], opacity=0.84,
        )
        plot_layout(fig_k_scatter, 470)
        render_plot(fig_k_scatter)
    with col4:
        count_df = k_counts.rename_axis("Cluster").rename("Customers").reset_index()
        fig_counts = px.bar(count_df, x="Cluster", y="Customers", color="Cluster", title="Customers in each cluster", text="Customers")
        fig_counts.update_traces(textposition="outside")
        fig_counts.update_layout(showlegend=False)
        plot_layout(fig_counts, 300)
        render_plot(fig_counts)
        st.markdown("**Average spending profile**")
        st.dataframe(k_profile, use_container_width=True)

    with st.expander("K-Means evaluation table"):
        st.dataframe(k_evaluation.round(4), use_container_width=True, hide_index=True)

with tab_dbscan:
    section_heading("03 / DENSITY-BASED GROUPING", "DBSCAN explorer", "Tune the neighbourhood radius and minimum samples. The K-distance curve helps guide Epsilon.")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("EPSILON", f"{eps_value:.2f}")
    d2.metric("MINIMUM SAMPLES", min_samples)
    d3.metric("CLUSTERS FOUND", dbscan_cluster_count)
    d4.metric("NOISE POINTS", f"{dbscan_noise_count:,} · {noise_pct:.1f}%")

    fig_kdist = go.Figure()
    fig_kdist.add_trace(go.Scatter(
        x=np.arange(1, len(k_distances) + 1), y=k_distances,
        mode="lines", name=f"{min_samples}-neighbor distance",
        line=dict(color="#4DE4F3", width=3.0), fill="tozeroy",
        fillcolor="rgba(91,141,239,.12)",
    ))
    fig_kdist.add_hline(y=eps_value, line_dash="dash", line_color="#FFB86B", annotation_text=f"Current eps = {eps_value:.2f}")
    fig_kdist.update_layout(title="K-distance curve · sorted neighbour distances", xaxis_title="Customers (sorted)", yaxis_title="Distance")
    plot_layout(fig_kdist, 380)
    render_plot(fig_kdist)

    if dbscan_cluster_count < 2:
        st.warning("DBSCAN found fewer than two clusters with these settings. Try adjusting Epsilon or Minimum Samples. One cluster is not a coding error; it means the current density settings did not split the records into multiple dense groups.")
    else:
        st.success(f"DBSCAN identified {dbscan_cluster_count} clusters. Noise points are shown separately as -1.")

    db_left, db_right = st.columns([1.2, 0.8], gap="large")
    with db_left:
        fig_db_scatter = px.scatter(
            visual_data, x="PC 1", y="PC 2", color="DBSCAN cluster",
            hover_data=FEATURES, title="Customer map · DBSCAN clusters and noise",
            color_discrete_sequence=["#4DE4F3", "#A58AFF", "#4CE0BD", "#FFBC6E", "#FF7FB2", "#62A8FF", "#9CB2C8"], opacity=0.84,
        )
        plot_layout(fig_db_scatter, 470)
        render_plot(fig_db_scatter)
    with db_right:
        st.markdown("**DBSCAN cluster profile**")
        st.dataframe(d_profile, use_container_width=True)
        if dbscan_score is not None:
            st.metric("DBSCAN silhouette (non-noise points)", f"{dbscan_score:.4f}")
        else:
            st.info("DBSCAN silhouette is unavailable until at least two non-noise clusters are found.")

with tab_compare:
    section_heading("04 / MODEL REVIEW", "Compare & export", "Review both algorithms side by side and download the assignments for your report.")
    compare_data: list[dict[str, object]] = [
        {"Algorithm": "K-Means", "Clusters": chosen_k, "Noise points": 0, "Silhouette": kmeans_score, "Evaluation population": "All customers"},
        {"Algorithm": "DBSCAN", "Clusters": dbscan_cluster_count, "Noise points": dbscan_noise_count, "Silhouette": dbscan_score, "Evaluation population": "Non-noise points only"},
    ]
    comparison = pd.DataFrame(compare_data)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig_comp_k = px.scatter(visual_data, x="PC 1", y="PC 2", color="K-Means cluster", title=f"K-Means · K={chosen_k}", color_discrete_sequence=["#4DE4F3", "#A58AFF", "#4CE0BD", "#FFBC6E", "#FF7FB2", "#62A8FF"], opacity=0.82, hover_data=FEATURES)
        plot_layout(fig_comp_k, 400)
        render_plot(fig_comp_k)
    with c2:
        fig_comp_d = px.scatter(visual_data, x="PC 1", y="PC 2", color="DBSCAN cluster", title=f"DBSCAN · eps={eps_value:.2f}", color_discrete_sequence=["#4DE4F3", "#A58AFF", "#4CE0BD", "#FFBC6E", "#FF7FB2", "#62A8FF", "#9CB2C8"], opacity=0.82, hover_data=FEATURES)
        plot_layout(fig_comp_d, 400)
        render_plot(fig_comp_d)

    st.markdown("#### Performance snapshot")
    comparison_display = comparison.copy()
    comparison_display["Silhouette"] = comparison_display["Silhouette"].map(lambda value: "N/A" if pd.isna(value) else f"{value:.4f}")
    st.dataframe(comparison_display, use_container_width=True, hide_index=True)
    st.caption("Scores should be interpreted carefully: DBSCAN's silhouette is calculated only for non-noise points, unlike K-Means which uses all customers.")

    st.markdown("#### Export your analysis")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button("⇩ Customer assignments CSV", make_download(results), "customer_clustering_results.csv", "text/csv", use_container_width=True)
    with dl2:
        st.download_button("⇩ Model comparison CSV", make_download(comparison), "algorithm_comparison.csv", "text/csv", use_container_width=True)
    with dl3:
        st.download_button("⇩ K-Means evaluation CSV", make_download(k_evaluation), "kmeans_evaluation.csv", "text/csv", use_container_width=True)

    with st.expander("How to explain this project in your presentation"):
        st.markdown(
            """
            - **Clustering** is an unsupervised learning task that groups records by similarity.
            - **K-Means** requires a chosen K. In this dashboard, you can compare K values from 2 to the selected upper limit.
            - **DBSCAN** discovers dense groups using Epsilon and Minimum Samples and can mark isolated customers as noise.
            - The UCI dataset's Channel and Region columns are not the number of allowed clusters. This project clusters six numerical spending features.
            - PCA is used only to draw a two-dimensional picture; model training uses all six scaled spending features.
            """
        )

st.markdown(
    '<div class="footer-line">CUSTOMER INTELLIGENCE · UCI WHOLESALE CUSTOMERS · K-MEANS × DBSCAN</div>',
    unsafe_allow_html=True,
)
