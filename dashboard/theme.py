"""
ORBIT PULSE - Mission Control Theme & Styling
==============================================
Dark aerospace analytics canvas:
Deep navy / charcoal background, slate panels, cyan / electric blue data accents.
Accessible contrast, consistent typography, and semantic status badges.
"""

# Palette definition
COLOR_BG = "#0B132B"
COLOR_PANEL = "#1C2541"
COLOR_PANEL_LIGHT = "#243356"
COLOR_ACCENT_CYAN = "#00E5FF"
COLOR_ACCENT_BLUE = "#38BDF8"
COLOR_TEXT_WHITE = "#F8FAFC"
COLOR_TEXT_MUTED = "#94A3B8"
COLOR_BORDER = "#334155"

# Semantic alert colors
COLOR_LOW = "#10B981"      # Green
COLOR_UNCERTAIN = "#F59E0B"  # Amber
COLOR_HIGH = "#EF4444"     # Red
COLOR_CUTOFF = "#F43F5E"    # Rose/Red line for cutoff

CUSTOM_CSS = f"""
<style>
:root {{ color-scheme: dark; }}
.stApp {{ background: radial-gradient(ellipse at 15% 0%, #17264a 0%, {COLOR_BG} 48%); color: {COLOR_TEXT_WHITE}; }}
[data-testid="stHeader"] {{ background: rgba(11,19,43,.72); }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #101b36 0%, #0a1021 100%); border-right: 1px solid {COLOR_BORDER}; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {{ letter-spacing: .08em; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 5px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ border: 1px solid transparent; border-radius: 9px; padding: 7px 10px; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ background: rgba(0,229,255,.10); border-color: rgba(0,229,255,.42); }}
.block-container {{ max-width: 1600px; padding-top: 1.4rem; padding-bottom: 2.5rem; }}
h1, h2, h3 {{ letter-spacing: -.025em; }}
.mission-masthead {{ padding: 20px 24px 18px; margin: 0 0 18px; border: 1px solid rgba(56,189,248,.25); border-radius: 16px; background: linear-gradient(115deg, rgba(28,37,65,.96), rgba(15,29,55,.88) 58%, rgba(0,229,255,.08)); box-shadow: 0 14px 34px rgba(0,0,0,.20); }}
.masthead-kicker {{ color: {COLOR_ACCENT_CYAN}; font-size: .70rem; font-weight: 800; letter-spacing: .20em; }}
.masthead-title {{ margin-top: 7px; color: {COLOR_TEXT_WHITE}; font-size: clamp(1.35rem, 2.5vw, 2.1rem); font-weight: 850; letter-spacing: .055em; }}
.masthead-title span {{ color: {COLOR_ACCENT_CYAN}; }}
.masthead-subtitle {{ margin-top: 5px; color: {COLOR_TEXT_MUTED}; font-size: .88rem; }}
.status-bar {{ background: linear-gradient(90deg, rgba(28,37,65,.98), rgba(20,35,62,.92)); border: 1px solid {COLOR_BORDER}; border-left: 3px solid {COLOR_ACCENT_CYAN}; padding: 12px 18px; border-radius: 10px; margin-bottom: 22px; display: flex; flex-wrap: wrap; gap: 10px 22px; justify-content: space-between; align-items: center; }}
.metric-card {{ min-height: 126px; background: linear-gradient(145deg, {COLOR_PANEL} 0%, {COLOR_PANEL_LIGHT} 100%); border: 1px solid rgba(148,163,184,.20); border-top: 2px solid rgba(0,229,255,.55); border-radius: 12px; padding: 16px 17px; margin-bottom: 12px; box-shadow: 0 8px 22px rgba(0,0,0,.14); }}
.metric-title {{ font-size: .70rem; text-transform: uppercase; letter-spacing: .12em; color: {COLOR_TEXT_MUTED}; margin-bottom: 9px; font-weight: 700; }}
.metric-value {{ font-size: clamp(1.45rem, 2vw, 2rem); line-height: 1.1; font-weight: 800; color: {COLOR_TEXT_WHITE}; }}
.metric-subtitle {{ font-size: .72rem; color: {COLOR_ACCENT_CYAN}; margin-top: 8px; }}
.badge-live, .badge-replay, .badge-manual, .badge-high, .badge-uncertain, .badge-low {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 750; font-size: .74rem; letter-spacing: .025em; }}
.badge-live {{ background: rgba(16,185,129,.13); color: #34D399; border: 1px solid rgba(16,185,129,.6); }}
.badge-replay {{ background: rgba(56,189,248,.13); color: #7DD3FC; border: 1px solid rgba(56,189,248,.6); }}
.badge-manual {{ background: rgba(245,158,11,.13); color: #FCD34D; border: 1px solid rgba(245,158,11,.6); }}
.badge-high {{ background: rgba(239,68,68,.15); color: #FCA5A5; border: 1px solid rgba(239,68,68,.65); }}
.badge-uncertain {{ background: rgba(245,158,11,.15); color: #FCD34D; border: 1px solid rgba(245,158,11,.65); }}
.badge-low {{ background: rgba(16,185,129,.15); color: #6EE7B7; border: 1px solid rgba(16,185,129,.65); }}
[data-testid="stMetric"] {{ background: rgba(28,37,65,.72); border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 12px 14px; }}
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {{ border: 1px solid rgba(51,65,85,.65); border-radius: 10px; overflow: hidden; }}
[data-testid="stTabs"] button {{ font-weight: 650; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 8px; border: 1px solid rgba(0,229,255,.42); font-weight: 700; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: {COLOR_ACCENT_CYAN}; color: {COLOR_ACCENT_CYAN}; }}
.disclaimer-box {{ background: rgba(30,41,59,.62); border: 1px solid {COLOR_BORDER}; border-left: 4px solid {COLOR_ACCENT_CYAN}; padding: 13px 17px; border-radius: 9px; font-size: .78rem; line-height: 1.6; color: {COLOR_TEXT_MUTED}; margin-top: 28px; }}
@media (max-width: 900px) {{ .block-container {{ padding-left: 1rem; padding-right: 1rem; }} .status-bar {{ align-items: flex-start; }} }}
</style>
"""

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(28, 37, 65, 0.7)",
    plot_bgcolor="rgba(11, 19, 43, 0.8)",
    font=dict(color=COLOR_TEXT_WHITE, family="sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(
        gridcolor="#243356",
        zerolinecolor="#243356",
        tickfont=dict(color=COLOR_TEXT_MUTED)
    ),
    yaxis=dict(
        gridcolor="#243356",
        zerolinecolor="#243356",
        tickfont=dict(color=COLOR_TEXT_MUTED)
    ),
    legend=dict(
        bgcolor="rgba(28, 37, 65, 0.8)",
        bordercolor="#334155",
        borderwidth=1
    )
)
