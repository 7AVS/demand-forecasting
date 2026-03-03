"""Shared styles — Desaturated Cool palette."""

COLORS = {
    "slate":    "#374E55",
    "dark":     "#2B3A3F",
    "muted":    "#80796B",
    "teal":     "#6B9DAB",
    "rose":     "#997B7A",
    "sage":     "#7A9E8E",
    "taupe":    "#A08E78",
    "lavender": "#8585A0",
    "stone":    "#9C9488",
}

CHART_COLORS = [
    "#374E55", "#6B9DAB", "#A08E78", "#997B7A",
    "#7A9E8E", "#8585A0", "#9C9488",
]

DIVERGING_SCALE = ["#7A9E8E", "#F5F5F0", "#997B7A"]

SURFACES = {
    "background":     "#FFFFFF",
    "background_alt": "#F7F7F5",
    "card_border":    "#E8E6E1",
    "grid_line":      "#EDEBE8",
    "header_bg":      "#2B3A3F",
}

PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#80796B"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=40),
    hoverlabel=dict(
        bgcolor="#2B3A3F", font_size=12,
        font_family="Inter, system-ui, sans-serif", font_color="#FFFFFF",
    ),
)

AXIS_DEFAULTS = dict(
    showline=False,
    gridcolor="#EDEBE8",
    tickfont=dict(size=12, color="#80796B"),
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, sans-serif; }
    .main-header { background: #2B3A3F; padding: 1.5rem 2rem; border-radius: 8px; margin-bottom: 1.5rem; color: white; }
    .main-header h1 { margin: 0; font-size: 1.5rem; font-weight: 700; color: white; }
    .main-header p { margin: 0.25rem 0 0 0; font-size: 0.9rem; opacity: 0.75; color: white; }
    .kpi-card { background: white; border: 1px solid #E8E6E1; border-top: 3px solid #374E55; border-radius: 8px; padding: 16px; text-align: center; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #2B3A3F; line-height: 1.1; font-variant-numeric: tabular-nums; }
    .kpi-label { font-size: 0.8rem; color: #80796B; margin-top: 4px; }
    .section-header { color: #2B3A3F; font-size: 1.2rem; font-weight: 700; margin: 24px 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #E8E6E1; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
