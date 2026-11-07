"""Shared palette and final-size typography for public benchmark figures."""

INK = "#0A0E14"
MUTED = "#5B636B"
GRID = "#E3E5EA"
CREDITED = "#0072B2"
"""A data-reading analysis or policy: the honest report, the reference policy."""
DISTORTED = "#B54823"
"""A distorted report or a failing cell."""
ALTERNATIVE = "#008269"
"""A further report or policy drawn beside the first two."""
CAUTION = "#AD7800"
"""Thresholds and data-blind policies."""
PASSING = "#F5F6F8"
"""A cell or outcome with no failure."""
WIDTH = 7.2
"""Figure width in inches, shared by every demonstration."""
NOTE_SIZE = 7.5
"""Type size of annotations and notes: the smallest text in any figure."""

DEMONSTRATION_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "svg.hashsalt": "trialagentbench-demonstrations",
    "savefig.dpi": 200,
    "text.color": INK,
}
