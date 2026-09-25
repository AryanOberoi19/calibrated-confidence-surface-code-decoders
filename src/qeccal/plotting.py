"""Figure style shared with the report: palette validated for the light surface (dataviz validator), thin marks,
recessive grid. Colour carries identity only with a legend or direct label next to it."""
import matplotlib as mpl
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical order (light, white surface)
MK = ["o", "s", "^", "D"]
def apply():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 1.6, "lines.markersize": 5, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK2,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    })
