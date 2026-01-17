import matplotlib.pyplot as plt

from cycler import cycler

THEME_RED = [228/255, 21/255, 75/255, 1]      # #e4154b (Heading fill)
THEME_CREAM = [238/255, 236/255, 225/255, 1]   # #eeece1 (Body fill)
THEME_DARK = [0.1, 0.1, 0.1, 1]                # For edges/text

plt_style = {
    'figure.facecolor': '#eeece1',
    'axes.facecolor': '#eeece1',

    'axes.prop_cycle': cycler(color=['#e4154b', '#1a1a1a', '#575757', '#9e9e9e']),

'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'text.color': '#000000',
    'axes.labelcolor': '#000000',
    'xtick.color': '#000000',
    'ytick.color': '#000000',

    # Title and Label Sizes
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'legend.fontsize': 10,

    # Grid and Spines
    'axes.grid': True,
    'grid.color': '#d1d0c5',  # Slightly darker than background for subtlety
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,

    # Export quality
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
}