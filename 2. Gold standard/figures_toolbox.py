import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import StrMethodFormatter
from pprint import pprint

# v2

##################
# Figures design #
##################

# tick_labelsize = 8

# sns.set(
#     context='paper',
#     style='ticks',
#     rc={
#         'xtick.direction':'in',
#         'ytick.direction':'in',
#         'axes.grid':True,
#         'font.size':6,
#         'legend.fontsize':6,
#         'xtick.labelsize':tick_labelsize,
#         'ytick.labelsize':tick_labelsize,
#         'axes.labelsize':8,
#         'axes.axisbelow':True,
#     }
# )

performancePlot_style = {
    'xtick.direction':'in',
    'xtick.labelsize':11,
    'ytick.direction':'in',
    'ytick.labelsize':11,
    'font.size':10,
    'legend.loc':'upper right',
#     'legend.frameon':False,
    'legend.fontsize':13,
    'axes.grid':True,
    'axes.titlesize':11,
    'axes.labelsize':16,
    'axes.labelweight':'semibold',
    'axes.axisbelow':True,
    'axes.spines.top':False,
    'axes.spines.right':False,
    'axes.spines.right':False,
    'axes.grid.which':'both',
    'figure.titlesize':20,
    'figure.titleweight':'bold',
    'figure.figsize':(8.,4.),
    'lines.linewidth':1.3
}

# format_export = 'svg'
dpi_export = 300

path_output_fig = '../Figures/'

###########
# Colours #
###########

# PPI paper
# from https://github.com/BlakeRMills/MetBrewer
palette_hiroshige = ["#e76254", "#ef8a47", "#f7aa58", "#ffd06f", "#ffe6b7", "#aadce0", "#72bcd5", "#528fad", "#376795", "#1e466e"]
palette_redon = ["#5b859e", "#1e395f", "#75884b", "#1e5a46", "#df8d71", "#af4f2f", "#d48f90", "#732f30", "#ab84a5", "#59385c", "#d8b847", "#b38711"]

my_palette1 = ['#edf8b1','#7fcdbb','#2c7fb8']
palette_tiepolo = ["#802417", "#c06636", "#ce9344", "#e8b960", "#646e3b", "#2b5851", "#508ea2", "#17486f"]
palette_renoir = ["#17154f", "#2f357c", "#6c5d9e", "#9d9cd5", "#b0799a", "#f6b3b0", "#e48171", "#bf3729", "#e69b00", "#f5bb50", "#ada43b", "#355828"]


#############
# Functions #
#############

def convert_to_float(x):
    return float(x.replace(',',''))

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def export_figure(fig, name_fig, addSupp=False):
    L = ['svg','pdf']
    if addSupp:
        L += ['eps']

    for format_export in L:
        fig.savefig(
            os.path.join(path_output_fig, '{}.{}'.format(name_fig, format_export)),
            format=format_export,
            dpi=dpi_export,
            transparent=True,
        )

