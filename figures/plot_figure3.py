import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
import numpy as np
import random
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerBase

# Path to the Excel file containing the results.
# Replace the empty string with the path to your results file.
loc = "path/to/results.xlsx"

vpr_techniques = {"HOG": "s", "CALC": "^", "AMOSNet": "P", "HybridNet": "X", "NetVLAD": "D"}

data_frame = pd.read_excel(loc, skiprows=[0, 0])

k = data_frame.iloc[1:, 0]

x = data_frame.iloc[0, 1:]

input_points = []

plt.xlabel('Single-Frame Matching Performance')
plt.ylabel('Sequence Matching Performance')
plt.title('Dataset')

for col in data_frame.iloc[:, 1:]:
    for index, y_coord in enumerate(data_frame[col][1:]):
        input_points.append(tuple((data_frame[col][0], y_coord)))
        plt.plot(data_frame[col][0], y_coord, marker=vpr_techniques[col], color=cm.hsv((index + 1) / 19)) #change here the number to correspond with the number of k values in the excel values (except k=1)

vpr_handles = []
for technique in vpr_techniques:
    vpr_handles.append(
        mlines.Line2D([], [], color='black', marker=vpr_techniques[technique], linestyle='None',
                      markersize=10, label=technique)
    )

plt.ylim(0,1.05)

k_legend = plt.legend(k.values, title="K =", bbox_to_anchor=(0.98, 0.88), borderaxespad=0.)
plt.legend(vpr_techniques, title="VPR Techniques", bbox_to_anchor=(0.83, 0.37), borderaxespad=0.,
           handles=vpr_handles)
plt.gca().add_artist(k_legend)

plt.xticks(x.values)
plt.show()