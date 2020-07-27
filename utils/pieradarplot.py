import matplotlib.pyplot as plt
from math import pi
import numpy as np
import cv2
import io
from PIL import Image

class pieradarplot:
    # Set data
    def __init__(self):
        self.group = ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180', '180-210', '210-240', '240-270',
                      '270-300', '300-330', '330-360']
        self.colors = ['rosybrown','chocolate','orange','blue','pink','red','lime','indigo','teal','tomato','lawngreen','aqua']

    def fig2img(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    def avgValues(self, values):
        sum = np.sum(values)
        if sum != 0:
            for i in range(len(values)):
                values[i] = int((values[i]/sum)*100)
        return values

    def getpieradarplot(self, values):
        values = self.avgValues(values)
        categories = self.group
        N = len(categories)
        angles = np.linspace(0, 2 * pi, N, endpoint=False)
        angles_mids = angles + (angles[1] / 2)

        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
        ax.axis('off')
        # ax.set_theta_offset(0)
        # ax.set_xticks(angles_mids)
        # ax.set_xticklabels(categories)
        # ax.xaxis.set_minor_locator(FixedLocator(angles))

        # Draw ylabels
        # ax.set_rlabel_position(90)
        # ax.set_yticks([20, 40, 60, 80, 100])
        # ax.set_yticklabels(["20", "40", "60", "80", "100"], color="black", size=8)
        # ax.set_ylim(0, 100)

        for i in range(12):
            ax.bar(angles_mids[i], values[i], width=angles[1] - angles[0],
                   facecolor=self.colors[i], alpha=0.7, edgecolor='k', linewidth=1)

        ax.grid(False, axis='x', which='minor')
        ax.grid(False, axis='x', which='major')
        ax.grid(False, axis='y', which='major')
        ax.grid(False, axis='y', which='minor')
        img = self.fig2img(fig)
        plt.close()
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img,(224,224))
        return img