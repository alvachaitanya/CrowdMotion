import matplotlib.pyplot as plt
from math import pi
import math
import numpy as np
import cv2
import io
from PIL import Image
import pandas as pd

class pieradarplot:
    # Set data
    def __init__(self):
        self.group = ['0-30', '30-60', '60-90', '90-120', '120-150', '150-180', '180-210', '210-240', '240-270',
                      '270-300', '300-330', '330-360']
        self.colors = ['rosybrown', 'chocolate', 'orange', 'blue', 'pink', 'red', 'lime', 'indigo', 'teal', 'tomato', 'lawngreen', 'aqua']

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

    def getanglefromindex(self, index):
        if index == 0:
            return -15, self.colors[index]
        elif index == 1:
            return -45, self.colors[index]
        elif index == 2:
            return -75, self.colors[index]
        elif index == 3:
            return -105, self.colors[index]
        elif index == 4:
            return -135, self.colors[index]
        elif index == 5:
            return -165, self.colors[index]
        elif index == 6:
            return 15, self.colors[index]
        elif index == 7:
            return 45, self.colors[index]
        elif index == 8:
            return 75, self.colors[index]
        elif index == 9:
            return 105, self.colors[index]
        elif index == 10:
            return 135, self.colors[index]
        elif index == 11:
            return 165, self.colors[index]

    # def plotblockdirection(self, directionarr, magnitudearr):
    #     x = range(10)
    #     y = range(10)
    #     dirmagarr = np.multiply(directionarr,magnitudearr)
    #     max = np.amax(dirmagarr)
    #     rng = max * 0.25
    #     blocksize = dirmagarr.shape[0]
    #     # maxVal = np.amax(directionarr)
    #     # directionarr = directionarr/maxVal
    #     fig, ax = plt.subplots(nrows=blocksize, ncols=blocksize)
    #
    #     for i in range(blocksize):
    #         for j in range(blocksize):
    #             ax[i, j].axis('off')
    #             directions = dirmagarr[i][j]
    #             flat = directions.flatten()
    #             flat.sort()
    #             length = flat[-1]
    #             length2 = flat[-2]
    #             if length > rng and length2 > rng:
    #                 index = np.where(directions == length)
    #                 index2 = np.where(directions == length2)
    #                 angle, clr = self.getanglefromindex(index[0][0])
    #                 angle2, clr2 = self.getanglefromindex(index2[0][0])
    #                 endy = math.sin(math.radians(angle)) * 0.75
    #                 endx = math.cos(math.radians(angle)) * 0.75
    #                 endy2 = math.sin(math.radians(angle2)) * 0.75
    #                 endx2 = math.cos(math.radians(angle2)) * 0.75
    #                 ax[i, j].set_ylim(ymin=-1, ymax=1)
    #                 ax[i, j].set_xlim(xmin=-1, xmax=1)
    #                 ax[i, j].arrow(0, 0, endx, endy, fc=clr, ec=clr, head_width=0.3, head_length=0.2)
    #                 ax[i, j].arrow(0, 0, endx2, endy2, fc=clr2, ec=clr2, head_width=0.3, head_length=0.2)
    #             else:
    #                 index = np.where(directions == length)
    #                 angle, clr = self.getanglefromindex(index[0][0])
    #                 endy = math.sin(math.radians(angle)) * 0.75
    #                 endx = math.cos(math.radians(angle)) * 0.75
    #                 ax[i, j].set_ylim(ymin=-1, ymax=1)
    #                 ax[i, j].set_xlim(xmin=-1, xmax=1)
    #                 ax[i, j].arrow(0, 0, endx, endy, fc=clr, ec=clr, head_width=0.3, head_length=0.2)
    #
    #     img = self.fig2img(fig)
    #     plt.close()
    #     img = np.asarray(img)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    #     cv2.imshow("plot", img)
    #     cv2.waitKey(1)

    def plotblockdirection(self, directionarr, magnitudearr):
        x = range(10)
        y = range(10)
        dirmagarr = np.multiply(directionarr,magnitudearr)
        blocksize = dirmagarr.shape[0]
        pdrow = np.zeros_like(dirmagarr)
        max = np.amax(dirmagarr)
        rng = max * 0.05
        fig, ax = plt.subplots(nrows=blocksize, ncols=blocksize)
        counter = 0
        for i in range(blocksize):
            for j in range(blocksize):
                ax[j, i].axis('off')
                directions = dirmagarr[i][j]
                flat = directions.flatten()
                flat.sort()
                length = flat[-1]
                length2 = flat[-2]
                if length > rng and length2 > rng:
                    index = np.where(directions == length)
                    index2 = np.where(directions == length2)
                    pdrow[j][i][index] += 1
                    pdrow[j][i][index2] += 1
                    angle, clr = self.getanglefromindex(index[0][0])
                    angle2, clr2 = self.getanglefromindex(index2[0][0])
                    endy = math.sin(math.radians(angle)) * 0.75
                    endx = math.cos(math.radians(angle)) * 0.75
                    endy2 = math.sin(math.radians(angle2)) * 0.75
                    endx2 = math.cos(math.radians(angle2)) * 0.75
                    ax[j, i].set_ylim(ymin=-1, ymax=1)
                    ax[j, i].set_xlim(xmin=-1, xmax=1)
                    ax[j, i].arrow(0, 0, endx, endy, fc=clr, ec=clr, head_width=0.3, head_length=0.2)
                    ax[j, i].arrow(0, 0, endx2, endy2, fc=clr2, ec=clr2, head_width=0.3, head_length=0.2)
                elif length > rng:
                    index = np.where(directions == length)
                    pdrow[j][i][index] += 1
                    angle, clr = self.getanglefromindex(index[0][0])
                    endy = math.sin(math.radians(angle)) * 0.75
                    endx = math.cos(math.radians(angle)) * 0.75
                    ax[j, i].set_ylim(ymin=-1, ymax=1)
                    ax[j, i].set_xlim(xmin=-1, xmax=1)
                    ax[j, i].arrow(0, 0, endx, endy, fc=clr, ec=clr, head_width=0.3, head_length=0.2)
        img = self.fig2img(fig)
        plt.close()
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imshow("plot", img)
        cv2.waitKey(1)
        dirs = np.split(pdrow.flatten(), (blocksize ** 2))
        endarr =[]
        for dir in dirs:
            index = np.where(dir == 1)
            if len(index[0]) == 0:
                endarr.append(0)
                endarr.append(0)
            elif len(index[0]) == 1:
                endarr.append(index[0][0])
                endarr.append(0)
            elif len(index[0]) == 2:
                endarr.append(index[0][0])
                endarr.append(index[0][1])
        # print("Enter the input : ")
        # value = input()
        endarr.append(3)
        return endarr

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
        ax.set_theta_direction(-1)
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

# x = pieradarplot()
# x.plotblockdirection([9, 25, 19, 8, 8, 14, 20, 60, 34, 17, 8, 2])