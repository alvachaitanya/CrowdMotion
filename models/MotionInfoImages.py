import os
import cv2
import numpy as np
import math
import pandas as pd
from utils.CornerDetection import *
from utils.pieradarplot import *


class MotionInfo:
    def __init__(self, directory, refPt=[], output_counter=0, category=['Bottleneck']):#'Bottleneck', 'Lane', 'Arc', 'Block'
        self.directory = directory
        self.refPt = refPt
        self.output_counter = output_counter
        self.category = category
        self.mask = []
        self.masks = []
        self.anglecount = []
        self.blocksize = 28
        self.directionarr = np.zeros((int(224/self.blocksize), int(224/self.blocksize), 12))
        self.magnitudearr = np.zeros((int(224 / self.blocksize), int(224 / self.blocksize), 12))
        #self.df = pd.DataFrame(columns=np.arange((int(224/self.blocksize) * int(224/self.blocksize) * 12)+1))
        self.df = pd.DataFrame(columns=np.arange((int(224/self.blocksize) * int(224/self.blocksize) * 2)+1))

    def loadvideos(self):
        for cat in self.category:
            print(f"Crop the images for {cat} directory")
            for f in os.listdir(os.path.join(self.directory, 'Videos')):
                if not f.startswith('.') and f.find('.csv') < 0 :
                    path = os.path.join(self.directory, 'Videos', f)
                    self.generatemii(path, f, cat)
                    filename = os.path.join(self.directory, 'Videos', f+'.csv')
                    # self.df.to_csv(filename, index=False)
                    #self.df = pd.DataFrame(columns=np.arange((int(224 / self.blocksize) * int(224 / self.blocksize) * 12) + 1))
                    self.df = pd.DataFrame(columns=np.arange((int(224 / self.blocksize) * int(224 / self.blocksize) * 2) + 1))
        return True

    def click_and_crop(self, event, x, y, flags, param):
        cropping = False
        if event != 0 and event != 10 and event != 11 and event != 5:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.refPt = [(x, y)]
            elif event == cv2.EVENT_LBUTTONUP:
                self.refPt.append((x, y))
            if event == cv2.EVENT_RBUTTONDOWN:
                self.mask = np.zeros_like(param[0])
            if len(self.refPt) == 2 and self.refPt[0] != self.refPt[1]:
                image = param[0][self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
                print(f"image dimentions are : {image.shape}")
                self.printimg(image, param[1], param[2], folder='cropimg')
                self.refPt = []

    def printimg(self, image, f, cat="", folder='miidata'):
        outpath = os.path.join(self.directory, folder, f + '-Img-' + str(self.output_counter) + '.png')
        cv2.imwrite(outpath, image)
        self.output_counter += 1

    def getmaskimg(self, a, b, c, d, angle, dist):
        colors = [(0, 0, 225), (0, 225, 0), (225, 0, 0), (0, 225, 225), (225, 0, 225), (225, 225, 0), (0, 100, 225),
                  (100, 0, 225), (0, 225, 100), (100, 225, 0), (225, 0, 100), (225, 100, 0)]

        xblock = int(a / self.blocksize)-1
        yblock = int(b / self.blocksize)-1

        if 0 < angle <= 30:
            self.masks[0] = cv2.arrowedLine(self.masks[0], (c, d), (a, b), colors[0], 1, 4, 0, 0.1)
            self.anglecount[0] += 1
            self.directionarr[xblock][yblock][0] += 1
            self.magnitudearr[xblock][yblock][0] = (self.magnitudearr[xblock][yblock][0] + dist)/2
        elif 30 < angle <= 60:
            self.masks[1] = cv2.arrowedLine(self.masks[1], (c, d), (a, b), colors[1], 1, 4, 0, 0.1)
            self.anglecount[1] += 1
            self.directionarr[xblock][yblock][1] += 1
            self.magnitudearr[xblock][yblock][1] = (self.magnitudearr[xblock][yblock][1] + dist)/2
        elif 60 < angle <= 90:
            self.masks[2] = cv2.arrowedLine(self.masks[2], (c, d), (a, b), colors[2], 1, 4, 0, 0.1)
            self.anglecount[2] += 1
            self.directionarr[xblock][yblock][2] += 1
            self.magnitudearr[xblock][yblock][2] = (self.magnitudearr[xblock][yblock][2] + dist)/2
        elif 90 < angle <= 120:
            self.masks[3] = cv2.arrowedLine(self.masks[3], (c, d), (a, b), colors[3], 1, 4, 0, 0.1)
            self.anglecount[3] += 1
            self.directionarr[xblock][yblock][3] += 1
            self.magnitudearr[xblock][yblock][3] = (self.magnitudearr[xblock][yblock][3] + dist)/2
        elif 120 < angle <= 150:
            self.masks[4] = cv2.arrowedLine(self.masks[4], (c, d), (a, b), colors[4], 1, 4, 0, 0.1)
            self.anglecount[4] += 1
            self.directionarr[xblock][yblock][4] += 1
            self.magnitudearr[xblock][yblock][4] = (self.magnitudearr[xblock][yblock][4] + dist)/2
        elif 150 < angle <= 180:
            self.masks[5] = cv2.arrowedLine(self.masks[5], (c, d), (a, b), colors[5], 1, 4, 0, 0.1)
            self.anglecount[5] += 1
            self.directionarr[xblock][yblock][5] += 1
            self.magnitudearr[xblock][yblock][5] = (self.magnitudearr[xblock][yblock][5] + dist)/2
        elif 0 >= angle > -30:
            self.masks[6] = cv2.arrowedLine(self.masks[6], (c, d), (a, b), colors[6], 1, 4, 0, 0.1)
            self.anglecount[11] += 1
            self.directionarr[xblock][yblock][6] += 1
            self.magnitudearr[xblock][yblock][6] = (self.magnitudearr[xblock][yblock][6] + dist)/2
        elif -30 >= angle > -60:
            self.masks[7] = cv2.arrowedLine(self.masks[7], (c, d), (a, b), colors[7], 1, 4, 0, 0.1)
            self.anglecount[10] += 1
            self.directionarr[xblock][yblock][7] += 1
            self.magnitudearr[xblock][yblock][7] = (self.magnitudearr[xblock][yblock][7] + dist)/2
        elif -60 >= angle > -90:
            self.masks[8] = cv2.arrowedLine(self.masks[8], (c, d), (a, b), colors[8], 1, 4, 0, 0.1)
            self.anglecount[9] += 1
            self.directionarr[xblock][yblock][8] += 1
            self.magnitudearr[xblock][yblock][8] = (self.magnitudearr[xblock][yblock][8] + dist)/2
        elif -90 >= angle > -120:
            self.masks[9] = cv2.arrowedLine(self.masks[9], (c, d), (a, b), colors[9], 1, 4, 0, 0.1)
            self.anglecount[8] += 1
            self.directionarr[xblock][yblock][9] += 1
            self.magnitudearr[xblock][yblock][9] = (self.magnitudearr[xblock][yblock][9] + dist)/2
        elif -120 >= angle > -150:
            self.masks[10] = cv2.arrowedLine(self.masks[10], (c, d), (a, b), colors[10], 1, 4, 0, 0.1)
            self.anglecount[7] += 1
            self.directionarr[xblock][yblock][10] += 1
            self.magnitudearr[xblock][yblock][10] = (self.magnitudearr[xblock][yblock][10] + dist)/2
        elif -150 >= angle > -180:
            self.masks[11] = cv2.arrowedLine(self.masks[11], (c, d), (a, b), colors[11], 1, 4, 0, 0.1)
            self.anglecount[6] += 1
            self.directionarr[xblock][yblock][11] += 1
            self.magnitudearr[xblock][yblock][11] = (self.magnitudearr[xblock][yblock][11] + dist)/2

    def generatemii(self, path, f, cat, counter=0, counter2=0):
        newcornerpoints = np.zeros(5)
        cap = cv2.VideoCapture(path)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_frame = cv2.resize(old_frame, (224, 224))
        goodcorner = CornerDetection(old_frame)
        p0, old_gray = goodcorner.tomasiandfastCD()
        tmp_points = p0
        # Create a mask image for drawing purposes
        self.mask = np.zeros_like(old_frame)
        self.masks = []
        self.anglecount = []
        self.directionarr = np.zeros((int(224/self.blocksize), int(224/self.blocksize), 12))
        self.magnitudearr = np.zeros((int(224 / self.blocksize), int(224 / self.blocksize), 12))
        for i in range(12):
            self.masks.append(np.zeros_like(old_frame))
            self.anglecount.append(0)
        while ret:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            # good_old = p0[st == 1]
            tmp_points = tmp_points[st == 1]
            if counter == 5:
                counter = 0
                for i, (new, old) in enumerate(zip(good_new, tmp_points)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    dist = goodcorner.calculateDistance(a, b, c, d)
                    angle = goodcorner.calculateAngle(a, c, b, d)
                    self.getmaskimg(a, b, c, d, angle, dist)
                    if dist > 3:
                        frame = cv2.circle(frame, (a, b), 1, (225, 225, 225), -1)
                plot = pieradarplot()
                datarow = plot.plotblockdirection(self.directionarr, self.magnitudearr)
                df_length = len(self.df)
                self.df.loc[df_length] = datarow
                # plt = plot.getpieradarplot(self.anglecount)
                for mask in self.masks:
                    self.mask = cv2.add(self.mask, mask)
                goodcorner = CornerDetection(frame)
                newcornerpoints, _ = goodcorner.tomasiandfastCD()
                img = cv2.add(frame, self.mask)
                cv2.namedWindow("Mask")
                cv2.setMouseCallback("Mask", self.click_and_crop, param=[self.mask, f, cat])
                cv2.imshow('frame', img)
                cv2.imshow("Mask", self.mask)
                # cv2.imshow("plt", plt)
                # for i in range(len(self.masks)):
                #     mask = cv2.resize(self.masks[i], (224, 224))
                #     cv2.imshow(f"mask{i}", mask)
                tmp_points = good_new
                mskgray = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
                val, count = np.unique(mskgray, return_counts=True)
                num1 = np.sum(count[val == 0])
                num2 = np.sum(count[val != 0])
                if num1 * 0.2 < num2:
                    self.mask = np.zeros_like(old_frame)
                    self.directionarr = np.zeros((int(224 / self.blocksize), int(224 / self.blocksize), 12))
                    self.magnitudearr = np.zeros((int(224 / self.blocksize), int(224 / self.blocksize), 12))
                    for i in range(12):
                        self.masks[i] = np.zeros_like(old_frame)
                        self.anglecount[i] = 0
                # self.printimg(self.mask, f)
                cv2.waitKey(1)
            counter += 1
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            tmp_points = tmp_points.reshape(-1, 1, 2)
            if newcornerpoints.shape != (5,):
                p0 = np.concatenate((p0, newcornerpoints), axis=0)
                tmp_points = np.concatenate((tmp_points, newcornerpoints), axis=0)
                # p0 =  newcornerpoints
                # tmp_points = newcornerpoints
                newcornerpoints = np.zeros(5)
            if not p0.any():
                cv2.destroyAllWindows()
                cap.release()
                break
