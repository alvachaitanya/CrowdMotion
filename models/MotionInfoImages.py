import os
import cv2
import numpy as np
import math
from utils.CornerDetection import *


class MotionInfo:
    def __init__(self, directory, refPt=[], output_counter=0, category=['Bottleneck', 'Lane', 'Arc', 'Block']):
        self.directory = directory
        self.refPt = refPt
        self.output_counter = output_counter
        self.category = category
        self.mask = []
        self.masks = []

    def loadvideos(self):
        names = []
        for cat in self.category:
            print(f"Crop the images for {cat} directory")
            for f in os.listdir(os.path.join(self.directory, 'Videos')):
                names.append(f)
                path = os.path.join(self.directory, 'Videos', f)
                self.generatemii(path, f, cat)
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

    def printimg(self, image, f, cat, folder='miidata'):
        outpath = os.path.join(self.directory, folder, cat, f + '-Img-' + str(self.output_counter) + '.png')
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(outpath, image)
        self.output_counter += 1

    @staticmethod
    def calculateDistance(x1, y1, x2, y2):
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    @staticmethod
    def calculateAngle(a, c, b, d):
        angle = np.degrees(math.atan2(b - d, a - c))
        return angle

    def getmaskimg(self, a, b, c, d, angle):
        colors = [(0, 0, 225), (0, 225, 0), (225, 0, 0), (0, 225, 225), (225, 0, 225), (225, 225, 0), (0, 100, 225),
                  (100, 0, 225), (0, 225, 100), (100, 225, 0), (225, 0, 100), (225, 100, 0)]
        for i in range(12):
            if 0 < angle <= 30:
                self.masks[i] = cv2.arrowedLine(self.masks[i], (c, d), (a, b), colors[0], 1, 4, 0, 0.1)
            elif 30 < angle <= 60:
                self.masks[i] = cv2.arrowedLine(self.masks[i], (c, d), (a, b), colors[1], 1, 4, 0, 0.1)
            elif 60 < angle <= 90:
                self.masks[i] = cv2.arrowedLine(self.masks[i], (c, d), (a, b), colors[2], 1, 4, 0, 0.1)
            elif 90 < angle <= 120:
                self.masks[i] = cv2.arrowedLine(self.masks[i], (c, d), (a, b), colors[3], 1, 4, 0, 0.1)
            elif 120 < angle <= 150:
                self.masks[i] = cv2.arrowedLine(self.masks[i], (c, d), (a, b), colors[4], 1, 4, 0, 0.1)
            elif 150 < angle <= 180:
                self.masks[i] = cv2.arrowedLine(self.masks[i], (c, d), (a, b), colors[5], 1, 4, 0, 0.1)
            elif 0 >= angle > -30:
                self.masks[i] = cv2.arrowedLine(self.masks[i], (c, d), (a, b), colors[6], 1, 4, 0, 0.1)
            elif -30 >= angle > -60:
                self.masks[i] = cv2.arrowedLine(self.masks[i], (c, d), (a, b), colors[7], 1, 4, 0, 0.1)
            elif -60 >= angle > -90:
                self.masks[i] = cv2.arrowedLine(self.masks[i], (c, d), (a, b), colors[8], 1, 4, 0, 0.1)
            elif -90 >= angle > -120:
                self.masks[i] = cv2.arrowedLine(self.masks[i], (c, d), (a, b), colors[9], 1, 4, 0, 0.1)
            elif -120 >= angle > -150:
                self.masks[i] = cv2.arrowedLine(self.masks[i], (c, d), (a, b), colors[10], 1, 4, 0, 0.1)
            elif -150 >= angle > -180:
                self.masks[i] = cv2.arrowedLine(self.masks[i], (c, d), (a, b), colors[11], 1, 4, 0, 0.1)

    def generatemii(self, path, f, cat, counter=0, counter2=0):
        newcornerpoints = np.zeros(5)
        cap = cv2.VideoCapture(path)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Take first frame and find corners in it
        ret, old_frame = cap.read()

        goodcorner = CornerDetection(old_frame)
        p0, old_gray = goodcorner.tomasiandfastCD()
        tmp_points = p0
        # Create a mask image for drawing purposes
        self.mask = np.zeros_like(old_frame)
        for i in range(12):
            self.masks.append(np.zeros_like(old_frame))
        while ret:
            ret, frame = cap.read()
            if not ret:
                break
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
                    angle = self.calculateAngle(a, c, b, d)
                    dist = self.calculateDistance(a, b, c, d)
                    self.getmaskimg(a, b, c, d, angle)
                    if dist > 3:
                        frame = cv2.circle(frame, (a, b), 1, (225, 225, 225), -1)
                for mask in self.masks:
                    self.mask = cv2.add(self.mask, mask)
                goodcorner = CornerDetection(frame)
                newcornerpoints, _ = goodcorner.tomasiandfastCD()
                img = cv2.add(frame, self.mask)
                cv2.namedWindow("Mask")
                cv2.setMouseCallback("Mask", self.click_and_crop, param=[self.mask, f, cat])
                cv2.imshow('frame', img)
                cv2.imshow("Mask", self.mask)
                tmp_points = good_new
                if counter2 == 10:
                    counter2 = 0
                    # tmp_points = good_new
                    self.mask = np.zeros_like(old_frame)
                    for i in range(12):
                        self.masks[i] = np.zeros_like(old_frame)
                    # self.printimg(self.mask, f)
                counter2 += 1
                cv2.waitKey(1000)
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
                print(p0.shape)
            if not p0.any():
                cv2.destroyAllWindows()
                cap.release()
                break
