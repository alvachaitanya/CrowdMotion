from itertools import islice, cycle

import cv2
import math
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
import time
import matplotlib.pyplot as plt


class CornerDetection:

    def __init__(self, frame):
        self.good_feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=25, blockSize=7)
        self.frame = frame
        self.gray = None

    def calculateDistance(self, x1, y1, x2, y2):
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    def calculateAngle(self, a, c, b, d):
        angle = np.degrees(math.atan2(b - d, a - c))
        return angle

    def tomasiandfastCD(self):
        # tempframe1 = self.frame
        tempframe2 = self.frame
        # tempframe3 = self.frame
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(self.gray, mask=None, **self.good_feature_params)
        # print(points.shape)
        # corners = np.int0(points)

        # for i in corners:
        #     x, y = i.ravel()
        #     cv2.circle(tempframe1, (x, y), 3, 255, -1)
        #
        # cv2.imshow('Shi tomasi', tempframe1)
        # if cv2.waitKey(1) & 0xff == 27:
        #     cv2.destroyAllWindows()


        # dst = cv2.cornerHarris(self.gray,2,3,0.04)
        # # # result is dilated for marking the corners, not important
        # dst = cv2.dilate(dst, None)
        #
        # # Threshold for an optimal value, it may vary depending on the image.
        # tempframe2[dst > 0.01 * dst.max()] = [0, 0, 255]
        # print(tempframe2[dst > 0.1 * dst.max()].shape)
        # cv2.imshow('Harris', tempframe2)
        # if cv2.waitKey(1) & 0xff == 27:
        #     cv2.destroyAllWindows()


        # fast = cv2.FastFeatureDetector_create()
        # fastpoints = fast.detect(self.gray, None)
        # fastarr = np.array([[[fpoint.pt[0], fpoint.pt[1]]] for fpoint in fastpoints], dtype=np.float32)
        # points = np.concatenate((points, fastarr), 0)
        # print(points.shape)
        # img2 = cv2.drawKeypoints(tempframe3, fastpoints,np.array([]), color=(0, 225, 0))
        # cv2.imshow('Fast', img2)
        # if cv2.waitKey(1) & 0xff == 27:
        #     cv2.destroyAllWindows()

        tp = np.squeeze(points, axis=1)

        # now = time.time()
        dbscanclustering = DBSCAN(eps=30, min_samples=2).fit(tp)
        finalpoints = np.expand_dims(dbscanclustering.components_, axis=1)
        print(finalpoints.shape)
        # later = time.time()
        # dbscandiff = later - now
        # print(f"Computation Time of DBSCAN : {dbscandiff}")

        #now = time.time()
        # opticsclustering = OPTICS().fit(tp)
        # opticspoints = tp[opticsclustering.labels_ > -1]
        # finalpoints = np.expand_dims(opticspoints, axis=1)
        #
        # later = time.time()
        # opticsdiff = later - now
        # print(f"Computation Time of OPTICS : {opticsdiff}")
        return finalpoints, self.gray
