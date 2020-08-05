from itertools import islice, cycle

import cv2
import math
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
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
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(self.gray, mask=None, **self.good_feature_params)
        # fast = cv2.FastFeatureDetector_create()
        # fastpoints = fast.detect(self.gray, None)
        # fastarr = np.array([[[fpoint.pt[0], fpoint.pt[1]]] for fpoint in fastpoints], dtype=np.float32)
        # points = np.concatenate((points, fastarr), 0)
        tp = np.squeeze(points, axis=1)

        # dbscanclustering = DBSCAN(eps=10, min_samples=20).fit(tp)
        # finalpoints = np.expand_dims(dbscanclustering.components_, axis=1)

        # if tp.shape[0] < 5:
        #     opticsclustering = OPTICS(min_samples=tp.shape[0]).fit(tp)
        # else:
        opticsclustering = OPTICS().fit(tp)
        opticspoints = tp[opticsclustering.labels_ > -1]
        # colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
        #                                      '#f781bf', '#a65628', '#984ea3',
        #                                      '#999999', '#e41a1c', '#dede00']),
        #                               int(max(opticsclustering.labels_) + 1))))
        # plt.scatter(tp[:, 0], tp[:, 1], s=10, color=colors[opticsclustering.labels_])
        # plt.show()
        finalpoints = np.expand_dims(opticspoints, axis=1)
        return finalpoints, self.gray
