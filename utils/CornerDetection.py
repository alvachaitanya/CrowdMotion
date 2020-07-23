import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS


class CornerDetection():

    def __init__(self, frame):
        self.good_feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=25, blockSize=7)
        self.frame = frame
        self.gray = None

    def tomasiandfastCD(self):
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(self.gray, mask=None, **self.good_feature_params)
        # fast = cv2.FastFeatureDetector_create()
        # fastpoints = fast.detect(self.gray, None)
        # fastarr = np.array([[[fpoint.pt[0], fpoint.pt[1]]] for fpoint in fastpoints], dtype=np.float32)
        # points = np.concatenate((points, fastarr), 0)
        # tp = np.squeeze(points, axis=1)

        # dbscanclustering = DBSCAN(eps=10, min_samples=20).fit(tp)
        # finalpoints = np.expand_dims(dbscanclustering.components_, axis=1)
        #
        #
        # opticsclustering = OPTICS(min_samples=10).fit(tp)
        # opticspoints = tp[opticsclustering.labels_ != -1]
        # finalpoints = np.expand_dims(opticspoints, axis=1)
        # print(finalpoints.shape)
        return points, self.gray
