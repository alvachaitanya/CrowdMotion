import os
import cv2
import numpy as np


class MotionInfo:
    def __init__(self, directory, refPt=[], output_counter=0):
        self.directory = directory
        self.refPt = refPt
        self.output_counter = output_counter

    def loadvideos(self):
        names = []
        for f in os.listdir(os.path.join(self.directory, 'Videos')):
            print(f)
            names.append(f)
            path = os.path.join(self.directory,'Videos', f)
            self.generatemii(path,f)
        print(names)
        return True

    def click_and_crop(self, event, x, y, flags, param):
        cropping = False
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
            cropping = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.refPt.append((x, y))
            cropping = False
        if len(self.refPt) == 2:
            image = param[0][self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
            self.printimg(image, param[1])
            self.refPt = []

    def printimg(self, image, f):
        outpath = os.path.join(self.directory, 'miidata', f+'Img-'+str(self.output_counter)+'.png')
        print(outpath)
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(outpath, image)
        self.output_counter += 1

    def generatemii(self, path, f, counter=0):
        print(path)
        cap = cv2.VideoCapture(path)

        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=500,
                              qualityLevel=0.01,
                              minDistance=25,
                              blockSize=7)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))

        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        while ret:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            # print(p1.shape, st, err)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # print(f"Points 1 are {p1[st != 1]}")
            # print(f"Points 0 are {p0[st != 1]}")

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.arrowedLine(mask,(c, d), (a, b), (0, 255, 225), 1, 4, 0, 0.5)
                frame = cv2.circle(frame, (a, b), 5, (225, 225, 225), -1)
            img = cv2.add(frame, mask)
            cv2.namedWindow("Motion Image")
            cv2.setMouseCallback("Motion Image", self.click_and_crop, param=[mask, f])

            cv2.imshow('frame', img)
            cv2.imshow("Motion Image", mask)
            if counter == 25:
                counter = 0
                self.printimg(mask, f)
            cv2.waitKey(1)
            counter+=1
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            if not p0.any():
                cv2.destroyAllWindows()
                cap.release()
                break
