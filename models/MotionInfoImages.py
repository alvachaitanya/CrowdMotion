import os
import cv2
import numpy as np
from utils.CornerDetection import *

class MotionInfo:
    def __init__(self, directory, refPt=[], output_counter=0, category=['Bottleneck', 'Lane', 'Arc', 'Block']):
        self.directory = directory
        self.refPt = refPt
        self.output_counter = output_counter
        self.category = category
        self.mask = []

    def loadvideos(self):
        names = []
        for cat in self.category:
            print(f"Crop the images for {cat} directory")
            for f in os.listdir(os.path.join(self.directory, 'Videos')):
                names.append(f)
                path = os.path.join(self.directory,'Videos', f)
                self.generatemii(path,f,cat)
        return True

    def click_and_crop(self, event, x, y, flags, param):
        cropping = False
        if event != 0 and event != 10 and event != 11 and event != 5:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.refPt = [(x, y)]
                cropping = True
            elif event == cv2.EVENT_LBUTTONUP:
                self.refPt.append((x, y))
                cropping = False
            if event == cv2.EVENT_RBUTTONDOWN:
                self.mask = np.zeros_like(param[0])
            if len(self.refPt) == 2 and self.refPt[0] != self.refPt[1]:
                image = param[0][self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
                print(f"image dimentions are : {image.shape}")
                self.printimg(image, param[1], param[2], folder='cropimg')
                self.refPt = []

    def printimg(self, image, f, cat, folder='miidata'):
        outpath = os.path.join(self.directory, folder, cat, f+'-Img-'+str(self.output_counter)+'.png')
        print(outpath)
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(outpath, image)
        self.output_counter += 1

    def generatemii(self, path, f, cat, counter=0):
        cap = cv2.VideoCapture(path)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))

        # Take first frame and find corners in it
        ret, old_frame = cap.read()

        goodcorner = CornerDetection(old_frame)
        p0, old_gray = goodcorner.tomasiandfastCD()

        # Create a mask image for drawing purposes
        self.mask = np.zeros_like(old_frame)

        while ret:
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            print(p0.shape, p1.shape, st.shape, err.shape)
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.mask = cv2.arrowedLine(self.mask,(c, d), (a, b), (0, 255, 225), 1, 4, 0, 0.5)
                frame = cv2.circle(frame, (a, b), 1, (225, 225, 225), -1)
            img = cv2.add(frame, self.mask)
            cv2.namedWindow(f)
            cv2.setMouseCallback(f, self.click_and_crop, param=[self.mask, f, cat])
            cv2.imshow('frame', img)
            cv2.imshow(f, self.mask)
            if counter == 25:
                counter = 0
                # self.mask = np.zeros_like(old_frame)
                # self.printimg(self.mask, f)
            cv2.waitKey(1)
            counter += 1
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            if not p0.any():
                cv2.destroyAllWindows()
                cap.release()
                break
