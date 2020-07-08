import os
import cv2


class MotionInfo:
    def __init__(self, directory):
        self.directory = directory

    def loadvideos(self):
        names = []
        for f in os.listdir(os.path.join(self.directory)):
            path = os.path.join(self.directory, f)
            cap = cv2.VideoCapture(path)
            result, img = cap.read()
            print(f)
            while result:
                result, img = cap.read()
                print(result)
                if not result:
                    break
                cv2.imshow("Video", img)
                cv2.waitKey(1)
        print(names)
        return True
