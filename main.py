import os
import numpy as np
import cv2 as cv
from models.MotionInfoImages import *


def main():
    value = MotionInfo('/Users/chaitanyareddy/Downloads/UCF_CrowdsDataset')
    print(value.loadvideos2())


if __name__ == "__main__":
    main()
