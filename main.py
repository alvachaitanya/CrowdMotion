import os
import numpy as np
import cv2 as cv
from models.MotionInfoImages import *


def main():
    # , '/Users/chaitanyareddy/Downloads/ViratDataset'
    dataset_paths = ['/Users/chaitanyareddy/Downloads/UCF_CrowdsDataset']
    for i in range(len(dataset_paths)):
        value = MotionInfo(dataset_paths[i])
        value.loadvideos()


if __name__ == "__main__":
    main()
