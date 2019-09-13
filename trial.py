from imutils import paths
import argparse
import requests
import cv2
import os
import urllib

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True,
	help="path to file containing image URLs")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory of images")
args = vars(ap.parse_args())
total = 0
if len(os.listdir(args["output"]))!=0:
        total=len(os.listdir(args["output"]))+1
print(total)
