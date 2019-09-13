'''Before running this script please make sure that you have "requests" library
To check whether you have it installed you can run "pip list" command
This will give you a list of libraries installed, you can check for requests in the list'''

#workon cv
#pip install requests

# import the necessary packages
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

# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
rows = open(args["urls"]).read().strip().split("\n")
total = 0

''' Check whether the output folder is empty or not
If it is not empty then count the number of files and then
start the counter from there '''

if len(os.listdir(args["output"]))!=0:
        total=len(os.listdir(args["output"]))+1

def downloader(image_url,full_file_name):
    urllib.request.urlretrieve(image_url,full_file_name)

# loop the URLs
for url in rows:
	try:
		# try to download the image using the downloader function

		# save the image to disk
		p = os.path.sep.join([args["output"], "{}.jpg".format(
			str(total).zfill(8))])
		downloader(url,p)

		# update the counter
		print("[INFO] downloaded: {}".format(p))
		total += 1

	# handle if any exceptions are thrown during the download process
	except:
		print("[INFO] error downloading {}...skipping".format(p))

# loop over the image paths we just downloaded
for imagePath in paths.list_images(args["output"]):
	# initialize if the image should be deleted or not
	delete = False

	# try to load the image
	try:
		image = cv2.imread(imagePath)

		# if the image is `None` then we could not properly load it
		# from disk, so delete it
		if image is None:
			delete = True

	# if OpenCV cannot load the image then the image is likely
	# corrupt so we should delete it
	except:
		print("Except")
		delete = True

	# check to see if the image should be deleted
	if delete:
		print("[INFO] deleting {}".format(imagePath))
		os.remove(imagePath)

#code to exceute this script

#$ python downloading_images.py --urls urls.txt --output images\damaged
