import sys
from skimage.feature import local_binary_pattern
import numpy as np
sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2

import math

# Get user supplied values
imagePath = sys.argv[1]
cascPath = 'haarcascade_frontalface_default.xml'

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print "Found {0} faces!".format(len(faces))
faceImages = [];
# Draw a rectangle around the faces
size=256.0;
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2);
    scale=size/w;
    res = cv2.resize(gray[y:y+h,x:x+h],None,fx=scale,fy=scale, interpolation = cv2.INTER_CUBIC)
    faceImages.append(res);

cv2.imshow("Faces found", image);
cv2.waitKey(0);
radius = 3
no_points = 8 * radius
featureImage=[];
for face in faceImages:
	hist,bins = np.histogram(face.flatten(),256,[0,256]) 
	cdf = hist.cumsum()
	cdf_normalized = cdf * hist.max()/ cdf.max()
	cdf_m = np.ma.masked_equal(cdf,0)
	cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf = np.ma.filled(cdf_m,0).astype('uint8')
	cv2.imshow("faceHist", cdf[face]);
	# cv2.waitKey(0);
	cv2.imshow("face", face);
	cv2.waitKey(0);
	# featureImage.append(local_binary_pattern(face, no_points, radius, method='ror'))
	
for face in featureImage:
	cv2.imshow("features",face)
	cv2.waitKey(0);