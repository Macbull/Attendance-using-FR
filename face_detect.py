import sys
from skimage.feature import local_binary_pattern

sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2


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
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2);
    faceImages.append(image[y:y+h,x:x+h]);

cv2.imshow("Faces found", image);
cv2.waitKey(0);
radius = 3
no_points = 8 * radius
featureImage=[];
for face in faceImages:
	cv2.imshow("face", face);
	cv2.waitKey(0);
	im_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
	featureImage.append(local_binary_pattern(im_gray, no_points, radius, method='ror'))
	
for face in featureImage:
	cv2.imshow("features",face)
	cv2.waitKey(0);