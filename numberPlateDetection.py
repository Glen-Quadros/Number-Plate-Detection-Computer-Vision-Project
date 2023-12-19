import numpy as np
import matplotlib.pyplot as plt
import cv2
import easyocr # 
import imutils # Contour Detection

img = cv2.imread('/Users/glenquadros/Downloads/img.png')

greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(cv2.cvtColor(greyscale_img, cv2.COLOR_BGR2RGB))

bfilter = cv2.bilateralFilter(greyscale_img, 11, 17, 17)

"""
The bilateral filter is a non-linear filter that preserves the edges while reducing noise in an image.

cv2.bilateralFilter(input, diameter of each pixel neighborhood, sigmaColor, sigmaSpace)

sigmaColor: This parameter controls the filter's effect on color space. 
A higher value means that the pixels with a larger color difference will 
have less influence on each other during the filtering process. 

sigmaSpace: This parameter determines the filter's effect on spatial space. 
A higher value means that pixels farther away from the center pixel will have 
less influence on its filtering process.
"""

edged = cv2.Canny(bfilter, 30, 30)

'''
Canny algorithm requires greyscale image.

The cv2.Canny() function takes the filtered image (bfilter) as the input and 
applies the Canny edge detection algorithm.

The first threshold (30 in this case) is the lower threshold. Any gradient value 
below this threshold is considered non-edge and suppressed.
The second threshold (30 here) is the upper threshold. Any gradient value above 
this threshold is considered an edge.
'''

plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

'''
cv2.RETR_TREE retrieves all contours and reconstructs a full hierarchy of nested contours.

cv2.CHAIN_APPROX_SIMPLE: This parameter specifies the contour approximation method. It compresses 
horizontal, vertical, and diagonal segments into their respective end points, saving memory by 
discarding unnecessary points. 
'''

contours = imutils.grab_contours(keypoints)

'''
the function extracts and returns the actual contours. The resulting contours are stored in the contour variable.
'''

contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

'''
cv2.contourArea: It is a function provided by OpenCV that calculates the area of a contour.
'''

location = None
for contour in contours: 
  approx = cv2.approxPolyDP(contour, 10, True)
  if len(approx) == 4: # To find a rectangle/quadrilateral
    location = approx
    break

'''
approx = cv2.approxPolyDP(contour, 10, True): Applies the Douglas-Peucker algorithm 
(approxPolyDP) to approximate the contour shape with a simpler polygon. It reduces 
the number of vertices in the contour. 10 parameter round off the shape The True parameter 
indicates that the contour is closed.
'''

mask = np.zeros(greyscale_img.shape,np.uint8)

new_image = cv2.drawContours(mask,[location],0,255,-1,)

'''
new_image = cv2.drawContours(mask, [location], 0, 255, -1): Draws the contour 
stored in the location variable onto the mask image. The cv2.drawContours() 
function takes the mask image, the contour ([location] as a list of contours), 
the contour index (0 in this case), the color (255 for white), and the thickness 
(-1 fills the contour with the specified color).
'''

new_image = cv2.bitwise_and(img,img,mask=mask)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = greyscale_img[x1:x2+1, y1:y2+1]

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

# Use Easy OCR To Read Text

reader = easyocr.Reader(['en'])

result = reader.readtext(cropped_image)

print(f'The number plate of the car is: {result[0][1]}')