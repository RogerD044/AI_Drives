#pip3 install opencv-python

import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
import math

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.resize(cv2.imread('steering.jpg',0), (240,240))
rows,cols = img.shape

smoothed_angle = 0


#read data.txt
xs = []
ys = []
with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * scipy.pi / 180)

#get number of images
num_images = len(xs)


i = math.ceil(num_images*0.8)
print("Starting frameofvideo:" +str(i))

while(cv2.waitKey(10) != ord('q')):
    full_image = scipy.misc.imread("driving_dataset/" + str(i) + ".jpg", mode="RGB")
    # Rescaling the image focusing on the road thus using only last 150 pixels 
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    # Converting the angle in radian back to degree
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
    print("Steering angle: " + str(degrees) + " (pred)\t" + str(ys[i]*180/scipy.pi) + " (actual)")
    # Displaying the image of the road
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    # getRotationMatrix2D(Point2f center, double angle, double scale) // Calculates an affine matrix of 2D rotation.
    # Parameters:	
    #    center – Center of the rotation in the source image.
    #    angle – Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
    #    scale – Isotropic scale factor.
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    # Applies an affine transformation to an image (inp_image, affine Matrix, dimensions of o/p image)
    dst = cv2.warpAffine(img,M,(cols,rows))
    # Displaying the image of Steering
    cv2.imshow("steering wheel", dst)
    #cv2.waitKey(0)
    i += 1

cv2.destroyAllWindows()

 