import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image,line_parameters):
    slope,intercept = line_parameters
    y1= image.shape[0]
    y2= int(y1*(3/5))
    # x=(y-b)/m
    x1= int((y1-intercept)/slope)
    x2= int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit= []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        # y = mx+b to get slope m = (y2-y1)/(x2-x1)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        # slope = m
        slope = parameters[0]
        # intercept = snijpunt met x as = b
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])




def canny(image):
    # make it grayscale so you can detect changes in contrast
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # blur to reduce noise in the image, optional --> cv2.Canny() does this for you
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #Canny function takes derivate of X,Y coordinates of the matrix of pixels the image represents
    #Shows most rapid changes in brightness if it exceeds the thresh hold(150) or lowest threshhold(50)--- Gradients
    canny = cv2.Canny(blur,50,150)
    return canny

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            #x1,y1,x2,y2 = line  #.reshape(4) not needed anymore to convert 3dimensional array to 2dimensional
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def region_of_interest(image):
    #To get the triangle that shows the Road lines
    height = image.shape[0]
    polygons = np.array([
    [(200,height),(1100,height),(550,250)]
    ])
    #create a mask same size as image but with only the triangle indicating the road
    #mask will be black
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    # white part of triangle is  of bitwsie 1111 and black part of triangle is of bitwise 0000
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image


## load image
#image = cv2.imread('test_image.jpg')
## make a copy so it isn't overwritten by anything you do to it
#lane_image = np.copy(image)
#canny_image = canny(lane_image)
#cropped_image = region_of_interest(canny_image)
## to choose which points best describes the line "Hough-space"
## algorithm to fit the line.
#lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
#averaged_lines = average_slope_intercept(lane_image,lines)
#line_image= display_lines(lane_image,averaged_lines)
#overlap images
#Overlapped_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)
## show the result
#cv2.imshow('result',Overlapped_image)
## keeps window open
#cv2.waitKey(0)

#video capture algoritme laten checken op een video
cap = cv2.VideoCapture("test2.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    averaged_lines = average_slope_intercept(frame,lines)
    line_image= display_lines(frame,averaged_lines)
    Overlapped_image = cv2.addWeighted(frame,0.8,line_image,1,1)

    cv2.imshow('result',Overlapped_image)
    #wait 1 milisecond
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
