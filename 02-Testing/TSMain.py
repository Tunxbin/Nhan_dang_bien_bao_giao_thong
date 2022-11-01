import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

modelPath = 'C:/Users/win/Downloads/PBL4_Test/03-Training/Models'
model = keras.models.load_model(modelPath+'/TSModel')

def returnRedness(img):
	yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(yuv)
	return v

def threshold(img,T=180):
	_, img = cv2.threshold(img,T,255,cv2.THRESH_BINARY)
	return img 

def findContour(img):
	contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return contours

def findBiggestContour(contours):
	m = 0
	c = [cv2.contourArea(i) for i in contours]
	return contours[c.index(max(c))]

def boundaryBox(img,contours):
	x, y, w, h = cv2.boundingRect(contours)
	img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
	sign = img[y:(y+h) , x:(x+w)]
	return img, sign

def preprocessingImageToClassifier(image=None,imageSize=28,mu=92.09727905792265,std=71.1439712262546):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image,(imageSize,imageSize))
    image = (image - mu) / std
    image = image.reshape(1,imageSize,imageSize,1)
    return image

def predict(sign):
	img = preprocessingImageToClassifier(sign,imageSize=28)
	return np.argmax(model.predict(img))

#--------------------------------------------------------------------------
labelToText = { 0:"Dừng",
    			1:"Cấm đi ngược chiều",
    			2:"Giao nhau có tín hiệu đèn",
    			3:"Giao nhau với đường ưu tiên",}
cap=cv2.VideoCapture(r'C:\Users\win\Downloads\PBL4_Test\1.mp4')

while(True):
	ret, frame = cap.read()
	frame = cv2.resize(frame, (640,480))
	redness = returnRedness(frame) 
	thresh = threshold(redness) 	
	try:
		contours = findContour(thresh)
		big = findBiggestContour(contours)
		if  cv2.contourArea(big) > 0:
			#cv2.contourArea(big) > 3000:
			print(cv2.contourArea(big))
			img,sign = boundaryBox(frame,big)
			cv2.imshow('frame',img)
			print("Biển báo:",labelToText[predict(sign)])
		else:
			cv2.imshow('frame',frame)
	except:
		cv2.imshow('frame',frame)

	if cv2.waitKey(30) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()