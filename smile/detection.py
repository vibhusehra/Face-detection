import cv2
import numpy as np

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml') #load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#turn the webcam on
cam = cv2.VideoCapture(0)

#while the webcam is on
while True:
	#read the image
	ret,img = cam.read()

	#convert the image to Gray
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray,1.3,5)

	for x,y,w,h in faces:
		#cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
		roiGray = gray[y:y+h, x:x+w]
		roiColor = img[y:y+h, x:x+w]
		#find smile inside a given roi(i.e face )
		smile = smile_cascade.detectMultiScale(roiGray,1.1,10)

		for sx,sy,sw,sh in smile:
			cv2.rectangle(roiColor, (sx,sy), (sx+sw,sy+sh), (100,255,100), 3)

	cv2.imshow('img',img) #show the image
	k = cv2.waitKey(30) & 0xff #exit if Esc is pressed
	if k == 27:
		break

cam.release() #release the webcam
cv2.destroyAllWindows() #destroy the window

