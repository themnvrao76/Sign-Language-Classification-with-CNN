import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras 
from PIL import Image
import string
import os
categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
        'L', 'M', 'N','O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
        'W', 'X', 'Y', 'Z','middle finger','nothing']

model = keras.models.load_model("version98.h5")
cam=cv2.VideoCapture(cv2.CAP_DSHOW)

while True:
    ret,frame =cam.read()
    img=cv2.rectangle(frame,(0,0),(250,250),(255,0,0),3)
    roi=img[0:250,0:250]
    resizeimg=cv2.resize(roi,(64,64))
    resizeimg=cv2.cvtColor(resizeimg,cv2.COLOR_BGR2GRAY)
    imgarray=np.array(resizeimg)

    imgarray=imgarray/255
    imgarray=np.expand_dims(imgarray, axis=0)
    imgarray=np.expand_dims(imgarray, axis=-1)
  

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20,300)
    fontScale              = 1
    fontColor              = (0,0,255)
    lineType               = 2
    cv2.putText(frame,categories[(np.argmax(model.predict(imgarray)))], 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)

    cv2.imwrite("img.jpg",resizeimg)
    cv2.imshow("freame",img)
    cv2.imshow("roi",resizeimg)
    
    

   

    
    if cv2.waitKey(1) == ord('q'):
        break
        
cam.release()
cv2.destroyAllWindows()

