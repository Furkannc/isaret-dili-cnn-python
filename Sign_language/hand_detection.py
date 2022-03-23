import cv2
import numpy as np
from keras import models


MODEL_NAME = "sign.h5"

model = models.load_model(MODEL_NAME)
camera = cv2.VideoCapture(0)
text=[]
char=""
while True:
    ret, frame = camera.read()
    
    cv2.rectangle(frame, (100, 100), (600, 600), (255, 0, 0), 2)
    cropped = frame[100:600, 100:600]
    resized = (cv2.cvtColor(cv2.resize(cropped, (28, 28)), cv2.COLOR_RGB2GRAY)) / 255.0
    data = resized.reshape(-1, 28, 28, 1)
    model_out = model.predict([data])[0]
    label = np.argmax(model_out)
  
    if max(model_out) > 0.9:
        if label == 0:
            char = "A"
        elif label == 1:
            char = "B"
        elif label == 2:
            char = "C"
        elif label == 3:
            char = "D"
        elif label == 4:
            char = "E"
        elif label == 5:
            char = "F"
        elif label == 6:
            char = "G"
        elif label == 7:
            char = "H"
        elif label == 8:
            char = "I"
        elif label == 10:
            char = "K"
        elif label == 11:
            char = "L"
        elif label == 12:
            char = "M"
        elif label == 13:
            char = "N"
        elif label == 14:
            char = "O"
        elif label == 15:
            char = "P"
        elif label == 16:
            char = "Q"
        elif label == 17:
            char = "R"
        elif label == 18:
            char = "S"
        elif label == 19:
            char = "T"
        elif label == 20:
            char = "U"
        elif label == 21:
            char = "V"
        elif label == 22:
            char = "W"
        elif label == 23:
            char = "X"
        elif label == 24:
            char = "Y"
     
        cv2.putText(frame,char,(10,120),cv2.FONT_HERSHEY_SIMPLEX,5,(0,255,0),3,cv2.LINE_AA)
    
    cv2.putText(frame,str(text)[1:-1],(10,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3,cv2.LINE_AA)

    cv2.imshow('frame', frame)
   
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    elif cv2.waitKey(32) == ord(" "):
        text.append(str(char))
  

camera.release()

