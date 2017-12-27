import cv2
import matplotlib.pyplot as plt
from keras.models import model_from_json
from scipy.misc import imresize
import numpy as np
import os
import copy
import scipy

label_dict = { 0: 'save driving',
            1:'texting with right hand',
            2: 'talking on the phone with right hand',
            3: 'texting with left hand',
            4: 'talking on the phone with left hand',
            5: 'operating the radio',
            6: 'drinking',
            7: 'reaching behind',
            8: 'hair and makeup',
            9: 'talking to passenger'}

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture('p022.mov')

model = model_from_json(open('model_vgg_4.json', 'r').read())
model.load_weights("model_vgg_4.h5")


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while video_capture.isOpened():
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if frame is None:
        break;

    image = np.expand_dims(frame, axis=0)
    pred = model.predict(image)
    label = np.argmax(pred)

    if label!=0:
        x = 51 - len(label_dict[label])
        cv2.putText(frame, label_dict[label], (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], thickness=2);
    
    # Display the resulting image
    cv2.imshow('frame', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

