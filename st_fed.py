import streamlit as st
import cv2
from fer import FER
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

st.title("Facial Emotion Detector")

picture = st.camera_input("Take a picture")
if picture :
    bytes_data = picture.getvalue()
    input_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # input_image=cv2.imread('Photo.jpg')

    emotion_detector = FER(mtcnn=True)

    result=emotion_detector.detect_emotions(input_image)

    bounding_box=result[0]["box"]
    emotions = result[0]["emotions"]
    cv2.rectangle(input_image,(
    bounding_box[0], bounding_box[1]),(
    bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                (0, 155, 255), 2,)

    emotion_name, score = emotion_detector.top_emotion(input_image)
    for index, (emotion_name, score) in enumerate(emotions.items()):
        color = (255, 255,255) if score < 0.01 else (255, 255, 255)
        emotion_score = "{}: {}".format(emotion_name, "{:.2f}".format(score))

        cv2.putText(input_image,emotion_score,
                (bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + index * 15),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA,)
    
    #Save the result in new image file
    # output=cv2.imwrite("emotion.jpg", input_image)



    st.image(input_image)

   

    st.success("Your emotion is {}".format(emotion_name))


