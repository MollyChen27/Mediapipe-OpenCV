from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from playsound import playsound
import threading

audio_file_path = 'AlarmClock.mp3'  # 替换为你的音频文件路径


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model: from teachable machine website
model = load_model("model/keras_Model.h5", compile=False)

# Load the labels
class_names = open("model/labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
width = 1280
height = 720
camera = cv2.VideoCapture(0)
camera.set(3,width)
camera.set(4,height)

def play_audio(audio_file_path):
    playsound(audio_file_path)  # 播放音频文件

def start_audio_thread(audio_file_path):
    thread = threading.Thread(target=play_audio, args=(audio_file_path,))
    thread.daemon = True  # 设置为守护线程，程序退出时会自动结束线程
    thread.start()

frame_counter = 0
alarm = False
while True:
    # Grab the webcamera's image.
    ret, bgImg = camera.read()
    bgImg = cv2.resize(bgImg,(width, height))

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(bgImg, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    # cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    # confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    label = class_name[2:-1]
    cv2.putText(bgImg, label, (600,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 5)


    if label == "Drowsy":
        if frame_counter == 0:
            print("play alarm")
            start_audio_thread(audio_file_path)
            alarm = True
        
    else:
        pass
    

    if alarm:
        print(frame_counter)
        frame_counter += 1
        if frame_counter > 60:
            alarm = False
            frame_counter = 0

    cv2.imshow('img',  bgImg)
    if cv2.waitKey(5) == ord('q'):
        break     # 按下 q 鍵停止

camera.release()
cv2.destroyAllWindows()
