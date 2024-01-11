from tkinter import E
import cv2
import mediapipe as mp
import numpy as np
import os
import cvzone
from cvzone.PoseModule import PoseDetector

mp_drawing = mp.solutions.drawing_utils                    # mediapipe 繪圖功能
mp_selfie_segmentation = mp.solutions.selfie_segmentation  # mediapipe 自拍分割方法
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式

brand_index = 0
bg_img = os.listdir("backgroundImg")
bg_img_List = [] # H&M #UNIQLO #ZARA
for img in bg_img:
    bg_img_List.append(cv2.imread(os.path.join("backgroundImg", img)))

shirt_img_path = os.listdir("shirtImg")
shirt_img_List = []
shirtWidth = 260
shirtHeight = 320
for img in shirt_img_path:
    shirt_img_List.append(
        cv2.resize(
            (cv2.imread(os.path.join("shirtImg", img), cv2.IMREAD_UNCHANGED))
            , (shirtWidth,shirtHeight)
        ))
shirt_ShoulderWidth = [163, 132, 117] # H&M, UNIQLO, ZARA
shirt_offset = [(50,60), (60,50), (70,90)]   # w, h # H&M, UNIQLO, ZARA

leftArrow_img =cv2.resize(cv2.imread("leftArrow.png", cv2.IMREAD_UNCHANGED), (130,130))
rightArrow_img =  cv2.resize(cv2.imread("rightArrow.png", cv2.IMREAD_UNCHANGED), (130,130))

width, height = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

counter = 1
speed = 18
right_next, left_next = True, True

# Initialize the PoseDetector class with the given parameters
detector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=True,
                        smoothSegmentation=True,
                        detectionCon=0.5,
                        trackCon=0.5)







# 額外設定 enable_segmentation 參數
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5, enable_segmentation=True) as pose: 
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        
        img = cv2.resize(img,(1280,720))  
        img = cv2.flip(img, 1) # 水平翻轉 左上角依然是[0,0]
        
        # Find the human pose in the frame
        output_image = detector.findPose(img)
        # Find the landmarks, bounding box, and center of the body in the frame
        # Set draw=True to draw the landmarks and bounding box on the image
        lmList, bboxInfo = detector.findPosition(output_image, draw=False, bboxWithHands=False)

        pose_results = pose.process(img)                  # 取得姿勢偵測結果
        try:
            condition = np.stack((pose_results.segmentation_mask,) * 3, axis=-1) > 0.1 # 如果滿足模型判斷條件 ( 表示要換成背景 )，回傳 True
            output_image = np.where(condition, output_image, bg_img_List[brand_index])
        except:
            pass

        if lmList:

            rightShoulder_posX,rightShoulder_posY = lmList[11][0:2]# 因為cv2.flip 所以 medaipipe 的左右相反
            leftShoulder_posX, leftShoulder_posY = lmList[12][0:2]

            ShoulderWidth = rightShoulder_posX - leftShoulder_posX
            scale_rate = (round(ShoulderWidth, 2) / round(shirt_ShoulderWidth[brand_index], 2))
            
            if scale_rate < 0:
                print(scale_rate)
            
            shirtImgPNG = cv2.resize(shirt_img_List[brand_index]*1.4, (0,0),None,  scale_rate, scale_rate) # x, y
            shirt_X_righttop = leftShoulder_posX - shirt_offset[brand_index][0] * scale_rate
            shirt_Y_righttop = leftShoulder_posY - shirt_offset[brand_index][1] * scale_rate

            
            right_wrist =lmList[15]  # 因為有cv2.flip 所以左右要相反
            left_wrist = lmList[16]

            output_image = cvzone.overlayPNG(output_image, leftArrow_img, (80,300))
            output_image = cvzone.overlayPNG(output_image, rightArrow_img, (1070,300))

            if (300 < right_wrist[1] < 430) and (1070 < right_wrist[0] <1200):
                right_next = True
            else:
                right_next = False

            if (300 < left_wrist[1] < 430) and (80 < left_wrist[0] < 210):
                left_next = True
            else:
                left_next = False

            # left arrow
            if left_next and brand_index > 0: 
                cv2.ellipse(output_image, (145,365), (63,63), 0,0,counter*speed, (0,255,255), 10 )
                counter += 1
                if counter*speed > 360:
                    left_next = False
                    counter = 1
                    brand_index -= 1
            elif right_next and brand_index < 2:
                cv2.ellipse(output_image, (1135,365), (63,63),0 ,0,counter*speed, (0,255,255), 10 )
                counter += 1
                if counter*speed > 360:
                    right_next = False
                    counter = 1
                    brand_index += 1
            else:
                counter = 1
                left_next = False
                right_next = False

            

            output_image = cvzone.overlayPNG(output_image, shirtImgPNG, (int(shirt_X_righttop), int(shirt_Y_righttop))) # x, y
        
        else:
            counter = 1
            right_next = False
            left_next = False

        cv2.imshow('img',output_image)
        if cv2.waitKey(5) == ord('q'):
            break     # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()