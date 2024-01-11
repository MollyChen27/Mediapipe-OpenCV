from statistics import mode
import cv2
# from cvzone.HandTrackingModule import HandDetector
import os
import numpy as np
import face_recognition
import pickle
import cvzone
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
# variable 
width, height = 1280, 720
wCamara, hCamara = 650, 450

# folderMode path
folderMode_Path = os.listdir("resource/mode")
folderModeimgs_List = []
for img_path in folderMode_Path:
   folderModeimgs_List.append(cv2.resize(cv2.imread(os.path.join("resource/mode", img_path)),  (550, height)))
modeType = 0
# importing encoding face
with open('EncodeFile.pickle', 'rb') as f:
    encodeListKnownWithIds = pickle.load(f)
face_encoding_List, StudentIDs = encodeListKnownWithIds

# Importing student images
StudentImg_Path = os.listdir("image")
StudentImg_List = []
for img_path in StudentImg_Path:
    StudentImg_List.append(os.path.join("image", img_path))

cred = credentials.Certificate("ServiceAccount.json")
# databaseURL指定Firebase實時數據庫的URL，用於初始化與實時數據庫的連接。
# storageBucket指定Firebase存儲桶的URL，用於初始化與Firebase存儲服務的連接。
firebase_admin.initialize_app(cred, 
    {"databaseURL":"https://facerecognition-df4ce-default-rtdb.firebaseio.com/",
    "storageBucket":"facerecognition-df4ce.appspot.com"}
)

counter = 0

bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3,width) # width 
cap.set(4,height) # height

while True:
    success, img = cap.read()
    imgCamara= cv2.resize(img, (wCamara, hCamara))

    backgroundImg = cv2.imread("resource/background.png")
    backgroundImg[0:height, 730:width] = folderModeimgs_List[modeType] # h: 720； w: 550
    backgroundImg[220: 670, 40:690] = imgCamara # h: 450； width: 650 => y,x

    imgS = cv2.resize(imgCamara ,(0,0),None,  0.25, 0.25) # 為了節省计算资源，將圖片變小 # (0,0) means that the output size is not specified directly but will be determined based on the scale factors.
    imgS = cv2.cvtColor(imgS , cv2.COLOR_BGR2RGB)

    # 偵測多個人臉，若没有检测到人脸，以下兩個列表為空
    faceCurLocation = face_recognition.face_locations(imgS) # 检测图像中人脸的位置
    faceCurEncoding = face_recognition.face_encodings(imgS, faceCurLocation) # 使用前面检测到的人脸位置信息，获取每个人脸的编码
        

    if faceCurLocation: # 若有偵測到臉的情況下
        
        for encodeCurFace, faceCurLoc in zip(faceCurEncoding, faceCurLocation):
            matches = face_recognition.compare_faces(face_encoding_List,encodeCurFace) # 可能發生待检测人脸与已知列表中的多个人脸都有相似，因此返回多个 True 值
            faceDis = face_recognition.face_distance(face_encoding_List, encodeCurFace) # The distance tells you how similar the faces are.
            # print("matches", matches)
            # print("distance", faceDis)

            matchIndex = np.argmin(faceDis)
            # print(matchIndex)

            # Add a rectangle with styled corners to the image
            top, right, bottom, left = faceCurLoc
            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            bbox = left+40, top+220, right-left, bottom - top # (x, y, width, height) ，因為left 和 top 原本都是以左上方(0, 0)為原點，但現在imgCamara的(0,0)在backgroundIng被移到(40,200)
            backgroundImg = cvzone.cornerRect(
                backgroundImg,  # The image to draw on
                bbox,  # The position and dimensions of the rectangle (x, y, width, height)
                rt=0,  # Thickness of the rectangle
                colorC=(0, 255, 0)  # Color of the corner edges
            )


            student_id = StudentIDs[matchIndex]

            if counter == 0: # active mode
                cv2.putText(
                    backgroundImg, "Loading", (300,445),  # Image and starting position of the rectangle x,y
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    color=(0, 255, 0),  # BGR
                    fontScale = 1.5,
                    thickness = 5
                )
                cv2.imshow("background", backgroundImg) # 名稱相同的圖 會畫在同一張圖上
                cv2.waitKey(1) # 程序会等待1毫秒

                counter = 1 # show student info , marked mode or already marked mode

            # 一次只偵測一個人臉
            break

    if counter !=0:
        # show student info前 先下載image
        if counter == 1:
            # print(student_id)
            ref = db.reference(f'students/{student_id}')
            studentInfo = ref.get()
            # print(studentInfo)

            last_attendance_time = studentInfo["last_attendance_time"]
            # 取得現在的時間
            current_time = datetime.now()
            current_time_str =  current_time.strftime("%Y-%m-%d %H:%M:%S")
            # Convert strings to datetime objects
            current_time_format = datetime.strptime( current_time_str, "%Y-%m-%d %H:%M:%S")
            last_attendance_time_format = datetime.strptime(last_attendance_time, "%Y-%m-%d %H:%M:%S") 
            # Calculate the time difference
            time_difference = current_time_format - last_attendance_time_format
            # 若距離上一次簽到時間 在15秒內，則顯示already marked
            if time_difference < timedelta(seconds=15):
                modeType = 3 # already marked
                counter = 0 
                backgroundImg[0:height, 730:width] = folderModeimgs_List[modeType]
            else: # 重新簽到
                modeType = 1
                # 下載圖片
                blob = bucket.get_blob(f"image/{student_id}.png")
                image_np = np.frombuffer(blob.download_as_string(), np.uint8)
                StudentImage = cv2.imdecode(image_np, cv2.COLOR_BGRA2BGR)
                last_attendance_time_str = current_time_str
                ref.child("last_attendance_time").set(last_attendance_time_str)
                studentInfo["total_attendance"] += 1
                ref.child("total_attendance").set(studentInfo["total_attendance"])
        
        if modeType != 3: # 非在already marked
            # marked
            if 10 < counter < 20:
                modeType = 2

            backgroundImg[0:height, 730:width] = folderModeimgs_List[modeType]

            # show info
            if counter <= 10:
                text_size = cv2.getTextSize(studentInfo["name"], cv2.FONT_HERSHEY_COMPLEX, 1.2, 2 )[0]
                text_width = text_size[0]
                # name
                cv2.putText(backgroundImg, studentInfo["name"], (int((550 - text_width)//2) + 730, 480),cv2.FONT_HERSHEY_COMPLEX, 1.2, (255,255,255), 2 )
                # time
                cv2.putText(backgroundImg,  current_time_str, (945, 550),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1 )
                # major
                cv2.putText(backgroundImg, studentInfo["major"], (945, 635),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1 )
                # total_attendance
                cv2.putText(backgroundImg, str(studentInfo["total_attendance"]), (800, 100),cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 2 )

                backgroundImg[150:150+280, 870:870+280] = StudentImage

            counter += 1

            if counter >= 20: # reset
                counter = 0
                modeType = 0 # active
                backgroundImg[0:height, 730:width] = folderModeimgs_List[modeType]

    else: # 若counter為0，代表在active mode
        modeType = 0
        counter = 0
        

    
    cv2.imshow("background", backgroundImg)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()