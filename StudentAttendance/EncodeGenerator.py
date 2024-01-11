# pip install cmake
# conda install -c conda-forge dlib
# pip install face_recognition
import face_recognition
import os
import cv2
import pickle
import numpy as np
import pickle
from sqlite3 import Timestamp
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("ServiceAccount.json")
# databaseURL指定Firebase實時數據庫的URL，用於初始化與實時數據庫的連接。
# storageBucket指定Firebase存儲桶的URL，用於初始化與Firebase存儲服務的連接。
firebase_admin.initialize_app(cred, 
    {"databaseURL":"https://facerecognition-df4ce-default-rtdb.firebaseio.com/",
    "storageBucket":"facerecognition-df4ce.appspot.com"}
)

# Importing student images
StudentImg_Path = os.listdir("image")
StudentImg_List = []

for img_path in StudentImg_Path:
    StudentImg_List.append(os.path.join("image", img_path))
    
    # upload images to Firebase storage bucket
    filename = f"image/{img_path}"
    bucket = storage.bucket() # 取得對特定存儲桶的引用
    blob = bucket.blob(filename) # 創建文件對象
    blob.upload_from_filename(filename) # 從本地文件上傳文件到存儲桶

StudentID = []
for id in StudentImg_Path:
    StudentID.append(id.split(".")[0])

# encode face
face_encoding_List = []
for img_path in StudentImg_List:
    image = face_recognition.load_image_file(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_encoding = face_recognition.face_encodings(image)[0] # 只有一張臉，所以[0]
    face_encoding_List.append(face_encoding)

encodeListKnownWithIds = [face_encoding_List , StudentID]
# print(encodeListKnownWithIds)

with open("EncodeFile.pickle", "wb") as f:
    pickle.dump(encodeListKnownWithIds, f)
print(StudentImg_List)
