from sqlite3 import Timestamp
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("ServiceAccount.json")
firebase_admin.initialize_app(cred, 
    {"databaseURL":"https://facerecognition-df4ce-default-rtdb.firebaseio.com/"}
)

ref = db.reference("students")
data = {
    "109654320":{
        "name":"Sun Li",
        "major": "Commodity Science",
        "total_attendance":0,
        "last_attendance_time": "2023-11-2 10:01:00"
    },
    "110917025":{ 
        "name":"Chen Chia En",
        "major": "Educational Management",
        "total_attendance":0,
        "last_attendance_time": "2023-11-1 09:01:00"
    },
    "111569875":{
        "name":"Bai Jingting",
        "major": "Recording",
        "total_attendance":0,
        "last_attendance_time": "2023-11-3 06:51:00"
    },
    "222222222":{ 
        "name":"teacher",
        "major": "Math",
        "total_attendance":2,
        "last_attendance_time": "2023-11-1 09:00:00"
    }

}

for key, value in data.items():
    ref.child(key).set(value)