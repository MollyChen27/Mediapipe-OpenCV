import cv2
from cvzone.HandTrackingModule import HandDetector
import os
import numpy as np
import cvzone

# variable 
width, height = 1280, 720
wCamara, hCamara = int(213*1.2), int(120*1.2)
wComputer, hComputer = 460, 300
gestureThreshold = height//2 # 控制上下頁 必須把手伸超過某個高度

# 切換PPT: 需要相隔一段時間才能換頁
pauseCounter = 0  
pauseFrames = 15 
switchPPT = False

# PPT imgs path
PPT_Path = os.listdir("presentation")
PPTimgs_Array = []
for img_path in PPT_Path:
   PPTimgs_Array.append(os.path.join("presentation", img_path))
PPTMode = 0

# painting imgs path
Painting_Path = os.listdir("painting")
Paintingimgs_Array = []
for painting_path in Painting_Path:
   Paintingimgs_Array.append(os.path.join("painting", painting_path))

pointerColor = (0,0,255) # 一開始為紅色
PaintingMode = 1 # 一開始為紅色
PaintingPreviousMode = 1 # 一開始為紅色

xp, yp = 0,0 # 上一個畫線的位置

# trash can path
Trash_Path = os.listdir("trash")
Trashimgs_Array = []
for img_path in Trash_Path:
   Trashimgs_Array.append(cv2.imread(os.path.join("trash", img_path)))
TrashMode = 0

# draw lines
line_List = []
draw_line = False

# erase:　每擦掉一個圖後　要相隔一段時間才能繼續把圖擦掉
eraseDraw = False
eraseCounter = 0 #  迴圈跑一次，cap.read()就讀取一個frame
eraseFrames = 15 

# img Canvas
imgCanvas = np.zeros((720,1280,3), np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3,width) # width 
cap.set(4,height) # height

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.7)

while True:
    success, img = cap.read()
    imgCurrent = cv2.imread(PPTimgs_Array[PPTMode])
    paintingCurrent = cv2.imread(Paintingimgs_Array[PaintingMode])
    trashCurrent = cv2.imread(Trashimgs_Array[TrashMode])

    paintingCurrent = cv2.resize(paintingCurrent, (150, height))
    trashCurrent = cv2.resize(trashCurrent, (220, 360)) # w, h

    img = cv2.resize(img, (width, height))
    img = cv2.flip(img, 1) # 水平翻轉
    cv2.line(img, (0, gestureThreshold), (width,gestureThreshold), (0,255,0), 5) # 線的上半部才是作用區
    hands, img = detector.findHands(img, draw=True, flipType=False)
    
    # 畫盤
    imgCurrent[0:height, width-150:width] = paintingCurrent

    # Check if any hands are detected
    if hands and switchPPT ==  False:
        hand1 = hands[0]  # Get the first hand detected
        cx, cy = hand1['center']
        lmList1 = hand1["lmList"]
        
        fingers1 = detector.fingersUp(hand1)

        # 動作只要在某個小範圍，即可控制整個PPT
        xVal = int(np.interp(lmList1[8][0], [width // 2-150, width-100], [0, width]))
        yVal = int(np.interp(lmList1[8][1], [150, height-150], [0, height]))
        indexFinger = xVal, yVal
        

        if cy <= gestureThreshold:
            # Gesture1: move to previous page
            if fingers1 == [1,0,0,0,0]:
                line_List = []
                if PPTMode > 0:
                    PPTMode -= 1
                    switchPPT = True
                imgCanvas = np.zeros((720,1280,3), np.uint8)    

            # Gesture2: move to next page
            if fingers1 == [0,0,0,0,1]:
                line_List = [] # 清空畫筆
                if PPTMode < len(PPTimgs_Array)-1:
                    PPTMode += 1
                    switchPPT = True
                imgCanvas = np.zeros((720,1280,3), np.uint8)

        # Gesture3: pointer
        if fingers1 == [0, 1, 0, 0, 0]:
            # change pointer color
            if width-180 < xVal < width:
                # red
                if 25 < yVal < 120:
                    PaintingMode = 1
                    pointerColor = (0,0,255)
                # blue
                elif 195 < yVal < 293:
                    PaintingMode = 2
                    pointerColor = (255, 0,0)
                # yellow
                elif 330 < yVal < 460:
                    PaintingMode = 3
                    pointerColor = (0, 255,255)
                # erase
                elif 530 < yVal < 720:
                    PaintingMode = 0
                    pointerColor =(0, 0, 0)

            # erase pointer       
            if PaintingMode == 0:
                cv2.circle(imgCurrent, indexFinger, 60, pointerColor, cv2.FILLED)
            # draw pointer
            else:
                cv2.circle(imgCurrent, indexFinger, 10, pointerColor, cv2.FILLED)

        # Gesture5: draw
        # if fingers1 == [0, 1, 1, 0, 0]:
        #     cv2.circle(imgCurrent, indexFinger, 10, pointerColor, cv2.FILLED)
        #     if(draw_line==False):
        #         line_List.append([]) # 為了讓每次畫的圖 不要相互連接
        #         draw_line = True # 與上一個圖 分出區別
        #     line_List[len(line_List)-1].append(indexFinger +  pointerColor) # 將tutple合併
        # else:
        #     draw_line = False

        # Gesture6: erase
        # if eraseDraw == False:
        #     if fingers1 == [0, 1, 1, 1, 0]:
        #         eraseDraw = True
        #         if len(line_List) > 0:
        #             line_List.pop(-1)
        
        # Gesture4: draw and erase
        if fingers1 == [0, 1, 1, 0, 0]:
            # 當有變換畫筆顏色qqㄆ則初始化上一步動作xp, yp
            if PaintingPreviousMode != PaintingMode:
                xp, yp = 0,0
            PaintingPreviousMode = PaintingMode

            # imgCurrent 上的動作: 只有pointer       
            if PaintingMode == 0: # PaintingMode: erase
                cv2.circle(imgCurrent, indexFinger, 60, pointerColor, cv2.FILLED)
            else: # PaintingMode: draw
                cv2.circle(imgCurrent, indexFinger, 10, pointerColor, cv2.FILLED)

            # imgCanvas上的動作: draw lines
            if xp!=0 and yp !=0: # 若上一步有動作
                if PaintingMode == 0: # 若為erase，就在imgCanvas上塗上黑色畫筆，等同於erase
                    cv2.circle(imgCanvas, indexFinger, 60, pointerColor, cv2.FILLED)
                else: # 若為畫筆，就在imgCanvas上色
                    cv2.line(imgCanvas, (xp, yp), indexFinger, pointerColor, 10)
                xp, yp =  indexFinger
            else:
                xp, yp =  indexFinger
            draw_line = True 
        else: # 跳到其他非畫圖的動作，則初始化上一步動作xp, yp
            draw_line = False
            xp, yp = 0,0

    # # draw lines
    # for i in range(len(line_List)): # 有幾個圖
    #     for j in range(len(line_List[i])): # 每個圖中有幾個點
    #         if j != 0: # j, i 都是index 
    #             cv2.line(imgCurrent, line_List[i][j-1][:2],   line_List[i][j][:2], line_List[i][j][2:],10)

    if switchPPT:
        pauseCounter += 1
        if pauseCounter > pauseFrames: 
            switchPPT = False
            pauseCounter = 0

    if eraseDraw:
        eraseCounter += 1
        if eraseCounter > eraseFrames: 
            eraseDraw = False
            eraseCounter = 0

    # 疊加畫面: 讓畫在imgCanvas的塗鴉，可以顯示在imgCurrent上
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY) # imgGray的背景為黑色，畫筆顏色為不同深度的灰色
    _, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV) #THRESH_BINARY_INV: below 20 will be set to the maximum value (255)，因此imgInv的背景為白色，畫筆顏色為黑色
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR) # 改成三色通道才可以與imgCurrent疊加，但畫布顏色依舊是 背景為白，畫筆為黑
    imgAnd = cv2.bitwise_and(imgCurrent,imgInv) # 背景為色彩PPT，畫筆為黑色
    imgCurrent = cv2.bitwise_or(imgAnd,imgCanvas) # 背景為色彩PPT，畫筆為彩色

    # put camera on the background
    imgCamara = cv2.resize(img, (wCamara, hCamara))
    imgComputer = cv2.resize(img, (wComputer, hComputer))
    if PPTMode == 5:
        imgCurrent[215:515, 125:585] = imgComputer # hComputer , wComputer = 300, 460
        imgCurrent[210:570,840:1060 ] = trashCurrent #  h : 360, w:220
    else:
        imgCurrent[0:hCamara, 0:wCamara] = imgCamara
        

    cv2.imshow("imgCurrent", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
