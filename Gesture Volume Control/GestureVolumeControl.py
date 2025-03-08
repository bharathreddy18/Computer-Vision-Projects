from HandTrackingModule import HandDetector
import cv2
import time
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, IPropertyStore

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 20)
pTime = 0

detector = HandDetector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
# print(vol_range)
min_vol, max_vol = vol_range[0], vol_range[1]
print(f"Volume range: {min_vol} to {max_vol}")
volBar = 400
volPer = 0
while True:
    ret, img = cap.read()
    if not ret:
        break
    img = detector.findHands(img)
    lmList = detector.findPositions(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1,y1), 10, (255,0,255), -1)
        cv2.circle(img, (x2,y2), 10, (255, 0, 255), -1)
        cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 2)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), -1)

        length = math.hypot(x2-x1, y2-y1)
        # Apply log scaling to hand distance
        log_length = np.log10(length + 1) # Adding 1 to avoid log(0)

        # Adjusting input range to match hand distance
        vol = np.interp(log_length, [np.log10(30+1), np.log10(250+1)], [min_vol, max_vol])
        volBar = np.interp(length, [30, 250], [400, 150])
        # volPer = np.interp(length, [30, 250], [0, 100])
        print(length, vol)

        volume.SetMasterVolumeLevel(vol, None)

        if length < 30:
            cv2.circle(img, (cx,cy), 10, (0,255,0), -1)

    cv2.rectangle(img, (50,150), (85, 400), (0,255,0), 2)
    cv2.rectangle(img, (50, int(volBar)), (85,400), (0,255,0), -1)
    # cv2.putText(img, f'{int(volPer)}%', (55, 140), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (30,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1)

    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()