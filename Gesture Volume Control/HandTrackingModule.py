import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self,
                 mode=False,
                 maxHands=2,
                 detectionConf=0.5,
                 trackConf=0.5):

        self.hand_params = {
            "static_image_mode":mode,
            "max_num_hands":maxHands,
            "min_detection_confidence": detectionConf,
            "min_tracking_confidence": trackConf
        }

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(**self.hand_params)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,
                                               handLms,
                                               self.mpHands.HAND_CONNECTIONS,
                                               self.mpDraw.DrawingSpec((0,0,255), 2, 3),
                                               self.mpDraw.DrawingSpec((0,255,0), 2))
        return img

    def findPositions(self, img, handNo=0, draw=True):
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255,0,255), -1)
        return self.lmList

    def fingersUp(self):
        fingers = []

        # if self.lmList[17][1] < self.lmList[2][1]:
        # Right Thumb - This is for the cv2.flip(img, 1) image
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # else:
        # Left Thumb
        # if self.lmList[tipIds[0]][1] < self.lmList[tipIds[0] - 1][1]:
        #     fingers.append(1)
        # else:
        #     fingers.append(0)

        # Four Fingers Right
        for i in range(1, len(self.tipIds)):
            if self.lmList[self.tipIds[i]][2] < self.lmList[self.tipIds[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    pTime = 0
    cTime = 0

    while True:
        success, img = cap.read()
        if not success:
            print('Failed to capture image')
            break

        img = detector.findHands(img)
        lmList = detector.findPositions(img)
        print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

        cv2.imshow('Image', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


