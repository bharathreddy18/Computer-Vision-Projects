from PIL import Image
import cvzone
from cvzone.HandTrackingModule import HandDetector
import cv2
import google.generativeai as genai
import numpy as np
import streamlit as st


class GestureMathAI:
    def __init__(self, api_key, prompt):
        self.prompt = prompt
        self.canvas = None
        self.previous_position = None
        self.output_text = ""

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        self.detector = HandDetector(staticMode=False,
                                     maxHands=1,
                                     modelComplexity=1,
                                     detectionCon=0.7,
                                     minTrackCon=0.5)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_hands_info(self, frame):
        hands, _ = self.detector.findHands(frame, draw=False, flipType=True)
        if hands:
            return self.detector.fingersUp(hands[0]), hands[0]["lmList"]
        return None

    def draw_on_canvas(self, hand_info):
        fingers, landmarks = hand_info
        current_position = None

        if fingers == [0, 1, 0, 0, 0]:
            current_position = landmarks[8][:2]
            if self.previous_position is None:
                self.previous_position = current_position
            cv2.line(self.canvas, tuple(current_position), tuple(self.previous_position), (255, 0, 255), 10)

        elif fingers == [1, 1, 1, 1, 1]:
            self.canvas = np.zeros_like(self.canvas)
            self.output_text = ""

        self.previous_position = current_position

    def send_to_ai(self, hand_info):
        fingers, _ = hand_info
        if fingers == [1, 1, 1, 1, 0]:
            try:
                pil_image = Image.fromarray(self.canvas)
                response = self.model.generate_content([self.prompt, pil_image])
                return response.text
            except Exception as e:
                return f'Error: {str(e)}'

        elif fingers == [1, 1, 1, 1, 1]:
            return None

        return self.output_text

    def run(self):
        st.set_page_config(layout='wide')
        st.image('POV.png')
        col1, col2 = st.columns([2, 1])

        with col1:
            run_app = st.checkbox('Run', value=True)
            frame_window = st.image([])

        with col2:
            st.title("Answer")
            output_display = st.subheader("")

        try:
            while run_app:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                if self.canvas is None:
                    self.canvas = np.zeros_like(frame)

                hand_info = self.get_hands_info(frame)
                if hand_info:
                    self.draw_on_canvas(hand_info)
                    self.output_text = self.send_to_ai(hand_info)

                blended_image = cv2.addWeighted(frame, 0.7, self.canvas, 0.3, 0)
                frame_window.image(blended_image, channels="BGR")

                if self.output_text is None:
                    output_display.text("")
                else:
                    output_display.text(self.output_text)
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    ai_instance = GestureMathAI(api_key="AIzaSyBGcmDihi-O69Vv2y6AqBk6dmqPVo_haeE", prompt="Guess the shape")
    ai_instance.run()