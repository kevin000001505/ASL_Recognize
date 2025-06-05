import cv2
import os
import mediapipe as mp


class GestureRecognition:
    def __init__(self, model_path):
        self.hands_mp = mp.solutions.hands  # hand recognize algorithm
        self.hands = self.hands_mp.Hands()  # Configured the model
        self.mp_draw = mp.solutions.drawing_utils  # Draw the hand landmarks

    def recognize_gesture(self, image):
        results = self.hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
            return True
        return False

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        # Load the model (implementation depends on the specific model format)
