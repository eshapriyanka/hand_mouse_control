import cv2
import mediapipe
import pyautogui

capture_hands = mediapipe.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing_option = mediapipe.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

x1 = y1 = x2 = y2 = 0  # For index and thumb
scroll_y = 0  # For middle finger

while True:
    _, image = cap.read()
    image = cv2.flip(image, 1)
    image_height, image_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = capture_hands.process(rgb_image)
    hands = res.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_option.draw_landmarks(image, hand)
            one_hand_landmarks = hand.landmark
            
            for id, lm in enumerate(one_hand_landmarks):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)

                if id == 8:  # Index Finger (Mouse Movement)
                    mouse_x = int(x * screen_width / image_width)
                    mouse_y = int(y * screen_height / image_height)
                    pyautogui.moveTo(mouse_x, mouse_y)
                    x1, y1 = x, y

                if id == 4:  # Thumb (Click Detection)
                    x2, y2 = x, y

                if id == 12:  # Middle Finger (Scroll Detection)
                    scroll_y = y

            # Click Detection
            if abs(y2 - y1) < 30:
                pyautogui.click()

            # Scrolling Detection
            scroll_threshold = 20
            if abs(scroll_y - y1) > scroll_threshold:
                if scroll_y < y1:  # Scroll Up
                    pyautogui.scroll(5)
                else:  # Scroll Down
                    pyautogui.scroll(-5)

    cv2.imshow('Hand movement', image)
    key = cv2.waitKey(1)
    if key == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
