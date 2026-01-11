import cv2
import pyautogui
import numpy as np
import time
import math
import screen_brightness_control as sbc
import mediapipe as mp

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------------- CONFIG ----------------
SMOOTHING = 6
CLICK_THRESHOLD = 30
CLICK_COOLDOWN = 0.35
GESTURE_COOLDOWN = 0.5
MOVE_THRESHOLD = 35
FRAME_REDUCTION = 100
DETECTION_CONF = 0.75
TRACKING_CONF = 0.75
BRIGHTNESS_STEP = 10

# ----------------------------------------

hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=DETECTION_CONF,
                       min_tracking_confidence=TRACKING_CONF)

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 540)
cap.set(cv2.CAP_PROP_FPS, 60)

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
last_left_click = last_right_click = 0
last_gesture_time = 0
p_time = 0
paused = False

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def fingers_up(lm):
    fingers = []
    # Thumb (different for left/right hand, this assumes right hand)
    fingers.append(1 if lm[4].x < lm[3].x else 0)
    # Other fingers
    for tip_id in [8, 12, 16, 20]:
        fingers.append(1 if lm[tip_id].y < lm[tip_id - 2].y else 0)
    return fingers

print("Gesture Mouse + System Control Running...")
print("Press 'p' to pause/resume | 'q' to quit")
print("\n--- GESTURE GUIDE ---")
print("Index Only: Move cursor")
print("Index + Thumb Touch: Left Click")
print("Index + Middle Touch: Right Click")
print("Index + Middle Up: Volume (swipe left/right)")
print("Index Only Up: Brightness (swipe up/down)")
print("Three Fingers Up: Scroll (move hand up/down)")
print("----------------------\n")

prev_index_pos = (0, 0)
prev_two_finger_pos = (0, 0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    x1, y1 = FRAME_REDUCTION, FRAME_REDUCTION
    x2, y2 = w - FRAME_REDUCTION, h - FRAME_REDUCTION
    cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and not paused:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark
        finger_state = fingers_up(lm)

        thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))
        index_tip = (int(lm[8].x * w), int(lm[8].y * h))
        middle_tip = (int(lm[12].x * w), int(lm[12].y * h))
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Cursor Movement (Index Finger Always)
        ix, iy = np.clip(index_tip[0], x1, x2), np.clip(index_tip[1], y1, y2)
        target_x = np.interp(ix, (x1, x2), (0, screen_w))
        target_y = np.interp(iy, (y1, y2), (0, screen_h))
        curr_x = prev_x + (target_x - prev_x) / SMOOTHING
        curr_y = prev_y + (target_y - prev_y) / SMOOTHING
        pyautogui.moveTo(curr_x, curr_y, _pause=False)
        prev_x, prev_y = curr_x, curr_y

        # Clicks (Check these first)
        left_dist = distance(index_tip, thumb_tip)
        right_dist = distance(index_tip, middle_tip)

        if left_dist < CLICK_THRESHOLD and (time.time() - last_left_click) > CLICK_COOLDOWN:
            pyautogui.click(button='left')
            last_left_click = time.time()
            cv2.putText(frame, "Left Click", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        elif right_dist < CLICK_THRESHOLD and (time.time() - last_right_click) > CLICK_COOLDOWN:
            pyautogui.click(button='right')
            last_right_click = time.time()
            cv2.putText(frame, "Right Click", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

        # Gesture Detection (Non-clicking gestures)
        now = time.time()
        
        # VOLUME CONTROL: Index + Middle fingers UP (two fingers, not touching)
        # Swipe LEFT/RIGHT for volume
        if finger_state == [0, 1, 1, 0, 0] and left_dist > CLICK_THRESHOLD and right_dist > CLICK_THRESHOLD:
            mid_x = (index_tip[0] + middle_tip[0]) // 2
            mid_y = (index_tip[1] + middle_tip[1]) // 2
            dx = mid_x - prev_two_finger_pos[0]
            
            if abs(dx) > MOVE_THRESHOLD and (now - last_gesture_time > GESTURE_COOLDOWN):
                if dx < -MOVE_THRESHOLD:
                    pyautogui.press("volumeup")
                    cv2.putText(frame, "Volume Up", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    last_gesture_time = now
                elif dx > MOVE_THRESHOLD:
                    pyautogui.press("volumedown")
                    cv2.putText(frame, "Volume Down", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    last_gesture_time = now
            
            prev_two_finger_pos = (mid_x, mid_y)

        # BRIGHTNESS CONTROL: ONLY Index finger UP (one finger)
        # Swipe UP/DOWN for brightness
        elif finger_state == [0, 1, 0, 0, 0]:
            dy = index_tip[1] - prev_index_pos[1]
            
            if abs(dy) > MOVE_THRESHOLD and (now - last_gesture_time > GESTURE_COOLDOWN):
                if dy < -MOVE_THRESHOLD:
                    brightness = sbc.get_brightness(display=0)[0]
                    sbc.set_brightness(min(100, brightness + BRIGHTNESS_STEP))
                    cv2.putText(frame, "Brightness Up", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    last_gesture_time = now
                elif dy > MOVE_THRESHOLD:
                    brightness = sbc.get_brightness(display=0)[0]
                    sbc.set_brightness(max(0, brightness - BRIGHTNESS_STEP))
                    cv2.putText(frame, "Brightness Down", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    last_gesture_time = now

        prev_index_pos = index_tip

        # SCROLL: Three Fingers Up (Index, Middle, Ring)
        if finger_state[1] and finger_state[2] and finger_state[3]:
            if iy < h // 3:
                pyautogui.scroll(60)
                cv2.putText(frame, "Scroll Up", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif iy > 2 * h // 3:
                pyautogui.scroll(-60)
                cv2.putText(frame, "Scroll Down", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # FPS Display
    c_time = time.time()
    fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
    p_time = c_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if paused:
        cv2.putText(frame, "PAUSED (press 'p' to resume)", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, "Press 'p' to pause | 'q' to quit", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Gesture Mouse + System Control", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        paused = not paused
        time.sleep(0.25)
    if key == ord('q'):
        print("Exiting Gesture System...")
        break

cap.release()
cv2.destroyAllWindows()