import cv2
import mss
import numpy as np
import time

time.sleep(5)  # 5s to focus emulator and pause on encounter
with mss.mss() as sct:
    screenshot = sct.grab({'top': 43, 'left': 630, 'width': 658, 'height': 494})  # Use YOUR EMULATOR_WINDOW values here
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    cv2.imwrite("full_encounter_screenshot.png", img)