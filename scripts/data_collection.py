import cv2
import pyautogui
import mss
import numpy as np
import time
import os
from PIL import Image
import sys
import signal
import atexit
import pydirectinput as pdi

EMULATOR_WINDOW = {'top': 43, 'left': 630, 'width': 659, 'height': 494}
ROI_SPARKLE = (299, 65, 357, 225)
BRIGHTNESS_THRESHOLD = 255
BASE_FOLDER_TRAIN = 'data/train/'
BASE_FOLDER_VAL = 'data/val/'
FRAMES_PER_BATCH = 60
CAPTURE_FPS = 60
TARGET_SHINY_BATCHES = 100
TARGET_NORMAL_BATCHES = 100

os.makedirs(BASE_FOLDER_TRAIN + 'sparkle', exist_ok=True)
os.makedirs(BASE_FOLDER_TRAIN + 'normal', exist_ok=True)
os.makedirs(BASE_FOLDER_VAL + 'sparkle', exist_ok=True)
os.makedirs(BASE_FOLDER_VAL + 'normal', exist_ok=True)

shiny_batch_count = 0
normal_batch_count = 0
current_mode = 'shiny'
running = True
shift_is_held = False

sct = mss.mss()

def release_all_keys():
    keys_to_release = ['up', 'down', 'left', 'right', 'space', 'z', 'x', 
                       'enter', 'shiftright', 'shift', 'shiftleft']
    
    print("\n Releasing all keys (3 attempts)...")
    for attempt in range(3):
        for key in keys_to_release:
            try:
                pdi.keyUp(key)
            except:
                pass
        time.sleep(0.1)
    
    print("All keys released")

def cleanup_and_exit(signum=None, frame=None):
    global running, sct, shift_is_held
    running = False
    print("\n\n=== STOPPING SCRIPT ===")
    release_all_keys()
    shift_is_held = False
    try:
        sct.close()
    except:
        pass
    print(f"\n Summary:")
    print(f"Shiny batches: {shiny_batch_count}/{TARGET_SHINY_BATCHES}")
    print(f"Normal batches: {normal_batch_count}/{TARGET_NORMAL_BATCHES}")
    print(f"Total: {shiny_batch_count + normal_batch_count}")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)
atexit.register(release_all_keys)

def capture_screen():
    screenshot = sct.grab(EMULATOR_WINDOW)
    img = np.array(screenshot)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def is_white_fade(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness == BRIGHTNESS_THRESHOLD

def is_sustained_white():
    white_count = 0
    required_frames = 4
    
    for _ in range(required_frames):
        frame = capture_screen()
        if is_white_fade(frame):
            white_count += 1
        else:
            return False
    
    return white_count >= required_frames

def save_batch(frames, is_shiny):
    global shiny_batch_count, normal_batch_count
    
    folder = (BASE_FOLDER_TRAIN if np.random.rand() < 0.8 else BASE_FOLDER_VAL)
    folder += 'sparkle' if is_shiny else 'normal'
    
    batch_num = shiny_batch_count if is_shiny else normal_batch_count
    batch_folder = os.path.join(folder, f'batch_{batch_num:05d}')
    os.makedirs(batch_folder, exist_ok=True)
    
    for i, frame in enumerate(frames):
        crop = frame[ROI_SPARKLE[1]:ROI_SPARKLE[1]+ROI_SPARKLE[3], 
                     ROI_SPARKLE[0]:ROI_SPARKLE[0]+ROI_SPARKLE[2]]
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img.save(os.path.join(batch_folder, f'frame_{i:03d}.png'))
    
    if is_shiny:
        shiny_batch_count += 1
    else:
        normal_batch_count += 1

def move_in_grass():
    global shift_is_held
    if not running:
        return

    if current_mode == 'shiny' and not shift_is_held:
        pdi.keyDown('shiftright')
        shift_is_held = True
        time.sleep(0.1)
    elif current_mode == 'normal' and shift_is_held:
        pdi.keyUp('shiftright')
        shift_is_held = False
        time.sleep(0.1)
    
    directions = ['left', 'right']
    direction = np.random.choice(directions) 
    pdi.keyDown(direction)
    pdi.keyDown('z')
    
    sleep_time = 0.5 + np.random.uniform(-0.2, 0.2)
    end_time = time.time() + sleep_time
    while time.time() < end_time and running:
        time.sleep(0.05)
    
    pdi.keyUp(direction)

def flee_battle():
    if not running:
        return
    time.sleep(11)
    pdi.press('down')
    pdi.press('down')
    pdi.press('right')
    pdi.press('x')
    time.sleep(3)

def switch_mode():
    global current_mode, shift_is_held
    
    print("SWITCHING MODE")
    
    release_all_keys()
    shift_is_held = False
    time.sleep(0.5)
    
    if current_mode == 'shiny':
        current_mode = 'normal'
        print("Switched to: NORMAL MODE")
    else:
        current_mode = 'shiny'
        print("Switched to: SHINY MODE")
    
    for i in range(5, 0, -1):
        print(f"Resuming in {i}...")
        time.sleep(1)
    
    print("\n Resuming now\n")

print("Starting in 3 seconds...")
time.sleep(3)

pdi.click(x=EMULATOR_WINDOW['left'] + 50, y=EMULATOR_WINDOW['top'] + 50)
time.sleep(0.5)

try:
    while running:
        if current_mode == 'shiny' and shiny_batch_count >= TARGET_SHINY_BATCHES:
            switch_mode()
            continue
        
        if current_mode == 'normal' and normal_batch_count >= TARGET_NORMAL_BATCHES:
            print("\n All batches collected")
            break
        
        move_in_grass()
        
        if not running:
            break
        
        frame = capture_screen()
        if is_white_fade(frame):
            if is_sustained_white():
                pdi.keyUp('z')
                
                is_shiny = (current_mode == 'shiny')
                mode_str = "SHINY" if is_shiny else "NORMAL"
                current_count = shiny_batch_count if is_shiny else normal_batch_count
                target_count = TARGET_SHINY_BATCHES if is_shiny else TARGET_NORMAL_BATCHES
                
                print(f"\n[{mode_str} {current_count}/{target_count}] Encounter detected")
                
                print("Waiting for encounter to start...")
                time.sleep(2.4) 
                
                print(f"Capturing {FRAMES_PER_BATCH} frames...")
                frames = []
                frame_delay = 1.0 / CAPTURE_FPS
                
                for i in range(FRAMES_PER_BATCH):
                    if not running:
                        break
                    
                    frame = capture_screen()
                    frames.append(frame)
                    time.sleep(frame_delay)
                    
                    if (i + 1) % 10 == 0:
                        print(f"Captured {i + 1}/{FRAMES_PER_BATCH} frames...")
                
                if running and len(frames) == FRAMES_PER_BATCH:
                    save_batch(frames, is_shiny)
                    new_count = shiny_batch_count if is_shiny else normal_batch_count
                    print(f"Saved {mode_str} batch {new_count - 1} ({len(frames)} frames)")
                    flee_battle()
                else:
                    print(f"Incomplete batch, skipping...")

except Exception as e:
    print(f"\nError occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    cleanup_and_exit()

print("\n Collection complete!")