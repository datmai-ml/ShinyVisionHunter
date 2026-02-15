import pyautogui
import time

print("Focus on emu window")
time.sleep(5)

print("Hover over top left corner of emu window")
time.sleep(3)
top_left = pyautogui.position()
print(f"Top-left: {top_left}")

print("Hover over bottom right corner of emu window")
time.sleep(3)
bottom_right = pyautogui.position()
print(f"Bottom-right: {bottom_right}")

width = bottom_right.x - top_left.x
height = bottom_right.y - top_left.y
print(f"Estimated size: width={width}, height={height}")
print(f"Update script: monitor = {{'top': {top_left.y}, 'left': {top_left.x}, 'width': {width}, 'height': {height}}}")