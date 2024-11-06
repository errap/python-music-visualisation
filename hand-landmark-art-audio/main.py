# ========================================================
# Author: Erra P.
# GitHub: https://github.com/errap
# Date Created: November 6, 2024
#
# Special Thanks:
# - Special thanks to Luka Jovanovic (https://github.com/nelimalu/python/tree/main/hand-detection) for the
#   inspiration and code direction.
# ========================================================

import cv2
import mediapipe
import pygame
import math
import time

# Initialize pygame and mediapipe
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()
WIDTH, HEIGHT = 1100, 800
win = pygame.display.set_mode((WIDTH, HEIGHT))

# Load sounds
kick_sound = pygame.mixer.Sound("bass.wav")
hihat_sound = pygame.mixer.Sound("atmosphere.wav")
snare_sound = pygame.mixer.Sound("hithat.wav")
atmosphere_sound = pygame.mixer.Sound("beep_loop.wav")
synth_sound = pygame.mixer.Sound("synth.wav")

cap = cv2.VideoCapture(0)
mpHands = mediapipe.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mediapipe.solutions.drawing_utils

# Sounds for each finger
sounds = [kick_sound, hihat_sound, snare_sound, atmosphere_sound, synth_sound]

# Cooldown to prevent repeated triggers
last_bent_states = [False] * 5  # Track previous bent state of each finger

# Finger class for managing tracers
class Finger:
    def __init__(self):
        self.positions = []  # Track past positions for the pattern trail

    def update_position(self, positions):
        self.positions = positions

    def draw(self, color):
        # Draw small circles along the finger's length for tracers
        for i, (x, y) in enumerate(self.positions):
            pygame.draw.circle(win, color, (x, y), 5)

# Get fingertip and joint positions
def get_finger_positions_and_center(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        finger_positions = {}
        finger_segments = {}
        palm_x, palm_y = 0, 0
        palm_count = 0

        for hand in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
            height, width, _ = img.shape
            fingertip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
            all_joint_ids = [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
            palm_ids = [0, 5, 9, 13, 17]

            for id, landmark in enumerate(hand.landmark):
                x, y = int(landmark.x * width), int(landmark.y * height)
                if id in fingertip_ids:
                    finger_positions[id] = (x, y)
                if id in palm_ids:
                    palm_x += x
                    palm_y += y
                    palm_count += 1

            # Capture all joint positions for each finger
            for i, joint_ids in enumerate(all_joint_ids):
                finger_segments[i] = [(int(hand.landmark[j].x * width), int(hand.landmark[j].y * height)) for j in joint_ids]

            if palm_count > 0:
                center_x, center_y = palm_x // palm_count, palm_y // palm_count
                return finger_positions, finger_segments, (center_x, center_y)
    return None, None, None

# Animated color effect at the palm center
def draw_smokey_effect(center):
    center_x, center_y = center
    base_radius = 20 + int(5 * math.sin(pygame.time.get_ticks() / 300))
    color = (40, 80, 60, 50)  # Dark green-blue effect color for center animation
    pygame.draw.circle(win, color, (center_x, center_y), base_radius, 1)

# Draw a geometric eye effect at the palm center
def draw_geometric_eye(center):
    center_x, center_y = center

    # Colors for the geometric eye: light blue iris, dark green outline
    iris_color = (100, 150, 255)
    outline_color = (40, 80, 60)

    # Draw the iris (a small circle in the center)
    pygame.draw.circle(win, iris_color, (center_x, center_y), 10)

    # Draw the eye outline (ellipse around the iris)
    eye_width, eye_height = 50, 20
    pygame.draw.ellipse(win, outline_color, (center_x - eye_width // 2, center_y - eye_height // 2, eye_width, eye_height), 2)

    # Draw radial lines to create a geometric effect
    for angle in range(0, 360, 45):  # 8 lines around the iris
        radian_angle = math.radians(angle)
        line_length = 30
        end_x = center_x + int(line_length * math.cos(radian_angle))
        end_y = center_y + int(line_length * math.sin(radian_angle))
        pygame.draw.line(win, outline_color, (center_x, center_y), (end_x, end_y), 1)


# Update fingers' tracers and trigger sounds based on bending
def update_fingers(fingers, finger_positions, finger_segments, hand_center, img):
    win.fill((0, 0, 0))

    if finger_positions:
        fingertip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        mid_joint_ids = [3, 6, 10, 14, 18]  # Middle joints for bend detection
        base_joint_ids = [0, 5, 9, 13, 17]

        for i, tip_id in enumerate(fingertip_ids):
            if tip_id in finger_positions:
                fingers[i].update_position(finger_segments[i])
                target_x, target_y = finger_positions[tip_id]

                # Get joint positions for bend calculation
                base_x, base_y = finger_segments[i][0]
                mid_x, mid_y = finger_segments[i][2]

                # Calculate angle for bend detection
                angle1 = math.atan2(target_y - mid_y, target_x - mid_x)
                angle2 = math.atan2(mid_y - base_y, mid_x - base_x)

                # Adjust threshold for ring and pinky fingers
                if i in [3, 4]:  # Ring and pinky fingers
                    is_bent = abs(angle1 - angle2) > math.pi / 8  # Slightly more sensitive
                else:
                    is_bent = abs(angle1 - angle2) > math.pi / 6

                # Play sound on initial bend, stop on unbend
                if is_bent and not last_bent_states[i]:  # Transition from unbent to bent
                    sounds[i].play()
                elif not is_bent and last_bent_states[i]:  # Transition from bent to unbent
                    sounds[i].stop()

                # Update the last bent state
                last_bent_states[i] = is_bent

    # Draw finger tracers with specified colors
    finger_colors = [
        (100, 150, 255),  # Thumb: Blue
        (139, 0, 0),      # Index: Dark Red
        (40, 80, 60),     # Middle: Dark Green
        (40, 80, 60),     # Ring: Dark Green
        (169, 169, 169)   # Pinky: Dark White
    ]
    
    for i, finger in enumerate(fingers):
        finger.draw(finger_colors[i])

    if hand_center:
        draw_smokey_effect(hand_center)

    if hand_center:
        draw_geometric_eye(hand_center)  # Use the new geometric eye function

    pygame.display.flip()
    cv2.imshow("Image", cv2.flip(img, 1))

# Main loop
def main():
    fingers = [Finger() for _ in range(5)]
    run = True
    while run:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        finger_positions, finger_segments, hand_center = get_finger_positions_and_center(img)
        update_fingers(fingers, finger_positions, finger_segments, hand_center, img)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()
