import cv2
import mediapipe
import pygame
import math
import random

# Initialize pygame and mediapipe
pygame.init()
WIDTH, HEIGHT = 900, 750
win = pygame.display.set_mode((WIDTH, HEIGHT))

cap = cv2.VideoCapture(0)
mpHands = mediapipe.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mediapipe.solutions.drawing_utils

# Finger class to manage geometric shapes along each finger's movement
class Finger:
    def __init__(self):
        self.positions = []  # Track past positions for the pattern trail

    def update_position(self, x, y):
        self.positions.append((x, y))
        if len(self.positions) > 15:
            self.positions.pop(0)  # Limit trail length

    def draw(self):
        for i, pos in enumerate(self.positions):
            x, y = pos
            size = 10 + i * 1.5  # Increase size along the trail
            rotation_angle = pygame.time.get_ticks() / 10 + i * 15  # Rotating effect
            
            # Create a rotating and pulsing shape at each position
            draw_rotating_shape(x, y, size, rotation_angle, i)

# Get fingertip positions and hand center
def get_finger_positions_and_center(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        finger_positions = {}
        palm_x, palm_y = 0, 0
        palm_count = 0

        for hand in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
            height, width, _ = img.shape
            fingertip_ids = [4, 8, 12, 16, 20]
            palm_ids = [0, 5, 9, 13, 17]

            for id, landmark in enumerate(hand.landmark):
                x, y = int(landmark.x * width), int(landmark.y * height)
                if id in fingertip_ids:
                    finger_positions[id] = (x, y)
                if id in palm_ids:
                    palm_x += x
                    palm_y += y
                    palm_count += 1

            if palm_count > 0:
                center_x, center_y = palm_x // palm_count, palm_y // palm_count
                return finger_positions, (center_x, center_y)
    return None, None

# Draw rotating and pulsing shapes along each fingertip trail
def draw_rotating_shape(x, y, size, angle, trail_index):
    # Choose colors dynamically based on the time to create a vibrant effect
    color = (
        int(255 * abs(math.sin(pygame.time.get_ticks() / 500 + trail_index))),
        int(255 * abs(math.sin(pygame.time.get_ticks() / 700 + trail_index))),
        int(255 * abs(math.sin(pygame.time.get_ticks() / 900 + trail_index)))
    )

    # Pulsing effect
    pulse_size = size + int(10 * math.sin(pygame.time.get_ticks() / 200 + trail_index))

    # Draw a rotating shape (circle or triangle)
    if trail_index % 3 == 0:
        # Draw a rotating circle
        pygame.draw.circle(win, color, (x, y), pulse_size, 2)
    elif trail_index % 3 == 1:
        # Draw a rotating triangle
        draw_rotating_triangle(x, y, pulse_size, angle)
    else:
        # Draw a rotating square
        draw_rotating_square(x, y, pulse_size, angle)

# Function to draw a rotating triangle
def draw_rotating_triangle(x, y, size, angle):
    half_size = size / 2
    points = [
        (x + half_size * math.cos(angle), y + half_size * math.sin(angle)),
        (x + half_size * math.cos(angle + 2.094), y + half_size * math.sin(angle + 2.094)),
        (x + half_size * math.cos(angle + 4.188), y + half_size * math.sin(angle + 4.188))
    ]
    pygame.draw.polygon(win, (255, 255, 255), points, 2)

# Function to draw a rotating square
def draw_rotating_square(x, y, size, angle):
    square_surface = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.rect(square_surface, (255, 255, 255), (0, 0, size, size), 2)
    rotated_square = pygame.transform.rotate(square_surface, angle)
    rect = rotated_square.get_rect(center=(x, y))
    win.blit(rotated_square, rect)

# Draw rotating and pulsing pattern at the palm center
def draw_center_animation(center):
    center_x, center_y = center
    pygame.draw.circle(win, (255, 255, 255), (center_x, center_y), 30, 1)

    for i in range(6):
        size = 30 + int(20 * math.sin(pygame.time.get_ticks() / 300 + i))
        angle = math.radians(i * 60 + pygame.time.get_ticks() / 10)  # Rotating effect
        x = center_x + int(size * math.cos(angle))
        y = center_y + int(size * math.sin(angle))
        pygame.draw.rect(win, (100 + i * 20, 100 + i * 20, 255 - i * 30), (x - size // 4, y - size // 4, size, size), 1)

# Update the fingers' positions and draw geometric patterns along each finger
def update_fingers(fingers, finger_positions, hand_center, img):
    win.fill((0, 0, 0))

    if finger_positions:
        fingertip_ids = [4, 8, 12, 16, 20]
        for i, tip_id in enumerate(fingertip_ids):
            if tip_id in finger_positions:
                target_x, target_y = finger_positions[tip_id]
                fingers[i].update_position(target_x, target_y)

    # Draw each finger with rotating and pulsing geometric shapes
    for finger in fingers:
        finger.draw()

    # Draw animated geometric shape at hand center
    if hand_center:
        draw_center_animation(hand_center)

    pygame.display.flip()
    cv2.imshow("Image", cv2.flip(img, 1))

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

        finger_positions, hand_center = get_finger_positions_and_center(img)
        update_fingers(fingers, finger_positions, hand_center, img)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()
