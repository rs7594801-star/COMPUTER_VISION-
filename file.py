import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmark IDs
TIP_IDS = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip image for natural 'selfie' view and convert to RGB
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(rgb_image)
    
    fingers = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the image
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmark coordinates into a list
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            if len(lm_list) != 0:
                # Thumb: Check if tip is to the left/right of the knuckle (depends on hand)
                if lm_list[TIP_IDS[0]][1] > lm_list[TIP_IDS[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other 4 Fingers: Check if tip is above the knuckle (lower y value means higher on screen)
                for id in range(1, 5):
                    if lm_list[TIP_IDS[id]][2] < lm_list[TIP_IDS[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Display the count of fingers
                total_fingers = fingers.count(1)
                cv2.putText(image, f'Fingers: {total_fingers}', (45, 375), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    cv2.imshow("Hand Gesture Recognition", image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()