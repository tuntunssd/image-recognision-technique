import cv2
import mediapipe as mp
import random
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
fingerCoordinates1 = (8, 6)  # For the first finger
thumbCoordinate = (4, 2)

array_of_texts = ["Firstfinger", "Pinkyfinger", "Thumbfinger"]

# Timer variables
text_show_time = 3  # Number of seconds to show the random text
text_start_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiLandmarks = results.multi_hand_landmarks

    if multiLandmarks:
        handPoints = []
        for handLms in multiLandmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for idx, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handPoints.append((cx, cy))

        for point in handPoints:
            cv2.circle(img, point, 10, (0, 0, 255), cv2.FILLED)

        # Randomly choose a text and display it on the image for a certain time
        if time.time() - text_start_time > text_show_time:
            random_text = random.choice(array_of_texts)
            text_start_time = time.time()

        cv2.putText(img, str(random_text), (150, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Detect specific fingers
        upCount = "null"
        if handPoints[fingerCoordinates1[0]][1] < handPoints[fingerCoordinates1[1]][1]:
            upCount = "Firstfinger"
        if handPoints[20][1] < handPoints[18][1]:
            upCount = "Pinkyfinger"
        if handPoints[thumbCoordinate[0]][0] > handPoints[thumbCoordinate[1]][0]:
            upCount = "Thumbfinger"

        # Display the detected finger name


        # Check if random text matches the detected finger name
        if random_text.lower() == upCount.lower():
            cv2.putText(img, "Success!", (150, 250), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)
            success_count += 1
            if success_count == 6:
                break

        else:
            # Reset the success count if the texts don't match
            success_count = 0

    cv2.imshow("Finger Detection", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit the loop
        break

cap.release()
cv2.destroyAllWindows()