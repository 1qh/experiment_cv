from collections import deque

import cv2

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
maxlen = 30
pts = deque(maxlen=maxlen)

cap = cv2.VideoCapture(0)
codec = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(6, codec)
cap.set(5, 30)
cap.set(3, 1920)
cap.set(4, 1080)

while True:
    frame = cap.read()[1]
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        _, radius = cv2.minEnclosingCircle(c)
        m = cv2.moments(c)
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        if radius > 10:
            cv2.circle(
                frame,
                center,
                int(radius),
                (255, 255, 255),
                2,
            )
            pts.appendleft(center)

    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue

        thickness = (maxlen / i) ** (1 / 2) * 2
        thickness = int(thickness)
        cv2.line(
            frame,
            pts[i - 1],
            pts[i],
            (0, 0, 255),
            thickness,
        )

    cv2.imshow(' ', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
