import cv2
import numpy as np

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

color_search = np.zeros((200, 200, 3), np.uint8)
color_selected = np.zeros((200, 200, 3), np.uint8)

hue = 0


def select_color(event, x, y, flags, param):
    global hue

    B = frame[y, x][0]
    G = frame[y, x][1]
    R = frame[y, x][2]
    color_search[:] = (B, G, R)

    if event == cv2.EVENT_LBUTTONDOWN:
        color_selected[:] = (B, G, R)
        hue = hsv[y, x][0]


def search_contours(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 1000 < area < 10000:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                c_x = int(moments["m10"] / moments["m00"])
                c_y = int(moments["m01"] / moments["m00"])
            else:
                c_x, c_y = 0, 0
            cv2.circle(frame, (c_x, c_y), 5, (255, 255, 255), -1)


cv2.namedWindow('image')
cv2.setMouseCallback('image', select_color)

while True:

    _, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([hue - 5, 50, 100])
    upper_hsv = np.array([hue + 5, 255, 255])

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    search_contours(mask)

    # cv2.imshow('mask', mask)
    cv2.imshow('image', frame)
    cv2.imshow('color_search', color_search)
    cv2.imshow('color_selected', color_selected)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
