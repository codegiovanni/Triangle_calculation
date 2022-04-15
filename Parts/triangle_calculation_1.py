import cv2
import numpy as np

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

color_search = np.zeros((200, 200, 3), np.uint8)
color_selected = np.zeros((200, 200, 3), np.uint8)


def select_color(event, x, y, flags, param):
    B = frame[y, x][0]
    G = frame[y, x][1]
    R = frame[y, x][2]
    color_search[:] = (B, G, R)

    if event == cv2.EVENT_LBUTTONDOWN:
        color_selected[:] = (B, G, R)


cv2.namedWindow('image')
cv2.setMouseCallback('image', select_color)

while True:

    _, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    cv2.imshow('image', frame)
    cv2.imshow('color_search', color_search)
    cv2.imshow('color_selected', color_selected)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
