import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

font = cv2.FONT_HERSHEY_SIMPLEX

color_search = np.zeros((200, 200, 3), np.uint8)
color_selected = np.zeros((200, 200, 3), np.uint8)

hue = 0
circles = []


def select_color(event, x, y, flags, param):
    global hue

    B = frame[y, x][0]
    G = frame[y, x][1]
    R = frame[y, x][2]
    color_search[:] = (B, G, R)

    if event == cv2.EVENT_LBUTTONDOWN:
        color_selected[:] = (B, G, R)
        hue = hsv[y, x][0]


def length_square(x, y):
    x_diff = x[0] - y[0]
    y_diff = x[1] - y[1]
    return x_diff * x_diff + y_diff * y_diff


def connect_circles(count):
    lines = []
    for p0 in range(count - 1):
        for p1 in range(p0 + 1, count):
            lines.append([circles[p0][:2], circles[p1][:2]])
    return lines


def middle_calculator(x, y):
    x_middle = int((x[0] + y[0]) / 2)
    y_middle = int((x[1] + y[1]) / 2)
    return x_middle, y_middle


def triangle_calculation(A, B, C):
    a2 = length_square(B, C)
    b2 = length_square(A, C)
    c2 = length_square(A, B)

    a = math.sqrt(a2)
    b = math.sqrt(b2)
    c = math.sqrt(c2)

    alpha = math.acos((b2 + c2 - a2) / (2 * b * c))
    betta = math.acos((a2 + c2 - b2) / (2 * a * c))
    gamma = math.acos((a2 + b2 - c2) / (2 * a * b))

    alpha = alpha * 180 / math.pi
    betta = betta * 180 / math.pi
    gamma = gamma * 180 / math.pi

    AB_m = middle_calculator(A, B)
    BC_m = middle_calculator(B, C)
    CA_m = middle_calculator(C, A)

    # distances
    cv2.putText(frame, "{:.0f}".format(a), BC_m, font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "{:.0f}".format(b), CA_m, font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "{:.0f}".format(c), AB_m, font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    # angles
    cv2.putText(frame, "{:.0f}".format(alpha), (A[0] - 10, A[1] + 30), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "{:.0f}".format(betta), (B[0] - 10, B[1] + 30), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "{:.0f}".format(gamma), (C[0] - 10, C[1] + 30), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

    # corners
    cv2.putText(frame, "A", (A[0] - 10, A[1] - 10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "B", (B[0] - 10, B[1] - 10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "C", (C[0] - 10, C[1] - 10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

def search_contours(mask):
    contours_count = 0
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 1000 < area < 10000:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            contours_count += 1

            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                c_x = int(moments["m10"] / moments["m00"])
                c_y = int(moments["m01"] / moments["m00"])
            else:
                c_x, c_y = 0, 0

            circles.append((c_x, c_y))

            cv2.circle(frame, (c_x, c_y), 5, (255, 255, 255), -1)

    return contours_count


cv2.namedWindow('image')
cv2.setMouseCallback('image', select_color)

while True:

    _, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([hue - 5, 50, 100])
    upper_hsv = np.array([hue + 5, 255, 255])

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    count = search_contours(mask)

    for point in connect_circles(count):
        if count == 3:
            triangle_calculation(circles[0], circles[1], circles[2])
            cv2.line(frame, point[0], point[1], (255, 255, 255), 2)
        else:
            pass

    circles[:] = []

    # cv2.imshow('mask', mask)
    cv2.imshow('image', frame)
    cv2.imshow('color_search', color_search)
    cv2.imshow('color_selected', color_selected)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
