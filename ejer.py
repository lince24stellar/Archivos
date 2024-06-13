import cv2
import numpy as np


delay_ms = 50 



min_contour_width = 40  
min_contour_height = 40  
offset = 10  
line_height = 400  
matches = []

cap = cv2.VideoCapture('coche1.mp4')

if not cap.isOpened():
    print("Error al abrir el video")
    exit()

while True:
    ret, frame1 = cap.read()
    if not ret:
        break
    ret, frame2 = cap.read()
    if not ret:
        break

    d = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)
        if not contour_valid:
            continue

        cv2.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)
        cv2.line(frame1, (0, line_height), (frame1.shape[1], line_height), (0, 255, 0), 2)
        centroid = (x + int(w / 2), y + int(h / 2))
        matches.append(centroid)
        cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)



    for cx, cy in matches:
        if cy < (line_height + offset) and cy > (line_height - offset):
            matches.remove((cx, cy))

    cv2.putText(frame1, "Total Cars Detected: " + str(len(matches)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)

    cv2.imshow("Vehicle Detection", frame1)
    if cv2.waitKey(delay_ms) == 27: 
        
        
        break

# Imprimir el número total de coches detectados al final del programa

print("Total Cars Detected:", len(matches))

cap.release()
cv2.destroyAllWindows()
