import cv2

#Reconocimiento facial en camaras web

cap = cv2.VideoCapture(0)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,1.3,5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("frame",frame)

    if cv2.waitKey(1) and 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
