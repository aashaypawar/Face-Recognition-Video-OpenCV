import cv2, time

video = cv2.VideoCapture(0)
f_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    check, frame = video.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=5)
    for x, y, w, h in faces:
        gray_image = cv2.rectangle(gray_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Capturing",gray_image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


video.release()
cv2.destroyAllWindows()