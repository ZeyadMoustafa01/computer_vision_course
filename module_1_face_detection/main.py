import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(image=roi_gray, scaleFactor=1.1, minNeighbors=3)
        for (eye_x, eye_y, eye_w, eye_h) in eyes:
            cv2.rectangle(img=roi_color, pt1=(eye_x, eye_y), pt2=(eye_x+eye_w, eye_y+eye_h),
                          color=(0, 255, 0), thickness=2)
        smiles = smile_cascade.detectMultiScale(image=roi_gray, scaleFactor=1.5, minNeighbors=8)
        for (smile_x, smile_y, smile_w, smile_h) in smiles:
            cv2.rectangle(img=roi_color, pt1=(smile_x, smile_y), pt2=(smile_x+smile_w, smile_y+smile_h),
                          color=(255, 0, 0), thickness=2)
    return frame


video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
    canvas = detect(gray=gray, frame=frame)
    cv2.imshow(winname='Video', mat=canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyWindow(winname='Video')