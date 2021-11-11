import cv2
out = cv2.VideoWriter('01.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280, 720))
cap = cv2.VideoCapture('test-data/test_vod2.mp4')
while True:
    ret, frame = cap.read()
    if ret:
        try:
            out.write(frame)
        except:
            continue
    else:
        break
