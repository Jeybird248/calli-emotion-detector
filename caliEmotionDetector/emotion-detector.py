import cv2
import os
import numpy as np

emotions = ["", "sad", "happy", "neutral"]


def detect_face(img):
    # turning image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # read the face detecting cascade file
    face_cascade = cv2.CascadeClassifier('face-detector.xml')
    # run the face through the OpenCV detectMultiScale, scale factor determines how much the face
    # can vary by
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2)
    # if no faces are detected, return none for both
    if len(faces) == 0:
        return None, None
    # return the face cords and the face
    for (x, y, w, h) in faces:
        return gray[y:y + w, x:x + h], faces[0]


def prepare_training_data(data_folder_path):
    # get the directory
    dirs = os.listdir(data_folder_path)
    # create arrays to store faces and labels
    faces = []
    labels = []
    for dir_name in dirs:
        # extract the labels by taking off the s from the sLabel format
        if not dir_name.startswith("s"):
            continue
        label = int(dir_name.replace("s", ""))
        emotion_dir_path = data_folder_path + "/" + dir_name
        emotion_images_names = os.listdir(emotion_dir_path)
        for image_name in emotion_images_names:
            if image_name.startswith("."):
                continue
            image_path = emotion_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            cv2.imshow("Training on image...", cv2.resize(image, (889, 500)))
            cv2.waitKey(100)
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    label, confidence = face_recognizer.predict(face)
    label_text = emotions[label]
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1] - 5)
    return img


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

print("Predicting images...")

test_img1 = cv2.imread("test-data/1.png")
test_img2 = cv2.imread("test-data/2.png")
test_img3 = cv2.imread("test-data/3.png")
test_img4 = cv2.imread("test-data/4.png")
test_img5 = cv2.imread("test-data/5.png")

predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
predicted_img4 = predict(test_img4)
predicted_img5 = predict(test_img5)
print("Prediction complete")

cv2.imshow("Test Image 1", cv2.resize(predicted_img1, (889, 500)))
cv2.imshow("Test Image 2", cv2.resize(predicted_img2, (889, 500)))
cv2.imshow("Test Image 3", cv2.resize(predicted_img3, (889, 500)))
cv2.imshow("Test Image 4", cv2.resize(predicted_img4, (889, 500)))
cv2.imshow("Test Image 5", cv2.resize(predicted_img5, (889, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
