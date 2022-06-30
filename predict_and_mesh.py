from PIL import Image
import cv2
from cv2 import data
import numpy as np
import json

import mediapipe as mp
from tensorflow import keras


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh



def image_preprocessing(filename):
    # Reading sample image
    img = np.array(Image.open(filename))

    # crop the face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.01, 10)
    if len(faces) == 0:
        print("No face detected")
        return
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    
    # Resize
    inp = cv2.resize(face, dsize=[106, 106], interpolation=cv2.INTER_AREA)

    # Normalize
    inp = inp / 255.0 - 0.5
    inp = np.expand_dims(inp, 0)

    return inp

def classify_drunk(img):
    model = keras.models.load_model("my_sobr.h5")
    # predict_classes been deprecated after 2021-1-1
    # result = model.predict_classes(img)
    result = model.predict(img)
    # result = np.argmax(model.predict(img), axis=-1)
    class_result = np.argmax(result, axis=-1)
    class_confidence = np.max(result/np.sum(result), axis=-1)
    return class_result, class_confidence

def get_mesh_points(filename):
    #For static images:
    
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
        
        image = cv2.imread(filename)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            print("no face detected")
            return 
        annotated_image = image.copy()
        # for face_landmarks in results.multi_face_landmarks:
        #     # print('face_landmarks:', face_landmarks)
        face_landmarks = results.multi_face_landmarks[0]
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
        # cv2.imwrite('annotated_'+filename, annotated_image)
        
        landmarks = face_landmarks.landmark

        face_mesh = []
        for datapoints in landmarks:
            face_mesh.append((datapoints.x, datapoints.y))
        face_mesh = np.array(face_mesh)
        return face_mesh


if __name__ == "__main__":

    # Classification
    img_file = "test.jpg"
    img = image_preprocessing(img_file)
    classify_result, classify_confidence = classify_drunk(img)
    print("drunk" if np.isclose(classify_result, 1) else "sober")
    print("confidence is ", classify_confidence)

    # Mesh
    # mesh_points = get_mesh_points(img_file)
    # print(mesh_points.shape)

    # write mesh data into json
    # Reference: https://pynative.com/python-serialize-numpy-ndarray-into-json/
    # with open("Mesh.json", 'w') as f:
    #     json.dump(mesh_points.tolist(), f)
    # print("Write into JSON file completed")
    