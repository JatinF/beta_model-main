import json

import os
import copy
import cv2 
import numpy as np
import argparse

# from utils import CvFpsCalc
from face_mesh import FaceMesh
from iris_landmark import IrisLandmark
from predict_and_mesh import image_preprocessing, classify_drunk

def detect_iris(image, iris_detector, left_eye, right_eye):
    image_width, image_height = image.shape[1], image.shape[0]
    input_shape = iris_detector.get_input_shape()

    # left eye
    # crop the bbox area
    left_eye_x1 = max(left_eye[0], 0)
    left_eye_y1 = max(left_eye[1], 0)
    left_eye_x2 = min(left_eye[2], image_width)
    left_eye_y2 = min(left_eye[3], image_height)
    left_eye_image = copy.deepcopy(image[left_eye_y1:left_eye_y2,
                                         left_eye_x1:left_eye_x2])
    # detect the iris
    eye_contour, iris = iris_detector(left_eye_image)
    # calculate the iris point
    left_iris = calc_iris_point(left_eye, eye_contour, iris, input_shape)

    # right eye
    # crop the bbox area
    right_eye_x1 = max(right_eye[0], 0)
    right_eye_y1 = max(right_eye[1], 0)
    right_eye_x2 = min(right_eye[2], image_width)
    right_eye_y2 = min(right_eye[3], image_height)
    right_eye_image = copy.deepcopy(image[right_eye_y1:right_eye_y2,
                                          right_eye_x1:right_eye_x2])
    # detect the iris
    eye_contour, iris = iris_detector(right_eye_image)
    # calculate the iris point
    right_iris = calc_iris_point(right_eye, eye_contour, iris, input_shape)

    return left_iris, right_iris


def calc_iris_point(eye_bbox, eye_contour, iris, input_shape):
    iris_list = []
    for index in range(5):
        point_x = int(iris[index * 3] *
                      ((eye_bbox[2] - eye_bbox[0]) / input_shape[0]))
        point_y = int(iris[index * 3 + 1] *
                      ((eye_bbox[3] - eye_bbox[1]) / input_shape[1]))
        point_x += eye_bbox[0]
        point_y += eye_bbox[1]

        iris_list.append((point_x, point_y))

    return iris_list


def calc_min_enc_losingCircle(landmark_list):
    center, radius = cv2.minEnclosingCircle(np.array(landmark_list))
    center = (int(center[0]), int(center[1]))
    radius = int(radius)

    return center, radius

def get_sclera_color(img, iris_detector, face_mesh):
    face_results = face_mesh(img)
    if len(face_results) == 0:
        return (0, 0, 0)
    face_result = face_results[0]
    left_eye_lm, right_eye_lm = face_mesh.get_eye_landmarks(face_result)
    left_corner = np.array(left_eye_lm[0], dtype=np.int32)
    right_corner = np.array(left_eye_lm[7], dtype=np.int32)
    
    DIV = 5
    points = np.zeros((DIV-1, 2))
    left_sample_pos = np.array(np.floor((1/DIV) * left_corner + (1 - 1/DIV) * right_corner), np.int32)
    right_sample_pos = np.array(np.floor((1 - 1/DIV) * left_corner + (1/DIV) * right_corner), np.int32)
    
    color = (img[left_sample_pos[1], left_sample_pos[0], :] + img[right_sample_pos[1], right_sample_pos[0], :]) / 2
    return color


def detect_coordinate(img_path):

    # path to the image
    # img_file = img_path

    max_num_faces = 1
    min_detection_confidence = 0.7
    min_tracking_confidence = 0.7

    # open file
    image = cv2.imread(img_path)


    # mesh the face
    face_mesh = FaceMesh(
        max_num_faces,
        min_detection_confidence,
        min_tracking_confidence,
    )
    iris_detector = IrisLandmark(path = os.getcwd())


    image = cv2.flip(image, 1)  # flip as a mirror
    # debug_image = copy.deepcopy(image)

    # get the face mesh #############################################################
    # Face Mesh on possible multiple faces
    face_result = face_mesh(image)[0]

    # calculate the bounding box of eye
    left_eye, right_eye = face_mesh.calc_around_eye_bbox(face_result)

    # get the landmarks of eye
    # left_eye_lm, right_eye_lm = face_mesh.get_eye_landmarks(face_result)
    
    # detect the iris
    left_iris, right_iris = detect_iris(image, iris_detector, left_eye,
                                        right_eye)

    # calculate the iris radius
    left_center, left_radius = calc_min_enc_losingCircle(left_iris)
    right_center, right_radius = calc_min_enc_losingCircle(right_iris)

    sclera_color = get_sclera_color(image, iris_detector, face_mesh)


    return [left_center, left_radius, right_center, right_radius, sclera_color.tolist()]



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Classification")
    parser.add_argument('--img', default="test.jpg", help='image path to do the test')
    args = parser.parse_args()
    img_file = args.img

    # First drunk or not
    
    img = image_preprocessing(img_file)
    classify_result, classify_confidence = classify_drunk(img)
    print("\nresult: drunk" if np.isclose(classify_result, 1) else "\n result: sober")
    print("confidence: ", classify_confidence)

    # Second detect coordinate of eyes
    coords = detect_coordinate(img_file)
    # print(coords)

    # Write data into json

    data = {
        'user' : [
            {
                "classification_result" : classify_result.tolist(),
                "classification_confidence" : classify_confidence.tolist(),
                "eyes_location" : [coords[0], coords[2]],
                "eyes_radius" : [coords[1], coords[3]],
                "sclera_color" : [coords[4]] 
            }
        ]
    }
    with open("result.json", "w") as f:
        json.dump(data, f)
    print("Write into JSON file completed, results shown in result.json")


