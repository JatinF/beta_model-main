# Intro

> This repo is a complete prediction method, with no training involved. 

This repo contains all models we currently have for beta test, including 
1. Sobr model used for classification of drunkness
1. Iris detection model used for localization of Iris and pupil
1. Face mesh model used for generating the mesh structure of human face

# Dependency

* See `requirements.txt`
* For convenience, use `pip install -r requirements.txt`

# Structure

* `face_mesh.py`: used to generate the face mesh and eye localization
* `iris_landmark.py` & `iris_landmark.tflite`: used for iris and pupil localization, provides key parameters of eyes
* `my_sobr.h5`: the machine learning model to do the classificaiton
* `predict_and_mesh.py`: utility functions
* `result.json`: used to store all the information predicted by our model, including
    1. userid
    1. classification result: drunk(1)/sober(0)
    1. classification confidence: 0->1
    1. eyes_location: two 2-dimension coordinates representing the position of both left and right pupil centers. measured in pixels
    1. eyes_radius: two numbers representing the radius of both left and right iris. measured in pixels
    1. sclera_color: a 3-dimension vector representing RGB values of sclera color

# Usage

* `python main.py` for quick test, it will use the default input `test.jpg`
* Output contains
    1. Classification result indication whether or not this person is drunk, shown on console
    1. Classification confidence measured from 0 to 1, shown on console
    1. `result.json`, output file containing the classificaiton result, confidence and parameters mentioned above.
* To test with your own image, specify the image file path like this
    ```shell
    python main.py --img=[image_path]
    ```

# Hyperparameter

* `DIV` in `main.py`, used to decide the sampling points of sclera color. In this case, we first get the left point $p1$ and right point $p2$ of eye contour, then we sample two points $s1$ and $s2$ using following equation

    $$s1 = (1/\text{DIV}) * p1 + (1-1/\text{DIV})*p2$$
    $$s2 = (1- 1/\text{DIV}) * p1 + (1/\text{DIV})*p2$$
    
    Specifically, when `DIV=2`, the two sampling points are overlapped, which is the midpoint of $p1$ and $p2$, a point close to the center of pupil normally. 

    To get the sclera color, we use the mean RGB value of two sampling points.

    In our algorithm, it is set to be `5`, to avoid sampling on iris area.

* `max_num_faces` & `min_detection_confidence` & `min_tracking_confidence` in `main.py`: used by face mesh model to detect the face on image. Set to be `1`, `0.7`, `0.7` as recommended by Mediapipe since what we mainly deal with are single-face images
* `dsize` in `predict_and_mesh.py`: used by opencv to resize our images to fit in the input requirement of our model. The default value is `[106,106]`. If changed, the model need to be trained from scratch.
    > `dsize` is not the size of input image. The size of input image is arbitrary. The image will be gray-scaled, cropped and resized after being fed. More details can be seen in func `image_preprocessing` in `predict_and_mesh.py`

# Others

* The development of API is on different branch
* The face mesh function is available but not activated. To activate it, we can use line 96-104 in `predict_and_mesh.py`, add them to `main.py` and everything is good to go.

