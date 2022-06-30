# Requirements

* See `requirements.txt`

* For convenience, use `pip install -r requirements.txt` should be good

# Usage

* `python main.py` for quick test. it will use the default `test.jpg`.
  * Output contains
    1. Classification result indicating whether or not this person is drunk
    1. Classification confidence
    1. `result.json`, containing classificaiton results, and the coordinates of two eyes, radius of two eyes, and the RGB value of sclera color

* To test your own image, `python main,py --img=[image_path]`
* Sorry not offering other hyperparameter tuning, I'll add if asked.

# Others

* Pretrained models are in `my_sobr.h5` and `iris_landmark.tflite`, pls make sure the existence of two models
* No restriction of image size for the demo testing. 
* The function to add face mesh onto the original image is there, just not activated.# beta_model-main
