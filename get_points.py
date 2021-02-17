import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

    
def main(image, model='mobilenet_thin', resize='432x368'):
    w, h = model_wh(resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(image, None, None)
    if image is None:
        sys.exit(-1)

    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)

    points = TfPoseEstimator.get_points(image, humans)
    
    return points