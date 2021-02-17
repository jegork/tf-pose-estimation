import argparse
import logging
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    
    out_file = (args.video).split('.')[-2]
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    cap = cv2.VideoCapture(args.video)
    width = cap.get(3)
    height = cap.get(4)
    out = cv2.VideoWriter(out_file + '_out.avi', fourcc, 20.0, (int(width), int(height)))
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(int(width), int(height)))


    val = False
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    while cap.isOpened():
        ret_val, image = cap.read()

        if image is None:
            break

        humans = e.inference(image, upsample_size=4.0)
        if not args.showBG:
            image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out.write(image)

        if val == False:
            logger.debug(humans)
            plt.imshow(image)
            plt.savefig('img.png')
            val = True
            
        fps_time = time.time()
            
logger.debug('finished+')
