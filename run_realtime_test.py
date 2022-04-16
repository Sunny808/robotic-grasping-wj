import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

from hardware.camera_test import VideoCapture
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import save_results, plot_results

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str,
                        default='trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97',
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=0,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Connect to Camera
    logging.info('Connecting to camera...')
    cam = VideoCapture(0)
    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    # Load Network
    logging.info('Loading model...')
    net = torch.load(args.network,  map_location=torch.device('cpu'))
    logging.info('Done')

    # Get the compute device
    device = get_device(args.force_cpu)

    try:
        fig = plt.figure(figsize=(10, 10)) # 创建自定义图像
        while True:
            rgb = cam.read()
            x, depth_img, rgb_img = cam_data.get_data(rgb = rgb) # x是张量
            with torch.no_grad():
                xc = x.to(device)
                pred = net.predict(xc)
                q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

                plot_results(fig=fig,
                             rgb_img=cam_data.get_rgb(rgb, False),
                             grasp_q_img=q_img,
                             grasp_angle_img=ang_img,
                             depth_img = np.squeeze(depth_img),# depth_img is none.
                             no_grasps=args.n_grasps,
                             grasp_width_img=width_img)

    finally:
        rgb = cam.read()
        save_results(
            rgb_img=cam_data.get_rgb(rgb, False),
            depth_img=np.squeeze(cam_data.get_depth(None)),
            grasp_q_img=q_img,
            grasp_angle_img=ang_img,
            no_grasps=args.n_grasps,
            grasp_width_img=width_img
        )
        cam.release()
