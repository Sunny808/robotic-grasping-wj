import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results, save_results
from utils.dataset_processing import evaluation
import os
import glob
from utils.dataset_processing import grasp

logging.basicConfig(level = logging.INFO)


def parse_args():
	parser = argparse.ArgumentParser(description = 'Evaluate network')
	parser.add_argument('--network', type = str, default = 'logs/220116_2146_training_cornell/epoch_30_iou_0.99',
	                    help = 'Path to saved network to evaluate')
	parser.add_argument('--use-depth', type = int, default = 0,  # 修改
	                    help = 'Use Depth image for evaluation (1/0)')
	parser.add_argument('--use-rgb', type = int, default = 1,
	                    help = 'Use RGB image for evaluation (1/0)')
	parser.add_argument('--n-grasps', type = int, default = 1,
	                    help = 'Number of grasps to consider per image')
	parser.add_argument('--cpu', dest = 'force_cpu', action = 'store_true', default = True,  # 修改
	                    help = 'Force code to run in CPU mode')

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()

	# Load Network
	logging.info('Loading model...')
	net = torch.load(args.network, map_location = torch.device('cpu'))
	logging.info('Done')

	# Get the compute device
	device = get_device(args.force_cpu)

	grasp_files = glob.glob(os.path.join('cornell', '*', 'pcd*cpos.txt'))
	depth_files = [f.replace('cpos.txt', 'd.tiff') for f in grasp_files]
	rgb_files = [f.replace('d.tiff', 'r.png') for f in depth_files]
	length = len(grasp_files)
	split = int(0.9 * length)

	with torch.no_grad():
		for value in range(split, length):
			img_data = CameraData(include_depth = args.use_depth, include_rgb = args.use_rgb, path = grasp_files[value])
			pic = Image.open(rgb_files[value], 'r')
			rgb = np.array(pic)
			pic = Image.open(depth_files[value], 'r')
			depth = np.expand_dims(np.array(pic), axis = 2)
			gtbbs = grasp.GraspRectangles.load_from_cornell_file(grasp_files[value])
			center = gtbbs.center  # return 1X2 array
			left = max(0, min(center[1] - 224 // 2, 640 - 224))
			top = max(0, min(center[0] - 224 // 2, 480 - 224))
			gtbbs.rotate(0, center)
			gtbbs.offset((-top, -left))
			gtbbs.zoom(1, (224 // 2, 224 // 2))
			x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)
			xc = x.to(device)
			pred = net.predict(xc)

			q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

			s = evaluation.calculate_iou_match(q_img,
			                                   ang_img,
			                                   gtbbs,
			                                   no_grasps = 1,
			                                   grasp_width = width_img,
			                                   threshold = 0.25
			                                   )
			print(s, value)

			if not s:
				fig = plt.figure(figsize = (10, 10))
				plot_results(fig = fig,
				             rgb_img = img_data.get_rgb(rgb, False),
				             grasp_q_img = q_img,
				             grasp_angle_img = ang_img,
				             no_grasps = args.n_grasps,
				             grasp_width_img = width_img)
				fig.savefig('results/img_result{}.pdf'.format(value))
