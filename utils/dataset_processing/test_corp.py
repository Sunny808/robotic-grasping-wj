from PIL import Image
import numpy as np
from utils.data.camera_data import CameraData
from utils.dataset_processing import grasp, image
from imageio import imread


img = Image.open('../../cornell/10/pcd1012r.png')
img.show('rgb')
rgb = np.array(img)
print(type(rgb), rgb.shape)
rgb1 = rgb.transpose(2, 0, 1)
print(type(rgb1), rgb1.shape)
rgb1 = rgb1.transpose(2, 1, 0)
print(type(rgb1), rgb1.shape)
pic = Image.fromarray(rgb1.astype(np.uint8))
pic.show('pic')


img_data = CameraData(include_depth=0, include_rgb=1, path = '../../cornell/10/pcd1012cpos.txt')

print('Testing run_offline.corp\n')
rgb_img = img_data.get_rgb(rgb, norm = False)
'''print(rgb_img.shape, type(rgb_img))
rgb_img = rgb_img.transpose(2, 1, 0)
print(rgb_img.shape, type(rgb_img))
'''
rgb_img = Image.fromarray(rgb_img.astype(np.uint8))
rgb_img.show()

print('Testing train_network.corp\n')
gtbbs = grasp.GraspRectangles.load_from_cornell_file('../../cornell/10/pcd1012cpos.txt')
center = gtbbs.center
left = max(0, min(center[1] - 224 // 2, 640 - 224))
top = max(0, min(center[0] - 224 // 2, 480 - 224))
img1 = rgb[top:min(480, top + 224), left:min(640, left + 224)]
img1 = Image.fromarray(img1)
img1.show()
