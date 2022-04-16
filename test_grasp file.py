from utils.dataset_processing import grasp
import glob
import os
import numpy as np

file_path = 'cornell'
grasp_files = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))

def _get_crop_attrs(idx):
    gtbbs = grasp.GraspRectangles.load_from_cornell_file(grasp_files[idx])
    center = gtbbs.center  # return 1X2 array
    left = max(0, min(center[1] - 224 // 2, 640 - 224))
    top = max(0, min(center[0] - 224 // 2, 480 - 224))
    return center, left, top


def get_gtbb(idx, rot=0, zoom=1.0):
    gtbbs = grasp.GraspRectangles.load_from_cornell_file(grasp_files[idx])
    center, left, top = _get_crop_attrs(idx)
    gtbbs.rotate(rot, center)
    gtbbs.offset((-top, -left))
    gtbbs.zoom(zoom, (224 // 2, 224 // 2))
    return gtbbs


center, left, top = _get_crop_attrs(0)

print(grasp.GraspRectangles.load_from_cornell_file(grasp_files[0]))
print(grasp.GraspRectangles.load_from_cornell_file(grasp_files[0]).rotate)