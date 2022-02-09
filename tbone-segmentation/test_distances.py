"""tests.py
	0: malleus lateral 
	1: malleus manubrium
	2: incus short 
	3: incus long
	4: incus diam
	5: facial labryinthine
	6: facial 1st genu
	7: facial 2nd genu
	8: facial mastoid
	9: facial mastoid segment
	10: chorda tip
	11: chorda base 
	12: EAC point 
"""
import pyvista
import numpy as np
import nrrd
import pandas as pd
import glob

from utils.metrics import *
from utils.mesh_ops import *
from utils.file_io import *
from utils.centerline import *
from propagate_segments import adjust_file_path

groundtruth_dir = '/media/andyding/EXTREME SSD/ANTs-registration/ssm_surfaces_groundtruth/'
predictions_dir = '/media/andyding/EXTREME SSD/ANTs-registration/ssm_surfaces/'
structure = "Malleus"
side = "RT"
template = "153"

print(glob.glob(groundtruth_dir + structure + "/cpd/" + structure + "_" + side + "_*_cpd.vtk" ))
RT_142_cpd = read_vtk('/media/andyding/EXTREME SSD/ANTs-registration/ssm_surfaces_groundtruth/Malleus/cpd/Malleus_RT_142_cpd.vtk')
RT_153_142_cpd = read_vtk('/media/andyding/EXTREME SSD/ANTs-registration/ssm_surfaces/Malleus/cpd/Malleus_RT_153_142_cpd.vtk')

points_142 = return_surface_points(RT_142_cpd)
points_153_142 = return_surface_points(RT_153_142_cpd)*np.array([-1,-1,1])

points_142_zero = points_142 - np.mean(points_142, axis=0)
points_153_142_zero = points_153_142 - np.mean(points_153_142, axis=0)

print(np.mean(points_142, axis=0))
print(np.mean(points_153_142, axis=0))

print(np.mean(np.linalg.norm(points_142_zero-points_153_142_zero, axis=1)))