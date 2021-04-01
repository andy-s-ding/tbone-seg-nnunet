"""centerline.py

"""
import numpy as np
from scipy import interpolate 
import vtk
import pyvista as pv
import os

from .mesh_ops import construct_kd_tree, closest_points

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def surface_interpolation_1D(points, npoints=400, mesh=None, visualize=False): 

	print(points.shape)

	# spline parameters
	s=0.01 	# smoothness parameter
	k=1 	# spline order
	nest=-1 # estimate of number of knots needed (-1 = maximal)
	nSpoints_factor = 30

	tckp,u = interpolate.splprep(np.transpose(points), s=s,k=k, nest=6)

	xs,ys,zs = interpolate.splev(np.linspace(0,1,npoints), tckp)

	if visualize: 
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		ax.scatter(xs, ys, zs)
		plt.show()

	return np.transpose(np.stack((xs, ys, zs)))



def read_centerline(vtp_file): 

	# open the polydata
	reader = vtk.vtkXMLPolyDataReader()
	reader.SetFileName(vtp_file)
	reader.Update()
	polydata = reader.GetOutput()

	NoP = model.GetNumberOfPoints()
	points = np.zeros((NoP, 3))
	for i in range(NoP):
		points[i] = model.GetPoint(i)

	return points


def resample_centerline(centerline, length=None, with_derivative=False):

	print('trying to resample centerline')
	print(centerline.shape)

	# spline parameters
	s=0.01 	# smoothness parameter
	k=3 	# spline order
	nest=-1 # estimate of number of knots needed (-1 = maximal)
	nSpoints_factor = 30

	centerline = np.transpose(centerline)

	# find the knot points
	tckp,u = interpolate.splprep(centerline, s=s,k=k)

	print(centerline.shape)

	if length is None:
		length = len(centerline[0])
	nSpoints = length*30

	print('-----   the number of spline points is ', nSpoints)

	# evaluate spline, including interpolated points
	xs,ys,zs = interpolate.splev(np.linspace(0,1,nSpoints),tckp)

	print('done resampling centerline')
	print('--------------------------')

	if with_derivative:
		print('including local derivatives')
		print('---------------------------')
		xp, yp, zp = intplt.splev(np.linspace(0,1,nSpoints), tck, der=1)
		xpp, ypp, zpp = intplt.splev(np.linspace(0,1,nSpoints), tck, der=2)
		return (np.transpose(np.stack((xs, ys, zs))), np.transpose(np.stack((xp, yp, zp))), np.transpose(np.stack((xpp, ypp, zpp))))

	return np.transpose(np.stack((xs, ys, zs)))


def compute_reference_norm(centerline):
	"""Summary
	
	Args:
	    centerline (ndarray): points in the centerline
	
	Returns:
	    TYPE: Description
	"""
	print('computing reference norms')
	print('-------------------------')

	NoP = centerline.shape[0]
	p0 = np.roll(centerline, shift = -1, axis= 1)
	p1 = np.roll(centerline, shift = 0, axis = 1)
	p2 = np.roll(centerline, shift = 1, axis = 1)

	t21 = p2 - p1
	t21_normed = np.divide(t21, np.linalg.norm(t21, axis=1).reshape(NoP, 1))

	t10 = p1 - p0
	t10_normed = np.divide(t10, np.linalg.norm(t10, axis=1).reshape(NoP, 1))

	n1 = t21_normed - t10_normed
	return (np.divide(n1, np.linalg.norm(n1, axis=1).reshape(NoP, 1)), t21)


def normalized_centerline(center):
	'''
	input:
		* np array of shape (NoP, 3)

	output: 
		* NoP
		* np array of length NoP, containing normalized coordinate for each point
		* total centerline length

	Assigns each centerline point a total length-normalized position, holding assigned coordinate
	in form of np array with shape (NoP,). 
	
	'''

	print('normalizing the centerline')
	print('--------------------------')
	centerline_length = 0.0
	NoP = len(center)
	normalized = np.zeros(NoP)

	for i in range(1, NoP):
		pt = center[i]
		prev_pt = center[i-1]
		d_temp = vtk.vtkMath.Distance2BetweenPoints(pt, prev_pt)
		d_temp = np.sqrt(d_temp)
		centerline_length += d_temp
		normalized[i] = centerline_length

	normalized /= centerline_length

	return (NoP, normalized, centerline_length)


def project_onto_centerline(centerline, wallpoints, tree=None): 


	# centerline = resample_centerline(centerline)

	NoP_center, normalized_center, centerline_length = normalized_centerline(centerline)
	# reference_norms, reference_tangents = compute_reference_norm(centerline)
	transformed_wall_ref = np.zeros((len(wallpoints), 2))

	if tree is None: 
		tree = construct_kd_tree(centerline)

	dd, ii = closest_points(wallpoints, centerline, use_tree=tree)

	return dd, ii, normalized_center[ii]


def main():
	return

if __name__ == "__main__":
	main()
