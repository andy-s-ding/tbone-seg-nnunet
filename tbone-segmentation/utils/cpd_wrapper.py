"""cpd_wrapper.py


"""
import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial 

from pycpd import DeformableRegistration, RigidRegistration
from scipy.linalg import orthogonal_procrustes
# from scipy.spatial import procrustes

import os 
import sys 

from .file_io import *


def visualize(iteration, error, X, Y, ax):
	"""visualize

	code borrowed from PyCPD implementation example available at

	https://github.com/siavashk/pycpd/blob/master/examples/fish_deformable_3D.py
	
	Args:
		iteration (TYPE): Description
		error (TYPE): Description
		X (TYPE): Description
		Y (TYPE): Description
		ax (TYPE): Description
	"""
	plt.cla()
	ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
	ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
	ax.text2D(0.87, 0.92, 'Iteration: {:d}'.format(
		iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
	ax.legend(loc='upper left', fontsize='x-large')
	plt.draw()
	plt.pause(0.001)


def initial_rigid_registration(points_source, points_target, visualize_reg=False): 
	"""initial_rigid_registraiton
	
	the registration of a source cloud onto a target cloud
	
	Args:
		points_source (ndarray): should be of shape n_source_points, n_dimensions
		points_target (ndarray): should be of shape n_target_points, n_dimensions
		visualize_reg (bool, optional): whether the registration should be visualized using callback
	
	Returns:
		TYPE: Description
	"""
	print('--- rigid CPD registration')

	print('----- source points of shape:', points_source.shape)
	print('----- targ points of shape:', points_target.shape)

	mu_source = np.mean(points_source, axis=0)
	mu_target = np.mean(points_target, axis=0)

	points_source = points_source - mu_source
	points_target = points_target - mu_target

	reg = RigidRegistration(**{'X': points_target, 'Y': points_source})

	if visualize_reg: 
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		callback = partial(visualize, ax=ax)
		points_transformed, (s, R, t) = reg.register(callback)
	
	else: 
		points_transformed, (s, R, t) = reg.register()
	
	P = reg.P

	# order the source cloud points according to the correspondences determined by CPD
	# we already have the transformed source but re-ordering the original source will enable us 
	# to compute the difference pre and post transformation
	corresponded = points_target[np.argmax(P, axis=1), :]

	# the transformed points should be in the right order already, but I guess we can double check
	test_transformed = s*np.dot(points_source, R) + t
	# test_transformed = test_transformed[np.argmax(P, axis=1), :]

	# print(P.shape)
	print(np.argmax(P, axis=0).shape)
	# print('----- the norm of manual and automatic transform', np.linalg.norm(points_transformed - test_transformed))
	print('---- the norm of transformed - original points', np.linalg.norm(points_transformed - points_source))
	print('---- the norm of target - transformed points', np.linalg.norm(points_transformed - corresponded))
	# print('---- the difference in the means is', 
	# 		np.mean(points_transformed, axis=0), np.mean(points_source, axis=0), np.mean(points_target, axis=0))

	points_transformed += mu_target

	return points_transformed, (s, R, t, P)


def deformable_registration(points_source, points_target, visualize_reg=False): 
	"""deformable_registration
	
	the registration of a source cloud onto a target cloud, deformably
	
	Args:
		points_source (ndarray): should be of shape n_source_points, n_dimensions
		points_target (ndarray): should be of shape n_target_points, n_dimensions
		visualize_reg (bool, optional): whether the registration should be visualized using callback
	
	Returns:
		TYPE: Description
	"""
	print('--- deformable CPD registration')

	print('----- source points of shape:', points_source.shape)
	print('----- targ points of shape:', points_target.shape)

	mu_source = np.mean(points_source, axis=0)
	mu_target = np.mean(points_target, axis=0)

	points_source = points_source - mu_source
	points_target = points_target - mu_target

	reg = DeformableRegistration(**{'X': points_target, 'Y': points_source})

	if visualize_reg: 
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		callback = partial(visualize, ax=ax)
		points_transformed, (G, W) = reg.register(callback)
	
	else: 
		points_transformed, (G, W) = reg.register()
	
	P = reg.P

	# order the source cloud points according to the correspondences determined by CPD
	# we already have the transformed source but re-ordering the original source will enable us 
	# to compute the difference pre and post transformation
	corresponded = points_target[np.argmax(P, axis=1), :]

	# the transformed points should be in the right order already, but I guess we can double check
	# test_transformed = s*np.dot(points_source, R) + t
	# test_transformed = test_transformed[np.argmax(P, axis=1), :]

	# print(P.shape)
	# print(np.argmax(P, axis=0).shape)
	# print(np.sum(points_transformed - test_transformed))
	print('----- the norm of transformed - original points', np.linalg.norm(points_transformed - points_source))
	print('----- the norm of target - transformed points', np.linalg.norm(points_transformed - corresponded))

	points_transformed += mu_target

	return points_transformed, (G, W)



# def procrustes(X, iterations=10, tol=.01): 
# 	"""procrustes transformation
	
# 	Args:
# 		X (TYPE): Description
# 		tol (float, optional): Description
	
# 	Returns:
# 		TYPE: Description
	
# 	"""
# 	nshape, nfeat = X.shape
# 	initial_shape = X[0].reshape((nfeat/1000, 3), order="F")

# 	aligned = np.zeros(X)

# 	# procrustes loop
# 	for iteration in range(iterations): 

# 		#  alignment to mean shape or initial guess
# 		for i, shape in enumerate(X[1:]):
# 			shape = shape.reshape((nfeat/1000, 3), order="F")
# 			initial_shape, shape, disparity = procrustes(initial_shape, shape)

# 			if i == 0: 
# 				aligned[0] = initial_shape.reshape(nfeat*3)

# 			aligned[i+1] = shape.reshape(nfeat*3)

# 		# (re)compute the mean shape
# 		mean_shape = np.mean(aligned, axis=0)
# 		mean_shape, initial_shape, disparity = procrustes(mean_shape, initial_shape)
		
# 		# check if mean shape has changed
# 		if disparity > tol: 
# 			initial_shape = mean_shape

# 		else:
# 			break

# 	return X
	

def procrustes(X, Y, scaling=True, reflection='best'):
	"""
	A port of MATLAB's `procrustes` function to Numpy.

	Procrustes analysis determines a linear transformation (translation,
	reflection, orthogonal rotation and scaling) of the points in Y to best
	conform them to the points in matrix X, using the sum of squared errors
	as the goodness of fit criterion.

		d, Z, [tform] = procrustes(X, Y)

	Inputs:
	------------
	X, Y    
		matrices of target and input coordinates. they must have equal
		numbers of  points (rows), but Y may have fewer dimensions
		(columns) than X.

	scaling 
		if False, the scaling component of the transformation is forced
		to 1

	reflection
		if 'best' (default), the transformation solution may or may not
		include a reflection component, depending on which fits the data
		best. setting reflection to True or False forces a solution with
		reflection or no reflection respectively.

	Outputs
	------------
	d       
		the residual sum of squared errors, normalized according to a
		measure of the scale of X, ((X - X.mean(0))**2).sum()

	Z
		the matrix of transformed Y-values

	tform   
		a dict specifying the rotation, translation and scaling that
		maps X --> Y

	"""

	n,m = X.shape
	ny,my = Y.shape

	muX = X.mean(0)
	muY = Y.mean(0)

	X0 = X - muX
	Y0 = Y - muY

	ssX = (X0**2.).sum()
	ssY = (Y0**2.).sum()

	# centred Frobenius norm
	normX = np.sqrt(ssX)
	normY = np.sqrt(ssY)

	# scale to equal (unit) norm
	X0 /= normX
	Y0 /= normY

	if my < m:
		Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

	# optimum rotation matrix of Y
	A = np.dot(X0.T, Y0)
	U,s,Vt = np.linalg.svd(A,full_matrices=False)
	V = Vt.T
	T = np.dot(V, U.T)

	if reflection != 'best':

		# does the current solution use a reflection?
		have_reflection = np.linalg.det(T) < 0

		# if that's not what was specified, force another reflection
		if reflection != have_reflection:
			V[:,-1] *= -1
			s[-1] *= -1
			T = np.dot(V, U.T)

	traceTA = s.sum()

	if scaling:

		# optimum scaling of Y
		b = traceTA * normX / normY

		# standarised distance between X and b*Y*T + c
		d = 1 - traceTA**2

		# transformed coords
		Z = normX*traceTA*np.dot(Y0, T) + muX

	else:
		b = 1
		d = 1 + ssY/ssX - 2 * traceTA * normY / normX
		Z = normY*np.dot(Y0, T) + muX

	# transformation matrix
	if my < m:
		T = T[:my,:]
	c = muX - b*np.dot(muY, T)

	#transformation values 
	tform = {'rotation':T, 'scale':b, 'translation':c}

	return d, Z, tform


def main(): 
	pass

if __name__ == "__main__": 
	main()