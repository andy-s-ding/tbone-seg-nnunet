"""
metrics.py
"""
import numpy as np
import os 
import sys 
import nrrd
import pyvista as pv
from scipy.spatial import distance

# from .mesh_ops import obtain_surface_mesh, return_surface_points, construct_kd_tree
from .mesh_ops import *
from .file_io import *


def calc_dice(segmentation_truth, segmentation_pred, indices=None): 

	print('--calculating dice coefficients')

	if indices is None: 
		# calculate dice for each index 
		indices = list(range(segmentation_truth.shape[0]))

	elif isinstance(indices, list): 
		# calculate dice for each index specified in list
		indices = [int(idx) for idx in indices] 
		if any(idx >= segmentation_truth.shape[0] for idx in indices): 
			print('--- error with index specification in dice calculation')
			return 

	elif isinstance(indices, str): 
		# calculate dice for specified index
		indices = int(indices)
		if indices >= segmentation_truth.shape[0]:
			print('--- error with index specification in dice calculation')
			return

		indices = [indices]

	else: 
		print('--- error with index specification in dice calculation')
		return

	computed_dice = []
	for idx in indices: 
		segment_truth = segmentation_truth[idx]
		segment_pred = segmentation_pred[idx]

		if segment_truth.shape != segment_pred.shape: 
			print('%d shapes do not agree'%(idx), segment_truth.shape, segment_pred.shape)
			computed_dice.append(0)
		else:
			segment_truth = np.asarray(segment_truth).astype(np.bool)
			segment_pred = np.asarray(segment_pred).astype(np.bool)

			intersection = np.logical_and(segment_truth, segment_pred)

			computed_dice.append(2. * intersection.sum() / (segment_truth.sum() + segment_pred.sum()))
			# computed_dice.append(distance.dice(segmentation_truth[idx].ravel(), segmentation_pred[idx].ravel()))

	return computed_dice


def calc_volume_dice(data_truth, header_truth, data_pred, header_pred, 
			indices=None, mesh_cache=None, prefix_truth=None, prefix_pred=None): 
	"""calc_volume_dice


	"""
	print('-- calculating dices from meshes')

	if indices is None: 
		indices = [2, 3, 4]
	else: 
		indices = [int(idx) for idx in indices]

	dices = []

	surfs_truth = obtain_surface_mesh(data_truth, header_truth, 
									idx=indices, check_if_cached=mesh_cache, prefix=prefix_truth)
	surfs_pred = obtain_surface_mesh(data_pred, header_pred, 
									idx=indices, check_if_cached=mesh_cache, prefix=prefix_pred)

	connectivity_seeds = None
	for idx, surf_truth, surf_pred in zip(indices, surfs_truth, surfs_pred): 

		if idx == 1: 
			surf_truth = surf_truth.connectivity(largest=True)
			surf_pred = surf_pred.connectivity(largest=True)

		elif idx == 9: 
			connectivity_seeds = [get_centroid(surf_truth), get_centroid(surf_pred)]
			surf_truth = apply_connectivity(surf_truth, largest_region=True)
			surf_pred = apply_connectivity(surf_pred, largest_region=True)
		
		elif idx == 10: 
			surf_truth = apply_connectivity(surf_truth, seed_point=connectivity_seeds[0])
			surf_pred = apply_connectivity(surf_pred, seed_point=connectivity_seeds[1])

		# else: 
		# 	surf_truth = apply_connectivity(surf_truth, largest_region=True)
		# 	surf_pred = apply_connectivity(surf_pred, largest_region=True)
		
		# surf_truth.clean(inplace=True)
		# surf_pred.clean(inplace=True)

		# intersection_complement = surf_pred.boolean_difference(surf_truth)
		# intersection_complement_2 = surf_truth.boolean_difference(surf_pred)

		# union = surf_pred.boolean_union(surf_truth)

		# p = pv.Plotter(shape=(1, 2))
		# p.add_mesh(intersection_complement, show_edges=True, opacity=.7)
		# p.add_mesh(intersection_complement_2, show_edges=True, opacity=.7)

		# p.subplot(0, 1)
		# p.add_mesh(union, show_edges=True, opacity=.7)
		
		# p.link_views()

		# p.show()

		# computed_dice = (union.volume - intersection_complement.volume - intersection_complement_2.volume) / union.volume

		try:
			intersection_mesh = surf_pred.boolean_cut(surf_truth) # order of operations does not matter for this one 
			intersection = intersection.volume
		except:
			intersection = -1.

		try: 
			union_mesh = surf_pred.boolean_union(surf_truth)
			union = union_mesh.volume
		except:
			union = 1.0

		computed_dice = intersection/union

		print("---- dice for idx %d, result %f" % (idx, computed_dice))

		dices.append(computed_dice)

	return dices, surfs_truth, surfs_pred


def wrap_max_hausdorff(pts_truth, pts_pred): 
	d_1, idx_11, idx_12 = distance.directed_hausdorff(pts_truth, pts_pred)
	d_2, idx_21, idx_22 = distance.directed_hausdorff(pts_pred, pts_truth)

	return max(d_1, d_2), (d_1, idx_11, idx_12), (d_2, idx_21, idx_22)


def wrap_average_hausdorff(pts_truth, pts_pred, use_tree=None): 

	dd, ii = closest_points(pts_pred, pts_truth, use_tree=use_tree)

	return np.mean(dd)


def calc_hausdorff(data_truth, header_truth, data_pred, header_pred, 
			indices=None, mesh_cache=None, prefix_truth=None, prefix_pred=None): 
	"""calc_hausdorff

	Wrapper function that gets meshes, calculates hausdorff distance for the specified segmentation indices. 

	The header information is used by the obtain_surface_mesh functions to either compute the meshes from scratch
	or get naming information to read them from an optionally specified cache location.

	obtain_surface_mesh will look at [mesh_cache] / [prefix] [segment_name] mesh.vtk to see if there is a cached
	mesh. 

	Do not specify mesh cache if meshes should be recomputed from scratch. 
	
	Args:
		data_truth (ndarray): Description
		header_truth (OrderedDict): Description
		data_pred (ndarray): Description
		header_pred (OrderedDict): Description
		indices (None, optional): list of strings specifying indices that should be inputs into calculation
		mesh_cache (None, optional): optional location where meshes may be found
		prefix (None, optional): optional prefix that precedes mesh name 
	
	Returns:
		TYPE: Description
	"""
	print('-- calculating hausdorff distances')

	if indices is None: 
		indices = [2, 3, 4]
	else: 
		indices = [int(idx) for idx in indices]

	hausdorffs = []

	surfs_truth = obtain_surface_mesh(data_truth, header_truth, 
									idx=indices, check_if_cached=mesh_cache, prefix=prefix_truth)
	surfs_pred = obtain_surface_mesh(data_pred, header_pred, 
									idx=indices, check_if_cached=mesh_cache, prefix=prefix_pred)

	connectivity_seeds = None
	for idx, surf_truth, surf_pred in zip(indices, surfs_truth, surfs_pred): 

		if idx == 1: 
			surf_truth = surf_truth.connectivity(largest=True)
			surf_pred = surf_pred.connectivity(largest=True)

		elif idx == 9: 
			connectivity_seeds = [get_centroid(surf_truth), get_centroid(surf_pred)]
		
		elif idx == 10: 
			surf_truth = apply_connectivity(surf_truth, seed_point=connectivity_seeds[0])
			surf_pred = apply_connectivity(surf_pred, seed_point=connectivity_seeds[1])

		pts_truth = return_surface_points(surf_truth)
		pts_pred = return_surface_points(surf_pred)

		# computed_hausdorff = wrap_hausdorff(pts_truth, pts_pred)[0]

		# computed_hausdorff, (d_1, idx_11, idx_12), (d_2, idx_21, idx_22) = wrap_max_hausdorff(pts_truth, pts_pred)

		# p = pv.Plotter()
		# p.add_mesh(surf_truth, opacity=.4)
		# p.add_mesh(surf_pred, opacity=.4)
		# p.add_lines(np.array([pts_truth[idx_11], pts_pred[idx_12]]), color='black')
		# p.add_lines(np.array([pts_pred[idx_21], pts_truth[idx_22]]), color='black')
		# p.show()

		computed_hausdorff = wrap_average_hausdorff(pts_truth, pts_pred, use_tree=construct_kd_tree(surf_truth))

		print("---- hausdorff for idx %d, result %f" % (idx, computed_hausdorff))

		hausdorffs.append(computed_hausdorff)

	return hausdorffs, surfs_truth, surfs_pred


def calc_volume(meshes):

	volumes = {nrrd_num: {} for nrrd_num in meshes.keys()}
	
	for nrrd_num, mesh_dict in meshes.items(): 
		for mesh_name, mesh in mesh_dict.items(): 
			volumes[nrrd_num][mesh_name] = mesh.volume

	return volumes


def calculate_mesh_dist(mesh1, mesh2, mesh2_landmarks=None, plot=False, opacity=1): 

	tree = construct_kd_tree(mesh1)
	dd = None
	ii = None
	if mesh2_landmarks is not None: 
		dd, ii = closest_points(mesh2_landmarks, mesh1.points, use_tree=tree)
	else: 
		dd, ii = closest_points(mesh2.points, mesh1.points, use_tree=tree)

	p = None
	if plot: 
		p = pv.Plotter()
		p.background_color = 'white'

		p.add_mesh(mesh1, color=True, opacity=opacity)
		p.add_mesh(mesh2, color=True, opacity=opacity)

		closest = np.argmin(dd)

		p.add_points(mesh1.points[ii[closest]], point_size=1, color='black')
		p.add_points(mesh2_landmarks[closest], point_size=1, color='black')

		p.add_lines(np.array([mesh2_landmarks[closest], mesh1.points[ii[closest]]]), label="min dist: " + str(np.min(dd)) + " mm", color='black', width=10)
		# p.add_legend()

	return dd, ii, p


def calc_intra_ossicle_metrics(mesh_ossicle, landmark_short, landmark_long, plot=False, short=True, long=True, diag=True):
	"""calc_incus_metrics
	
	Args:
		mesh_incus (pyvista mesh): the mesh defining the incus
		landmark_incus_short (pyvista mesh): the mesh defining the landmark at short process of incus
		landmark_incus_long (pyvista mesh): mesh defining landmark at long process of incus
	
	Returns:
		list: angle, short process length, long process length, short - long dist
	"""
	# incus_mesh = read_vtk("ssm_out/RT 153 Incus mesh.vtk")
	centroid = get_centroid(mesh_ossicle)
	short_point = get_centroid(landmark_short)
	long_point = get_centroid(landmark_long)

	dist_short = np.linalg.norm(centroid - short_point)
	dist_long = np.linalg.norm(centroid - long_point)

	dist_between = np.linalg.norm(short_point - long_point)

	if plot: 
		p = pv.Plotter()
		p.background_color = 'white'
		p.add_mesh(mesh_ossicle, opacity=.7, color=True)
		p.add_points(short_point, point_size=1, color='black')
		p.add_points(long_point, point_size=1, color='black')

		if short: p.add_lines(np.array([short_point, centroid]), color='black', width=10)
		if long: p.add_lines(np.array([long_point, centroid]), color='black', width=10)
		if diag: p.add_lines(np.array([short_point, long_point]), color='black', width=10)
		p.show()

	return [calc_point_angle(centroid, short_point, long_point), dist_short, dist_long, dist_between]


def get_centroid(mesh):
	"""get_centroid 

	returns point-wise average (assumes mesh density relatively consistent)
	
	Args:
		mesh (pyvista mesh): Description
	
	Returns:
		TYPE: Description
	"""
	mesh_points = return_surface_points(mesh)
	return np.mean(mesh_points, axis=0)


def calc_point_angle(vertex, point1, point2):
	'''
	Returns angle between points in radians
	
	Args:
		vertex (ndarray): vertex at which angle is calculated
		point1 (ndarray): point 1 defining angle
		point2 (ndarray): point 2 defining angle
	
	Returns:
		angle: angle in radians
	'''
	ray1 = point1 - vertex
	ray2 = point2 - vertex
	return np.arccos(np.dot(ray1, ray2)/(np.linalg.norm(ray1)*np.linalg.norm(ray2)))


def triangle_area(point1, point2, point3): 
	u = point1 - point2
	v = point3 - point2
	return .5 * np.linalg.norm(np.cross(u, v))


def main():
	data_truth, header_truth = nrrd.read("/Users/alex/Documents/segmentations/Segmentation RT 146.seg.nrrd")
	data_pred, header_pred = nrrd.read("/Users/alex/Documents/predictions/RT 144 146.seg.nrrd")

	calc_hausdorff(data_truth, header_truth, data_pred, header_pred)

	
if __name__ == "__main__":
	# main()
	test_closest_points()
