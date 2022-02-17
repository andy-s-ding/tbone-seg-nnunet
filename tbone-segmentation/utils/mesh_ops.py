"""
mesh_ops.py

much of the obtain_surface_mesh function is borrowed from Max Li's surface extraction code

"""
import numpy as np
import vtk
from vtk.util import numpy_support as nps
import pyvista as pv
import os 
import sys 
import nrrd

from scipy.spatial import distance
from scipy.spatial import KDTree

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from .file_io import *
from .cpd_wrapper import initial_rigid_registration, deformable_registration, procrustes


def structured_points(points): 
	sp = vtk.vtkStructuredPoints()
	sp.SetOrigin(0, 0, 0)
	sp.SetDimensions(3,3,3)
	sp.SetSpacing(1,1,1)

	return 


def visualize(surf): 
	points = np.array(return_surface_points(surf))
	print('visualized points shape', points.shape)
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter3D(points[:,0], points[:,1], points[:,2])

	plt.show()

	return


def return_surface_points(surf): 

	points = []
	num_points = surf.GetNumberOfPoints()
	for i in range(num_points): 
		points.append(surf.GetPoints().GetPoint(i))

	return np.array(points) # (n_points, 3)


def wrap_structured_point(np_array):
	model = pv.UniformGrid()
	# Set the grid dimensions: shape because we want to inject our values on the
	#   POINT data
	model.dimensions = np.array(np_array.shape)
	# Add the data values to the point data
	model.point_arrays["values"] = np_array.flatten(order="F")  # Flatten the array!

	return model


def flip_mesh(mesh, axis=0):
	assert axis < len(mesh.bounds)//2 # [x_lower, x_upper, y_lower, y_upper, ...]
	lower_bound = mesh.bounds[axis*2]
	upper_bound = mesh.bounds[axis*2+1]
	new_mesh = mesh.copy()
	for point in new_mesh.points:
		point[axis] = upper_bound-(point[axis]-lower_bound)
	return new_mesh


def obtain_surface_mesh(data, header, idx=None, is_annotation=False, check_if_cached=None, prefix=None, write_surface=False):
	"""obtain_surface_mesh

	When called, will check for whether there is a cached version of the mesh that should be generated. 

	If you want to obtain a new mesh from scratch (e.g. new parameters, do not pass in the 
	check_if_cached parameter

	If there is no existing mesh, the 
	
	Args:
	    data (TYPE): Description
	    header (TYPE): Description
	    idx (None, optional): Description
	    is_annotation (bool, optional): Description
	    check_if_cached (None, optional): Description
	    prefix (None, optional): Description
	
	Returns:
	    TYPE: Description
	"""
	print('-- meshing surfaces')
	print(header)

	if idx is None: 
		idx = 1

	elif isinstance(idx, int): 
		idx = [idx]

	elif not isinstance(idx, list): 
		print('indices not specified correctly')
		return

	names = get_segmentation_names(header)
 
	# we will assume data is already one-hot
	# data = convert_to_one_hot(data, header)

	surfaces = []

	for i in idx: 

		if check_if_cached is not None: 
			print("---- checking cache for %d" % i)
			check_path = os.path.join(check_if_cached, "%s %s mesh.vtk" % (prefix, names[i]))
			print("---- checking at %s " % check_path)
			if os.path.exists(check_path): 
				print("---- found!")
				surfaces.append(read_vtk(check_path))
				continue
			else: 
				print("---- not found!")

		selected_data = 1*data[i].astype(np.uint8) #???
		print(selected_data.shape)

		print("---- selected data has shape", selected_data.shape)
		print("---- double check: max in data has value", np.max(selected_data))

		vol = vtk.vtkStructuredPoints()

		# the space directions in the header is actually of shape (4, 3), the extra top row is all NaNs because we have 
		# an extra dimension in the seg.nrrd corresponding to the segment index

		offset = 1 if header['space directions'].shape[0] == 4 else 0 
		origin = (
					header['space origin'][0]*np.sign(header['space directions'][0 + offset, 0]), 
					header['space origin'][1]*np.sign(header['space directions'][1 + offset, 1]), 
					header['space origin'][2]*np.sign(header['space directions'][2 + offset, 2])
				)

		spacing = (
					abs(header['space directions'][0 + offset, 0]), 
					abs(header['space directions'][1 + offset, 1]), 
					abs(header['space directions'][2 + offset, 2])
				)  # These are the cell sizes along each axis

		print("---- check: origin and spacing of structured points")
		print("-------", origin, spacing)

		vol.SetDimensions(selected_data.shape[0], selected_data.shape[1], selected_data.shape[2])
		vol.SetOrigin(origin[0], origin[1], origin[2])
		vol.SetSpacing(spacing[0], spacing[1], spacing[2])
		
		scalars = nps.numpy_to_vtk(selected_data.ravel(order='F'), deep=True)
		vol.GetPointData().SetScalars(scalars)

		dmc = vtk.vtkDiscreteMarchingCubes()
		dmc.SetInputData(vol)
		dmc.GenerateValues(1, 1, 1)
		dmc.ComputeGradientsOff()
		dmc.ComputeNormalsOff()
		dmc.Update()

		print("---- extract surface complete")

		# post processing
		surface = pv.wrap(dmc.GetOutput())

		# visualize(surface)

		if surface.is_all_triangles():
			surface.triangulate(inplace=True)
		surface.decimate_pro(
			0.01, feature_angle=60, splitting=False, preserve_topology=True, inplace=True)

		edges = surface.extract_feature_edges(60)
		# p = pv.Plotter()
		# p.add_mesh(surface, color=True)
		# p.add_mesh(edges, color="red", line_width=5)
		# p.show()

		# previously, the relaxation_factor was .33 and the feature_angle was 60. 
		# the relaxation factor was lowered to reduce the extent of volume reduction. Reductions in
		# surface smoothness were also observed as a result. 

		# make some adjustments based on observed volume loss in long processes of incus, malleus, chorda

		relaxation_factor = .16 if i in [1, 2, 3, 10] else .25
		n_iter = 23 if i in [1, 2, 3, 10] else 30

		surface.smooth(n_iter=n_iter, relaxation_factor=relaxation_factor, 
					   feature_angle=70, boundary_smoothing=False, inplace=True)

		if not is_annotation:
			# remove small parts for temporal bones
			if i == 0:
				surface = surface.connectivity(largest=True) # remove small isosurfaces

			# lmao i noticed there was one weird blob in a malleus in 138 so ... remove it...
			if i == 1: 
				surface = surface.connectivity(largest=True)

		surface.compute_normals(inplace=True)

		print("---- post processing complete")

		if write_surface and check_if_cached is not None: 
			print("---- writing surface for %d" % i)
			save_path = os.path.join(check_if_cached, "%s %s mesh.vtk" % (prefix, names[i]))
			surface.save(save_path)

		surfaces.append(surface)
		# visualize(surface)

	if len(surfaces) == 1: 
		return surfaces[0]

	return surfaces


def annotation_onto_meshes(annotation_meshes, mesh_original, mesh_targets, 
		average_annotation_points=True, decimation_factor=None, visualize_reg=False): 
	"""Summary
	
	Args:
	    annotation_meshes (TYPE): Description
	    mesh_original (TYPE): Description
	    mesh_targets (TYPE): Description
	    average_annotation_points (bool, optional): Description
	    decimation_factor (None, optional): Description
	    visualize_reg (bool, optional): Description
	
	Returns:
	    TYPE: Description
	"""

	print('mapping annotations from mesh to mesh')

	# get points of annotation_meshes
	annotations_as_points = [return_surface_points(mesh) for mesh in annotation_meshes]

	if average_annotation_points: 
		annotations_as_points = [np.mean(points, axis=0) for points in annotations_as_points]
	
	if decimation_factor is not None: 
		mesh_original = mesh_original.decimate_pro(decimation_factor, preserve_topology=True)

	# get points of mesh_original
	mesh_original_points = return_surface_points(mesh_original)

	# closest point of mesh_original to annotation landmark
	# min_dists should be of shape (n_annotations, points_per_annotation,)
	# min_indices should be of shape (n_annotations, points_per_annotation,)

	min_dists = []
	min_indices=  []
	for annotation in annotations_as_points:
		dists, indices = closest_points(annotation, mesh_original_points, use_tree=construct_kd_tree(mesh_original))
		min_dists.append(dists)
		min_indices.append(indices)

	print('---', min_indices)

	# deform mesh_original -> each of mesh_target
	original_to_target_projections = []
	target_mesh_annotation_correspondences = []
	target_mesh_annotation_indices = []
	for mesh_target in mesh_targets: 

		if decimation_factor is not None: 
			decimated_mesh_target = mesh_target.decimate_pro(decimation_factor, preserve_topology=True)
		
		mesh_target_points = return_surface_points(decimated_mesh_target)

		registered_original, _ = initial_rigid_registration(mesh_original_points, mesh_target_points, visualize_reg=visualize_reg)

		registered_original, _ = deformable_registration(registered_original, mesh_target_points, visualize_reg=visualize_reg)

		original_to_target_projections.append(registered_original)

		cur_mesh_annotations = []
		cur_mesh_indices = []
		for index_group in min_indices:

			mesh_target_points_undecimated = return_surface_points(mesh_target)

			reg_dists, reg_indices = closest_points(registered_original[index_group], mesh_target_points_undecimated, 
										use_tree=construct_kd_tree(mesh_target))
			cur_mesh_annotations.append(mesh_target_points_undecimated[reg_indices])
			cur_mesh_indices.append(reg_indices)

		target_mesh_annotation_correspondences.append(cur_mesh_annotations)
		target_mesh_annotation_indices.append(cur_mesh_indices)

	return target_mesh_annotation_correspondences, target_mesh_annotation_indices


def return_mesh_with_annotations(mesh_path, segment="facial"): 

	annotation_names = []

	if "facial" in segment: 
		annotation_names = [
			"1st Genu",
			"2nd Genu",
			"Nerve Labyrinthine",
			"Nerve Mastoid",
		]
	elif "chorda" in segment: 
		annotation_names = [
			"Tympani Base",
			"Tympani Tip",
		]
	elif "malleus" in segment: 
		annotation_names = [
			"Manubrium Tip", 
			"Process Tip",
		]
	elif "incus" in segment: 
		annotation_names = [
			"Long Process Tip",
			"Short Process Tip",
		]
	else: 
		print("segment specified incorrectly")
		return

	mesh = pv.read(mesh_path)
	points = mesh.points
	annotations = {}

	for annotation_name in annotation_names:
		whole_array = pv.get_array(mesh, annotation_name, preference="point")
		annotations[annotation_name] = np.squeeze(np.array(points[np.where(whole_array==1)]))

	return mesh, annotations


def apply_connectivity(polydata_input, seed_point=None, largest_region=False, as_pyvista=True): 

	connect = vtk.vtkPolyDataConnectivityFilter()
	connect.SetInputData(polydata_input)
	connect.SetExtractionModeToClosestPointRegion()

	if seed_point is not None: 
		connect.SetClosestPoint(seed_point[0], seed_point[1], seed_point[2])
	else: 
		connect.SetExtractionModeToLargestRegion()

	connect.Update()

	region = connect.GetOutput()

	if as_pyvista: 
		return pv.wrap(region)
	else: 
		return region


def construct_kd_tree(mesh, cache=None, reverse_points=False): 
	"""construct_kd_tree
	
	Args:
	    mesh (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	surface_points = return_surface_points(mesh)

	if reverse_points: 
		surface_points = surface_points[::-1]
		
	tree = KDTree(surface_points, copy_data=True)

	if cache is not None: 
		write_to_file(cache, tree)

	return tree


def closest_points(points_query, points_target, use_tree=None): 

	if use_tree is not None:
		print("using tree. points_target will be ignored.") 

		if isinstance(use_tree, str): 		
			print("- reading in cached tree.")
			cache_result = check_if_cached(use_tree)
			if cache_result: 
				file = open(path, 'rb')
				tree = pickle.load(file)
				file.close()

		else: 
			tree = use_tree

		return tree.query(points_query, k=1)

	else: 
		print("brute force closest points.")
		distances = np.sqrt(((points_query - points_target[:, np.newaxis])**2).sum(axis=2))
		print(distances.shape)
		return (np.amin(distances, axis=0), np.argmin(distances, axis=0))


def main(): 
	data, header = nrrd.read("/Users/alex/Documents/segmentations/annotated/Segmentation RT 153 Annotated.nrrd")
	segment_names = read_from_file('segment_names')
	obtain_surface_mesh(data, header, idx=[1, 2, 3, 4, 9, 10, 15])
	obtain_surface_mesh(data, header, idx=2)
	obtain_surface_mesh(data, header, idx=3)
	obtain_surface_mesh(data, header, idx=4)
	obtain_surface_mesh(data, header, idx=9)
	obtain_surface_mesh(data, header, idx=10)
	obtain_surface_mesh(data, header, idx=15)

	return


if __name__ == "__main__": 
	main()
