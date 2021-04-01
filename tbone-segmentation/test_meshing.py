import pyvista as pv
import numpy as np
from utils.mesh_ops import obtain_surface_mesh
from utils.file_io import convert_to_one_hot


def wrap_structured_cell(np_array):
	model = pv.UniformGrid()
	# Set the grid dimensions: shape because we want to inject our values on the
	#   CELL data
	model.dimensions = np.array(np_array.shape)+1
	# Add the data values to the cell data
	model.cell_arrays["values"] = np_array.flatten(order="F")  # Flatten the array!

	return model

def wrap_structured_point(np_array):
	model = pv.UniformGrid()
	# Set the grid dimensions: shape because we want to inject our values on the
	#   POINT data
	model.dimensions = np.array(np_array.shape)
	# Add the data values to the point data
	model.point_arrays["values"] = np_array.flatten(order="F")  # Flatten the array!

	return model

def filter_value(np_array, value):
	v_unstruct = wrap_structured_cell(np_array).cast_to_unstructured_grid()
	ghosts = np.argwhere(v_unstruct["values"] == value)
	v_unstruct.remove_cells(ghosts)

	return v_unstruct

import nrrd
import numpy as np
import vtk
import pyvista as pv
import os


def extract_surface(data, header, idx_list=[9]):
	# TODO: extract other components only when not existing

	composite = []

	for i in idx_list:

		anatomy = ((data[i] == 1)*1).astype(np.uint8)  # keep the label
		print(anatomy.shape)
		max_size = np.max(anatomy.shape)

		new_anatomy = np.zeros((max_size, max_size, max_size))
		new_anatomy[:anatomy.shape[0], :anatomy.shape[1], :anatomy.shape[2]] = anatomy

		model = wrap_structured_point(new_anatomy)

		# Edit the spatial reference
		model.origin = (header['space origin'][0]*np.sign(header['space directions'][1, 0]), 
						header['space origin'][1]*np.sign(header['space directions'][2, 1]), 
						header['space origin'][2]*np.sign(header['space directions'][3, 2])
						)
		model.spacing = (
						abs(header['space directions'][1, 0]), 
						abs(header['space directions'][2, 1]), 
						abs(header['space directions'][3, 2])
						)  # These are the cell sizes along each axis

		# discrete marching cubes
		dmc = vtk.vtkDiscreteMarchingCubes()
		dmc.SetInputDataObject(model)
		dmc.GenerateValues(1, 1, 1)
		dmc.ComputeGradientsOff()
		dmc.ComputeNormalsOff()
		dmc.Update()
		print("extract surface complete")

		# post processing
		surface = pv.wrap(dmc.GetOutput())

		edges = surface.extract_feature_edges(60)
		p = pv.Plotter()
		p.add_mesh(surface, color=True)
		p.add_mesh(edges, color="red", line_width=5)
		p.show()
		
		if surface.is_all_triangles():
			surface.triangulate(inplace=True)
		surface.decimate_pro(
			0.01, feature_angle=60, splitting=False, preserve_topology=True, inplace=True)
		surface.smooth(n_iter=40, relaxation_factor=0.33,
					   feature_angle=60, boundary_smoothing=False, inplace=True)
		# remove small parts for temporal bones
		
		edges = surface.extract_feature_edges(60)
		p = pv.Plotter()
		p.add_mesh(surface, color=True)
		p.add_mesh(edges, color="red", line_width=5)
		p.show()
		surface.compute_normals(inplace=True)

		print("post processing complete")

		composite.append(surface)

	return composite


def main(): 
	data, header = nrrd.read("/Users/alex/Documents/segmentations/Segmentation RT 138.seg.nrrd")
	# extract_surface(data, header)
	data = convert_to_one_hot(data, header)
	surfaces = obtain_surface_mesh(data, header, idx=[1, 2, 3])

	for i, surface in enumerate(surfaces): 
		surface.save('ssm_out/test %d.vtk' % i)
		print(surface)

	return


if __name__ == "__main__":
	main()