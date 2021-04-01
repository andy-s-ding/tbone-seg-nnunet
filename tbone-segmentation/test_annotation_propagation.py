"""annotation_propagation.py

mesh to mesh based annotation propagation
"""
import os 
import sys 
import numpy as np
import pyvista as pv

from utils.mesh_ops import annotation_onto_meshes, apply_connectivity
from utils.file_io import read_vtk



def main(): 
	base = "/Users/alex/Documents/TaylorLab/asm"
	mesh_folder = "ssm_out"
	use_vtk=False
	visualize_reg=False

	# facial_annotation_mesh_paths = [
	# 	"RT 153 Facial Nerve 1st Genu mesh.vtk",
	# 	"RT 153 Facial Nerve 2nd Genu mesh.vtk",
	# 	"RT 153 Facial Nerve Labyrinthine mesh.vtk",
	# 	"RT 153 Facial Nerve Mastoid mesh.vtk"
	# ]

	chorda_annotation_mesh_paths = [
		"RT 153 Chorda Tympani Base mesh.vtk",
		"RT 153 Chorda Tympani Tip mesh.vtk"
	]
	base_mesh_path = "RT 153 Chorda Tympani mesh.vtk"

	test_mesh_path = "RT 147 chorda tympani mesh.vtk"

	annotation_meshes = [read_vtk(os.path.join(base, mesh_folder, mesh_path), use_vtk=use_vtk) for mesh_path in chorda_annotation_mesh_paths]

	base_mesh = read_vtk(os.path.join(base, mesh_folder, base_mesh_path), use_vtk=use_vtk)
	test_mesh = read_vtk(os.path.join(base, mesh_folder, test_mesh_path), use_vtk=use_vtk)

	base_mesh = apply_connectivity(base_mesh, seed_point = annotation_meshes[0].points[0])	

	# decimated = base_mesh.decimate_pro(.8, preserve_topology=True)
	# print(decimated.points.shape)

	# plot
	p = pv.Plotter(shape=(1,2))
	p.add_mesh(base_mesh, opacity=0.4, color=True, show_edges=True)
	for mesh in annotation_meshes: 
		p.add_mesh(mesh)

	p.subplot(0, 1)
	p.add_mesh(test_mesh, opacity=.4, color=True, show_edges=True)

	p.show()

	corresponding_points, corresponding_indices = annotation_onto_meshes(annotation_meshes, base_mesh, [test_mesh], 
														average_annotation_points=True, decimation_factor=.8, visualize_reg=visualize_reg)

	print(corresponding_indices)
	print(corresponding_points)

	facial_path = "RT 147 facial nerve mesh.vtk"
	facial_mesh = read_vtk(os.path.join(base, mesh_folder, facial_path))

	p = pv.Plotter()
	p.add_mesh(test_mesh, opacity=0.4, color=True)
	p.add_mesh(facial_mesh, opacity=.4, color=True)
	for annotation in corresponding_points: 
		for point in annotation:
			p.add_points(point, point_size=30, color='black')

	p.show()


if __name__ == "__main__":
	main()