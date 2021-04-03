"""annotation_propagation.py

mesh to mesh based annotation propagation
"""
import os 
import sys 
import numpy as np
import pyvista as pv
import argparse
import vtk
from vtk.util import numpy_support as nps

from utils.mesh_ops import annotation_onto_meshes, apply_connectivity
from utils.file_io import read_vtk


def parse_command_line(args):
	'''

	'''

	print('parsing command line')

	parser = argparse.ArgumentParser(description='Registration pipeline for image-image registration')
	parser.add_argument('--base',
						action="store", 
						type=str, 
						default="/scratch/groups/rtaylor2/ANTs-registration"
						)
	parser.add_argument('--side', 
						action="store", 
						type=str, 
						default="RT"
						)
	parser.add_argument('--mesh_cache', '-m',
						action="store",
						default="ssm_out",
						)
	parser.add_argument('--source', 
						action="store",
						type=str,
						help="the mesh which should be registered onto other meshes"
						)
	parser.add_argument('--annotations', '-a',
						nargs='+',
						type=str,
						help="the annotation meshes corresponding to source mesh"
						)
	parser.add_argument('--targets', '-t',
						nargs='+',
						help="the target meshes that source should be propagated onto"
						)
	parser.add_argument('--out', 
						action="store", 
						type=str, 
						default="annotated_meshes"
						)
	parser.add_argument('--visualize_registration',
						action="store_true"
						)
	parser.add_argument('--hardcoded_options',
						action="store",
						type=str,
						)
	parser.add_argument('--decimation_factor',
						action="store",
						type=float,
						default=.8,
						help="extent to which meshes should be decimated.",
						)
	args = vars(parser.parse_args())
	return args


def hardcoded_options(segment, base, mesh_folder): 

	if 'facial' in segment.lower(): 
		mesh_path = "RT 153 Facial Nerve mesh.vtk"
		annotation_mesh_paths = [
			"RT 153 Facial Nerve 1st Genu mesh.vtk",
			"RT 153 Facial Nerve 2nd Genu mesh.vtk",
			"RT 153 Facial Nerve Labyrinthine mesh.vtk",
			"RT 153 Facial Nerve Mastoid mesh.vtk"
		]

	elif 'chorda' in segment.lower(): 
		mesh_path = "RT 153 Chorda Tympani mesh.vtk"
		annotation_mesh_paths = [
			"RT 153 Chorda Tympani Base mesh.vtk",
			"RT 153 Chorda Tympani Tip mesh.vtk"
		]

	elif 'malleus' in segment.lower(): 
		mesh_path = "RT 153 Malleus mesh.vtk"
		annotation_mesh_paths = [
			"RT 153 Malleus Lateral Process Tip mesh.vtk", 
			"RT 153 Malleus Manubrium Tip mesh.vtk",
		]

	elif 'EAC' in segment: 
		mesh_path = "RT 153 EAC mesh.vtk"
		annotation_mesh_paths = [
			"RT 153 EAC Superior mesh.vtk"
		]

	elif 'incus' in segment: 
		mesh_path = "RT 153 Incus mesh.vtk"
		annotation_mesh_paths = [
			"RT 153 Incus Long Process Tip mesh.vtk", 
			"RT 153 Incus Short Process Tip mesh.vtk"
		]

	return (os.path.join(base, mesh_folder, mesh_path), 
			[os.path.join(base, mesh_folder, mesh_path) for mesh_path in annotation_mesh_paths])


def main(): 

	# parse commandline
	args = parse_command_line(sys.argv)
	print(args)

	# initialize paths based on commandline arguments
	base = args['base']
	mesh_folder = args['mesh_cache']
	visualize_reg = args['visualize_registration']

	source_mesh_path = ""
	annotation_mesh_paths = []

	if args['hardcoded_options'] is not None: 
		source_mesh_path, annotation_mesh_paths = hardcoded_options(args['hardcoded_options'], base, mesh_folder)
	else: 
		source_mesh_path = os.path.join(base, mesh_folder, args['source'])
		annotation_mesh_paths = [os.path.join(base, mesh_folder, annotation) for annotation in args['annotations']]

	target_mesh_paths = [os.path.join(base, mesh_folder, target_mesh_path) for target_mesh_path in args['targets']]


	# read in annotation meshes 
	annotation_meshes = [read_vtk(mesh_path) for mesh_path in annotation_mesh_paths]

	# read in 
	source_mesh = read_vtk(source_mesh_path)
	target_meshes = [read_vtk(path) for path in target_mesh_paths]

	source_mesh = apply_connectivity(source_mesh, seed_point = annotation_meshes[0].points[0])	

	# plot
	# p = pv.Plotter()
	# p.add_mesh(source_mesh, opacity=0.4, color=True, show_edges=True)
	# for mesh in annotation_meshes: 
	# 	p.add_mesh(mesh)
	# p.show()

	corresponding_points, corresponding_indices = annotation_onto_meshes(annotation_meshes, source_mesh, target_meshes, 
														average_annotation_points=True, 
														decimation_factor=args['decimation_factor'], 
														visualize_reg=visualize_reg
													)

	# p = pv.Plotter()
	# p.add_mesh(target_meshes, opacity=0.4, color=True)
	# p.add_mesh(facial_mesh, opacity=.4, color=True)
	# for annotation in corresponding_points: 
	# 	for point in annotation:
	# 		p.add_points(point, point_size=30, color='black')

	# p.show()

	for mesh_path, target_mesh, grouped_indices in zip(target_mesh_paths, target_meshes, corresponding_indices): 
		for annotation_name, annotation_indices in zip(annotation_mesh_paths, grouped_indices):
			print(annotation_name.split()[-4:-1], annotation_indices)
			temp = np.zeros(len(target_mesh.points))
			temp[annotation_indices] = 1
			temp_vtk = nps.numpy_to_vtk(temp, deep=True)
			strs = annotation_name.split()
			array_name = " ".join(strs[-4:-1])
			temp_vtk.SetName(array_name)

			target_mesh.GetPointData().AddArray(temp_vtk)

		target_mesh.save(mesh_path.replace(args['mesh_cache'], args['out']))


if __name__ == "__main__":
	main()