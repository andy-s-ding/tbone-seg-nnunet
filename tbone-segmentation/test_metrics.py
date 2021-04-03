"""test_metrics.py

"""
import os 
import sys 
import numpy as np
import pyvista as pv
import argparse
import vtk
from vtk.util import numpy_support as nps

# from utils.mesh_ops import annotation_onto_meshes, apply_connectivity
from utils.file_io import read_vtk
from utils.metrics import calculate_EAC_dura

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
	parser.add_argument('--eac', 
						action="store",
						type=str,
						help="the eac mesh"
						)
	parser.add_argument('--dura',
						action="store",
						type=str,
						help="the dura mesh"
						)
	args = vars(parser.parse_args())
	return args


def main():

	# parse commandline
	args = parse_command_line(sys.argv)
	print(args)

	# initialize paths based on commandline arguments
	base = args['base']
	mesh_folder = args['mesh_cache']

	eac_mesh_path = os.path.join(base, mesh_folder, args['eac'])
	dura_mesh_path = os.path.join(base, mesh_folder, args['dura'])

	eac_mesh = read_vtk(eac_mesh_path)
	dura_mesh = read_vtk(dura_mesh_path)

	dd, ii = calculate_EAC_dura(dura_mesh, eac_mesh)

	print(np.min(dd))
	
	return


if __name__ == "__main__":
	main()