"""centerline_analysis.py


"""

import numpy as np
import vtk 
from vtk.util import numpy_support as nps
import pyvista as pv 
import sys 
import os 
import argparse 

from utils.centerline import *
from utils.mesh_ops import construct_kd_tree, closest_points, return_surface_points


def parse_command_line(args):

	print("parsing command line")

	parser = argparse.ArgumentParser(description="Registration pipeline for image-image registration")
	parser.add_argument("--base", 
						action="store", 
						type=str, 
						default="/scratch/groups/rtaylor2/ANTs-registration"
						)
	parser.add_argument("--side", 
						action="store", 
						type=str, 
						default="RT", 
						help="specify the anatomical side. used to identify paths, file names"
						)
	parser.add_argument("--centerline_path",
						action="store",
						type=str,
						default="predictions/centerlines",
						help="specify the directory where centerline files are located"
						)
	parser.add_argument("--mesh_path",
						action="store",
						type=str,
						default="ssm_out",
						help="specify the directory where meshes are located"
						)
	parser.add_argument("--images", 
						nargs="+", 
						metavar="id",
						type=str, 
						default="144",
						help="specify the scan IDs to be included in the analysis. used to identify paths, file names"
						)
	parser.add_argument("--propagated",
						type=str,
						help="if specified, analysis will be done over propagated segments"
						)
	parser.add_argument("--all", 
						action="store_true",
						help="if specified, all scan IDs will be included in the analysis." 
						)
	parser.add_argument("--incus",
						action="store_true",
						help="if specified, analysis will be done on incudes",
						)
	parser.add_argument("--facial",
						action="store_true",
						help="if specified, analysis will be done on incudes",
						)
	parser.add_argument("--chorda",
						action="store_true",
						help="if specified, analysis will be done on incudes",
						)
	parser.add_argument("--dry",
						action="store_true",
						help="if specified, calculations will be skipped. for debugging use. "
					   )

	args = vars(parser.parse_args())
	return args


def main(): 
	args = parse_command_line(sys.argv)
	print(args)

	base = args["base"]
	centerline_path = args["centerline_path"]
	side = args["side"]
	scans = args["images"]

	if args["all"]: 
		scans = [
			"138",
			"142", 
			"143",
			"144",
			"146",
			"147",
		]

	model = ""
	if args["incus"] is not None: 
		model = "incus"
	elif args["facial"] is not None: 
		model = "facial"
	elif args["chorda"] is not None: 
		model = "chorda"

	centerlines = [os.path.join(base, centerline_path, "%s %s %s centerline model.vtk" % (side, scan, model)) for scan in scans] 
	mesh_path = [os.path.join(args["mesh_path"], "%s %s %s mesh.vtk" % (side, scan, model)) for scan in scans]

	if args["propagated"] is not None: 
		centerlines = [os.path.join(base, centerline_path, "%s_%s_%s %s centerline model.vtk") % (side, args["propagated"], scan, model) 
						for scan in scans
						]
		meshes = [os.path.join(args["mesh_path"], "%s_%s_%s %s mesh.vtk" % (side, args["propagated"], scan, model.capitalize())) 
					for scan in scans
					] 

	else: 
		centerlines = [os.path.join(base, centerline_path, "%s %s %s centerline model.vtk") % (side, scan, model) 
						for scan in scans
						]

		meshes = [os.path.join(args["mesh_path"], "%s %s %s mesh.vtk" % (side, scan, model.capitalize())) 
					for scan in scans
					] 

	for centerline_path, mesh_path, scan in zip(centerlines, meshes, scans):
		print('--'*10)
		print(centerline_path)
		print(mesh_path)

		centerline = pv.read(centerline_path)
		kd_tree = construct_kd_tree(centerline)

		try:
			mesh = pv.read(mesh_path)
		except:
			mesh = pv.read(mesh_path.replace("Incus", "incus"))

		centerline_points = return_surface_points(centerline)
		print("the number of points in the centerline is:", len(centerline_points))
		print("rough direction is:", centerline_points[-1] - centerline_points[0])

		if centerline_points[-1, 2]-centerline_points[0,2] > 0: 
			print('reversing centerline')
			kd_tree = construct_kd_tree(centerline, reverse_points=True)

		mesh_points = return_surface_points(mesh)

		dd, ii, norm_pos = project_onto_centerline(centerline_points, mesh_points, kd_tree)

		plt.figure() 
		plt.scatter(norm_pos, dd)
		plt.xlabel('position along centerline')
		plt.ylabel('distance from wall to centerline')
		if args["propagated"] is not None:
			plt.title('Radius along centerline, %s_%s_%s %s' % (side, args["propagated"], scan, model.capitalize()))
		else: 
			plt.title('Radius along centerline, %s %s %s' % (side, scan, model.capitalize()))

		plt.savefig(os.path.join("figures", centerline_path.split("/")[-1].replace(" model.vtk", ".png")))
		# plt.show()

		distances_as_vtk_array = nps.numpy_to_vtk(dd, deep=True)
		pos_as_vtk_array = nps.numpy_to_vtk(norm_pos, deep=True)

		print(dd.shape)
		print(ii.shape)

		distances_as_vtk_array.SetName("dist to centerline")
		pos_as_vtk_array.SetName("pos along centerline")

		mesh.GetPointData().AddArray(distances_as_vtk_array)
		mesh.GetPointData().AddArray(pos_as_vtk_array)

		mesh.save(os.path.join("annotated_meshes", mesh_path.split("/")[-1].replace("mesh", "projected mesh")), binary=False)

		average_rad = np.mean(dd[np.argwhere((norm_pos > .7) & (norm_pos < .9))])

		print(scan, average_rad)

	return


if __name__ == "__main__": 
	main()