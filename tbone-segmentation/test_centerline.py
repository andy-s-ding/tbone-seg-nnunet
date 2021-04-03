"""test_centerline.py

"""

import numpy as np
import vtk 
from vtk.util import numpy_support as nps
import pyvista as pv 

from utils.centerline import *
from utils.mesh_ops import construct_kd_tree, closest_points, return_surface_points


def main(): 

	base = "/Users/alex/Documents"
	centerline_path = "predictions/centerlines"
	side = "RT"
	scan = "153"
	model = "incus"

	centerline_path = os.path.join(base, centerline_path, "%s %s %s centerline model.vtk" % (side, scan, model))
	centerline = pv.read(centerline_path)
	kd_tree = construct_kd_tree(centerline)

	mesh_path = "ssm_out/RT 153 incus mesh.vtk"
	mesh = pv.read(mesh_path)

	centerline_points = return_surface_points(centerline)
	centerline_points = np.flip(centerline_points, axis=0)

	print(len(centerline_points))
	mesh_points = return_surface_points(mesh)

	dd, ii, norm_pos = project_onto_centerline(centerline_points, mesh_points, kd_tree)

	plt.figure() 
	plt.scatter(norm_pos, dd)
	plt.xlabel('position along centerline')
	plt.ylabel('distance from wall to centerline')
	plt.show()

	distances_as_vtk_array = nps.numpy_to_vtk(dd, deep=True)
	pos_as_vtk_array = nps.numpy_to_vtk(norm_pos, deep=True)

	print(dd.shape)
	print(ii.shape)

	distances_as_vtk_array.SetName("dist to centerline")
	pos_as_vtk_array.SetName("pos along centerline")

	mesh.GetPointData().AddArray(distances_as_vtk_array)
	mesh.GetPointData().AddArray(pos_as_vtk_array)

	mesh.save("annotated_meshes/%s %s %s projected mesh.vtk" % (side, scan, model))

	return


if __name__ == "__main__": 
	main()