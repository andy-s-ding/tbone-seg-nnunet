"""rewrite_meshes.py

"""

import os 
import pyvista as pv 


def main(): 

	all_files = os.listdir("ssm_out")

	prop_meshes = [f for f in all_files if "RT_153_" in f]

	for mesh in prop_meshes: 
		print(mesh)
		temp = pv.read(os.path.join("ssm_out", mesh))

		temp.save(os.path.join("ssm_out", mesh), binary=False)

	return 


if __name__ == "__main__": 
	main()