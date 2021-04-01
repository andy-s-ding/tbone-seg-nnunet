import vtk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
    
name = "Andy"
# name = "Alex"
if name == "Andy":
    base_dir = '/Users/andyding/Desktop/Dyna CT Segmentations/Segmentations'
else: base_dir = '/Users/alex/Documents/Segmentations'

part = 'vestibule_+_cochlea'
side = 'LT' # 'RT' = right, 'LT' = left

if side == 'RT':
    index = [142, 143, 144, 146, 147, 150, 152, 153] # 138 was template
else: index = [143, 144, 145, 146, 147, 148, 150, 151, 152, 169, 171]

vtk_files = []
for i in range(len(index)):
    vtk_files.append(os.path.join(base_dir, 'surfaces', '{0}_{1}_{2}.vtk'.format(part, index[i], side)))
cpd_files = []
for i in range(len(index)):
    cpd_files.append(os.path.join(base_dir, 'surfaces',
                              '{0}_cpd_{1}_{2}.vtk'.format(part, index[i], side)))
def get_vtk_points(file):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllTensorsOn()
    reader.Update()
    vtkdata = reader.GetOutput()
    num_points = vtkdata.GetNumberOfPoints()
    xs = np.zeros(num_points)
    ys = np.zeros(num_points)
    zs = np.zeros(num_points)
    for i in range(num_points):
        x, y, z = vtkdata.GetPoint(i)
        xs[i] = x
        ys[i] = y
        zs[i] = z
    return xs, ys, zs

def plot_vtk_vs_cpd(vtk_files, cpd_files):
    assert(len(vtk_files) == len(cpd_files))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(vtk_files)):
        xs, ys, zs = get_vtk_points(vtk_files[i])
        ax.scatter(xs, ys, zs)
        xs, ys, zs = get_vtk_points(cpd_files[i])
        ax.scatter(xs, ys, zs)
        plt.show()

plot_vtk_vs_cpd(vtk_files, cpd_files)
