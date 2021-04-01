"""tests.py
	0: malleus lateral 
	1: malleus manubrium
	2: incus short 
	3: incus long
	4: incus diam
	5: facial labryinthine
	6: facial 1st genu
	7: facial 2nd genu
	8: facial mastoid
	9: facial mastoid segment
	10: chorda tip
	11: chorda base 
	12: EAC point 
"""
import pyvista
import numpy as np
import nrrd
import pandas as pd

from utils.metrics import *
from utils.mesh_ops import *
from utils.file_io import *
from utils.centerline import *
from propagate_segments import adjust_file_path


def test_incus_angle(seg_data, seg_header, ann_data, ann_header, cached=None, prefix=None, plot=False):

	# read in the mesh for 153 incus
	incus = obtain_surface_mesh(seg_data, seg_header, 2, check_if_cached=cached, prefix=prefix, write_surface=True)

	# find the 153 incus short process landmark
	incus_short_mesh = obtain_surface_mesh(ann_data, ann_header, idx=2, check_if_cached=cached, prefix=prefix, write_surface=True)  # 2 is incus short process
	incus_short_point = get_centroid(incus_short_mesh)
	print('incus short point: {}'.format(incus_short_point))

	# find the 153 incus long process landmark
	incus_long_mesh = obtain_surface_mesh(ann_data, ann_header, idx=3, check_if_cached=cached, prefix=prefix, write_surface=True)  # 3 is incus long process
	incus_long_point = get_centroid(incus_long_mesh)
	print('incus long point: {}'.format(incus_long_point))

	# find the incus centroid
	incus_centroid = get_centroid(incus)
	print('incus centroid: {}'.format(incus_centroid))

	# calculate incus angle
	angle = calc_point_angle(incus_centroid, incus_short_point, incus_long_point)
	print('incus angle: {} ({} degrees)'.format(angle, angle*180/np.pi))

	# plot
	if plot:
		p = pv.Plotter()

		p.add_mesh(incus, opacity=0.7, color=True)
		p.add_mesh(incus_short_mesh, color=True)
		p.add_mesh(incus_long_mesh, color=True)
		p.add_points(incus_centroid, point_size=30, color='black')

		p.add_lines(np.array([incus_centroid, incus_short_point]), color='red')
		p.add_lines(np.array([incus_centroid, incus_long_point]), color='red')
		p.show()

	return [angle, angle*180/np.pi]


def test_facial_angles(seg_data, seg_header, ann_data, ann_header, cached=None, prefix=None, plot=False):

	# read in the mesh for 153 facial nerve
	facial = obtain_surface_mesh(seg_data, seg_header, 9, check_if_cached=cached, prefix=prefix, write_surface=True)

	facial_landmarks = obtain_surface_mesh(ann_data, ann_header, idx=[5, 6, 7, 8], check_if_cached=cached, prefix=prefix, write_surface=True)

	# find the facial nerve landmark points
	facial_lab = get_centroid(facial_landmarks[0])
	facial_genu1 = get_centroid(facial_landmarks[1])
	facial_genu2 = get_centroid(facial_landmarks[2])
	facial_mastoid = get_centroid(facial_landmarks[3])

	# calculate genu angles
	angle_genu1 = calc_point_angle(facial_genu1, facial_lab, facial_genu2)
	angle_genu2 = calc_point_angle(facial_genu2, facial_genu1 , facial_mastoid)

	print('1st genu angle: {} ({} degrees)'.format(angle_genu1, angle_genu1*180/np.pi))
	print('2nd genu angle: {} ({} degrees)'.format(angle_genu2, angle_genu2*180/np.pi))

	# plot
	if plot:
		p = pv.Plotter()

		p.add_mesh(facial, opacity=0.7, color=True)
		p.add_points(facial_lab, point_size=30, color='black')
		p.add_points(facial_genu1, point_size=30, color='black')
		p.add_points(facial_genu2, point_size=30, color='black')
		p.add_points(facial_mastoid, point_size=30, color='black')

		p.add_lines(np.array([facial_lab, facial_genu1]), color='red')
		p.add_lines(np.array([facial_genu1, facial_genu2]), color='red')
		p.add_lines(np.array([facial_genu2, facial_mastoid]), color='red')
		p.show()

	return [angle_genu1*180/np.pi, angle_genu2*180/np.pi]

def test_facial_recess(seg_data, seg_header, cached=None, prefix=None, plot=False):

	# read in the mesh for 153 facial nerve
	facial_path = adjust_file_path(seg_dir, "%s %s malleus mesh" % (side, target), ".vtk")
	facial = obtain_surface_mesh(seg_data, seg_header, 9, check_if_cached=cached, prefix=prefix, write_surface=True)
	# read in the mesh for 153 chorda tympani
	chorda = obtain_surface_mesh(seg_data, seg_header, 10, check_if_cached=cached, prefix=prefix, write_surface=True)

	facial_mastoid_segment = obtain_surface_mesh(ann_data, ann_header, 9, check_if_cached=cached, prefix=prefix, write_surface=True)
	chorda_landmarks = obtain_surface_mesh(ann_data, ann_header, [10, 11], check_if_cached=cached, prefix=prefix, write_surface=True) # tip, base
	chorda_tip = get_centroid(chorda_landmarks[0])
	chorda_base = get_centroid(chorda_landmarks[1])

	facial_mastoid_segment_tree = construct_kd_tree(facial_mastoid_segment)
	dd, ii = closest_points(chorda_tip, facial.points, use_tree=facial_mastoid_segment_tree)

	facial_mastoid_segment_point = facial_mastoid_segment.points[ii]

	recess_angle = calc_point_angle(chorda_base, chorda_tip, facial_mastoid_segment_point)
	recess_area = 1/2*np.sin(recess_angle)*np.linalg.norm(chorda_tip-chorda_base)*np.linalg.norm(facial_mastoid_segment_point-chorda_base)

	# plot
	if plot:
		p = pv.Plotter()

		p.add_mesh(facial, opacity=0.7, color=True)
		p.add_mesh(chorda, opacity=0.7, color=True)
		p.add_points(chorda_tip, point_size=30, color='black')
		p.add_points(chorda_base, point_size=30, color='black')
		p.add_points(facial_mastoid_segment_point, point_size=30, color='black')

		p.add_lines(np.array([chorda_tip, chorda_base]), color='red')
		p.add_lines(np.array([facial_mastoid_segment_point, chorda_base]), color='red')
		p.add_lines(np.array([chorda_tip, facial_mastoid_segment_point]), color='red')
		p.show()

	return [recess_angle*180/np.pi, recess_area, dd]


def test_ossicle_calculations(side, target, ann_mesh_dir=None, plot=False):
	# read in the incus and malleus
	malleus_path = adjust_file_path(ann_mesh_dir, "%s %s malleus mesh" % (side, target), ".vtk", registration=None)
	malleus, malleus_landmarks = return_mesh_with_annotations(malleus_path, segment="malleus")

	incus_path = adjust_file_path(ann_mesh_dir, "%s %s incus mesh" % (side, target), ".vtk", registration=None)
	incus, incus_landmarks = return_mesh_with_annotations(incus_path, segment="incus")

	malleus_lateral = malleus_landmarks["Process Tip"]
	malleus_manubrium = malleus_landmarks['Manubrium Tip']

	incus_short = incus_landmarks['Short Process Tip']
	incus_long = incus_landmarks['Long Process Tip']

	malleus_centroid = get_centroid(malleus)
	malleus_short_long_length = np.linalg.norm(malleus_lateral - malleus_manubrium)
	if plot:
		p = pv.Plotter()
		p.add_mesh(malleus, opacity=.4, color=True)
		p.add_points(malleus_lateral, point_size=30, color='black')
		p.add_points(malleus_manubrium, point_size=30, color='black')

		p.add_lines(np.array([malleus_lateral, malleus_centroid]), color='red')
		p.add_lines(np.array([malleus_manubrium, malleus_centroid]), color='red')
		p.add_lines(np.array([malleus_lateral, malleus_manubrium]), color='blue')
		p.show()


	incus_centroid = get_centroid(incus)
	incus_short_dist = np.linalg.norm(incus_centroid - incus_short)
	incus_long_dist = np.linalg.norm(incus_centroid - incus_long)
	incus_angle = calc_point_angle(incus_centroid, incus_short, incus_long)
	if plot:
		p = pv.Plotter()
		p.add_mesh(incus, opacity=.4, color=True)
		p.add_points(incus_short, point_size=30, color='black')
		p.add_points(incus_long, point_size=30, color='black')

		p.add_lines(np.array([incus_short, incus_centroid]), color='red')
		p.add_lines(np.array([incus_long, incus_centroid]), color='red')
		p.add_lines(np.array([incus_short, incus_long]), color='blue')
		p.show()


	print('malleus short long length', malleus_short_long_length)
	print('incus_angle', incus_angle*(180./np.pi))
	print('incus short dist', incus_short_dist)
	print('incus long dist', incus_long_dist)

	return [malleus_short_long_length, incus_short_dist, incus_long_dist, incus_angle*180/np.pi]


if __name__ == "__main__": 
	side = 'RT'
	template = '153'
	include_other_side = False
	write = True
	downsample = True
	downsample_size = 300
	base = '/Volumes/Extreme SSD/ANTs-registration'
	seg_dir = os.path.join(base, 'segmentations')
	ann_dir = os.path.join(base, 'annotations')
	pred_dir = os.path.join(base, 'predictions')
	ann_mesh_dir = os.path.join(base, 'annotated_meshes')

	RT = [  '138',
			'142',
			'143',
			'144',
			'146',
			'147',
			'152',
		]
	LT = []

	if side == 'RT':
		scan_id = RT
		opposite_scan_id = LT
		other_side = 'LT'
	else:
		scan_id = LT
		opposite_scan_id = RT
		other_side = 'RT'

	output_dict = dict()
	output_dict['scan'] = []
	output_dict['malleus manubrium length'] = []
	output_dict['incus short process length'] = []
	output_dict['incus long process length'] = []
	output_dict['incus angle'] = []
	# output_dict['facial nerve genu1 angle'] = []
	# output_dict['facial nerve genu2 angle'] = []
	# output_dict['facial recess angle'] = []
	# output_dict['facial recess area'] = []
	# output_dict['facial recess span'] = []
	# output_dict['eac-dura distance'] = []

	for target in scan_id:
		if target == template: continue

		# seg_path = adjust_file_path(pred_dir, "%s %s %s"%(side, template, target), ".seg.nrrd", downsample=downsample)
		# ann_path = adjust_file_path(pred_dir, "%s %s %s"%(side, template, target), ".seg.nrrd", downsample=downsample, is_annotation=True)

		print('reading segmentations header')
		# seg_header = nrrd.read_header(seg_path)
		
		ossicles = test_ossicle_calculations(side, target, ann_mesh_dir=ann_mesh_dir, plot=True)
		# facial = test_facial_angles(side, target, cached=pred_dir, plot=True)
		# recess = test_facial_recess(seg_header, side, target, seg_dir=seg_dir, ann_mesh_dir=ann_mesh_dir, plot=True)
		# eac_dura = test_eac_dura_distance(seg_data, seg_header, ann_data, ann_header, cached='ssm_out', prefix="{}_{}_{}".format(side, template, target))

		print('malleus manubrium length: {}'.format(ossicles[0]))
		print('incus short process length: {}'.format(ossicles[1]))
		print('incus long process length: {}'.format(ossicles[2]))
		print('incus angle: {}'.format(ossicles[3]))
		# print('facial nerve angles: {}'.format(facial))
		# print('facial recess angle: {}'.format(recess[0]))
		# print('facial recess area: {}'.format(recess[1]))
		# print('facial recess span: {}'.format(recess[2]))
		# print('eac-dura distance: {}'.format(eac_dura))

		output_dict['scan'].append(side + ' ' + target)
		output_dict['malleus manubrium length'].append(ossicles[0])
		output_dict['incus short process length'].append(ossicles[1])
		output_dict['incus long process length'].append(ossicles[2])
		output_dict['incus angle'].append(ossicles[3])
		# output_dict['facial nerve genu1 angle'].append(facial[0])
		# output_dict['facial nerve genu2 angle'].append(facial[1])
		# output_dict['facial recess angle'].append(recess[0])
		# output_dict['facial recess area'].append(recess[1])
		# output_dict['facial recess span'].append(recess[2])
		# output_dict['eac-dura distance'].append(eac_dura)	

	if write:
		write_to_file('Test Groundtruth Ossicles Dictionary', output_dict)
		output_df = pd.DataFrame(output_dict).set_index('scan')
		write_to_file('Test Groundtruth Ossicles Dataframe', output_df)
		output_df_path = os.path.join(os.getcwd(), "pickles/" + "Test Groundtruth Ossicles" + ".csv")
		output_df.to_csv(output_df_path)
