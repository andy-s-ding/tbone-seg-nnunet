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


def test_closest_points(): 

	# read in mesh 1 
	mesh1 = read_vtk("ann_mesh_dir/RT 153 Facial Nerve mesh.vtk")

	# read in mesh 2 
	mesh2 = read_vtk("ann_mesh_dir/RT 153 Chorda Tympani mesh.vtk")

	# get surface points of each 
	surface1 = return_surface_points(mesh1)
	surface2 = return_surface_points(mesh2)

	print(surface1.shape)
	print(surface2.shape)

	# build a KD tree using mesh 1 
	tree = construct_kd_tree(mesh1)

	# test pickling the KD tree 
	write_to_file('test_tree', tree)

	# query the KD tree 
	dd, ii = closest_points(surface2[-1], surface1, use_tree=tree)

	print(dd, ii)

	p = pv.Plotter()
	p.add_mesh(mesh1, color=True)
	p.add_mesh(mesh2, color=True)
	p.add_points(surface2[-1], point_size=30)
	p.add_points(surface1[ii], point_size=30)

	p.show()

	# compare the result of querying the KD tree against brute force calculation
	dd, ii = closest_points(surface2[-1], surface1)

	print(dd, ii)
	print(dd.shape, ii.shape)

	return


def test_eac_dura_distance(seg_data, seg_header, ann_data, ann_header, cached=None, prefix=None, plot=False):

	# read in the mesh for 153 dura 
	dura = obtain_surface_mesh(seg_data, seg_header, 12, check_if_cached=cached, prefix=prefix, write_surface=True)
	eac = obtain_surface_mesh(seg_data, seg_header, 15, check_if_cached=cached, prefix=prefix, write_surface=True)

	# read in the annotations for 153 and extract the EAC segment 
	# data_eac, header_eac = nrrd.read("/Users/alex/Documents/segmentations/annotated/Annotations RT 153.seg.nrrd")
	# segment_indices = get_label_values(header_eac)
	# data_eac = convert_to_one_hot(data_eac, segment_indices)
	
	# mesh the 153 EAC landmark 
	# eac_mesh = obtain_surface_mesh(data_eac, header_eac, idx=11) # 11 is eac point 
	eac_mesh = obtain_surface_mesh(ann_data, ann_header, 12, check_if_cached=cached, prefix=prefix, write_surface=True)
	eac_landmarks = return_surface_points(eac_mesh)

	# calculate distances
	dd, ii = calculate_EAC_dura(dura, eac, eac_landmarks, plot=plot)

	print("distance", dd)

	print("min distance", np.min(dd))

	return np.min(dd)


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

def test_facial_recess(seg_data, seg_header, ann_data, ann_header, cached=None, prefix=None, plot=False):

	# read in the mesh for 153 facial nerve
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


def test_facial_interpolation(seg_data, seg_header, ann_data, ann_header, cached=None, prefix=None):
	# read in the mesh for 153 facial nerve
	facial = read_vtk("ann_mesh_dir/RT 153 Facial Nerve mesh.vtk")

	returned_points = surface_interpolation_1D(return_surface_points(facial), visualize=True)
	p = pv.Plotter()
	p.add_mesh(facial, opacity=.5, color=True)
	p.add_points(returned_points, point_size=10, color='black')
	p.show()

def test_ossicle_calculations(seg_data, seg_header, ann_data, ann_header, cached=None, prefix=None, plot=False):
	# read in the mesh for 153 incus and malleus
	malleus = obtain_surface_mesh(seg_data, seg_header, 1, check_if_cached=cached, prefix=prefix, write_surface=True)
	incus = obtain_surface_mesh(seg_data, seg_header, 2, check_if_cached=cached, prefix=prefix, write_surface=True)
	
	# mesh the 153 ossicle landmarks
	meshes = obtain_surface_mesh(ann_data, ann_header, idx=[0, 1, 2, 3], check_if_cached=cached, prefix=prefix, write_surface=True) 

	malleus_lateral_mesh, malleus_manubrium_mesh = meshes[0], meshes[1]
	incus_short_mesh, incus_long_mesh = meshes[2], meshes[3]

	_, _, _, malleus_short_long_length = calc_intra_ossicle_metrics(malleus, malleus_lateral_mesh, malleus_manubrium_mesh, plot=plot)

	incus_angle, incus_short_dist, incus_long_dist, _ = calc_intra_ossicle_metrics(incus, incus_short_mesh, incus_long_mesh, plot=plot)

	print('malleus short long length', malleus_short_long_length)
	print('incus_angle', incus_angle*(180./np.pi))
	print('incus short dist', incus_short_dist)
	print('incus long dist', incus_long_dist)

	return [malleus_short_long_length, incus_short_dist, incus_long_dist, incus_angle*180/np.pi]


if __name__ == "__main__": 
	side = 'RT'
	template = '153'
	downsample = True
	downsample_size = 300
	include_other_side = True
	write = True
	base = '/media/andyding/EXTREME SSD/ANTs-registration'
	seg_dir = os.path.join(base, 'segmentations')
	ann_dir = os.path.join(base, 'annotations')
	pred_dir = os.path.join(base, 'predictions')
	ann_mesh_dir = os.path.join(base, 'annotation_meshes')

	RT = [	'138',
            '142',
            '143',
            '144',
            '146',
            '147',
            '152',
            '153',
            '168',
            '170',
            '172',
            '174',
            '175',
            '177',
            '179',
            '181',
            '183',
            '184',
            '186',
            '187',
            '189',
            '191',
            '192',
            '194',
            '195'
        ]
	LT = [	'138',
            '143',
            '144',
            '145',
            '146',
            '147',
            '148',
            '151',
            '152',
            '168',
            '169',
            '170',
            '171',
            '172',
            '173',
            '175',
            '176',
            '177',
            '181',
            '183',
            '184',
            '185',
            '186',
            '191',
            '192',
            '193',
            '194',
            '195'
        ]

	if side == 'RT':
		scan_id = RT
		opposite_scan_id = LT
		other_side = 'LT'
	else:
		scan_id = LT
		opposite_scan_id = RT
		other_side = 'RT'

	print('reading segmentations')
	seg_data, seg_header = nrrd.read(os.path.join(seg_dir, 'Segmentation RT 153.seg.nrrd'))
	seg_data = convert_to_one_hot(seg_data, seg_header)
	print('reading annotations')
	ann_data, ann_header = nrrd.read(os.path.join(ann_dir, 'Annotations RT 153.seg.nrrd'))
	ann_data = convert_to_one_hot(ann_data, ann_header)
	
	ossicles = test_ossicle_calculations(seg_data, seg_header, ann_data, ann_header, cached=ann_mesh_dir, prefix='RT 153')
	facial = test_facial_angles(seg_data, seg_header, ann_data, ann_header, cached=ann_mesh_dir, prefix='RT 153')
	recess = test_facial_recess(seg_data, seg_header, ann_data, ann_header, cached=ann_mesh_dir, prefix='RT 153', plot=False)
	eac_dura = test_eac_dura_distance(seg_data, seg_header, ann_data, ann_header, cached=ann_mesh_dir, prefix='RT 153')

	print('malleus manubrium length: {}'.format(ossicles[0]))
	print('incus short process length: {}'.format(ossicles[1]))
	print('incus long process length: {}'.format(ossicles[2]))
	print('incus angle: {}'.format(ossicles[3]))
	print('facial nerve angles: {}'.format(facial))
	print('facial recess angle: {}'.format(recess[0]))
	print('facial recess area: {}'.format(recess[1]))
	print('facial recess span: {}'.format(recess[2]))
	print('eac-dura distance: {}'.format(eac_dura))

	output_dict = dict()
	output_dict['scan'] = [side + ' ' + template]
	output_dict['malleus manubrium length'] = [ossicles[0]]
	output_dict['incus short process length'] = [ossicles[1]]
	output_dict['incus long process length'] = [ossicles[2]]
	output_dict['incus angle'] = [ossicles[3]]
	output_dict['facial nerve genu1 angle'] = [facial[0]]
	output_dict['facial nerve genu2 angle'] = [facial[1]]
	output_dict['facial recess angle'] = [recess[0]]
	output_dict['facial recess area'] = [recess[1]]
	output_dict['facial recess span'] = [recess[2]]
	output_dict['eac-dura distance'] = [eac_dura]

	# output_dict = dict()
	# output_dict['scan'] = []
	# output_dict['malleus manubrium length'] = []
	# output_dict['incus short process length'] = []
	# output_dict['incus long process length'] = []
	# output_dict['incus angle'] = []
	# output_dict['facial nerve genu1 angle'] = []
	# output_dict['facial nerve genu2 angle'] = []
	# output_dict['facial recess angle'] = []
	# output_dict['facial recess area'] = []
	# output_dict['facial recess span'] = []
	# output_dict['eac-dura distance'] = []

	for target in scan_id:
		if target == template: continue

		seg_path = adjust_file_path(pred_dir, "Segmentation %s %s %s"%(side, template, target), ".seg.nrrd", downsample=downsample)
		ann_path = adjust_file_path(pred_dir, "%s %s %s"%(side, template, target), ".seg.nrrd", downsample=downsample, is_annotation=True)

		print('reading segmentations')
		seg_data, seg_header = nrrd.read(seg_path)
		seg_data = convert_to_one_hot(seg_data, seg_header)
		print('reading annotations')
		ann_data, ann_header = nrrd.read(ann_path)
		ann_data = convert_to_one_hot(ann_data, ann_header)
		
		ossicles = test_ossicle_calculations(seg_data, seg_header, ann_data, ann_header, cached=ann_mesh_dir, prefix="{}_{}_{}".format(side, template, target))
		facial = test_facial_angles(seg_data, seg_header, ann_data, ann_header, cached=ann_mesh_dir, prefix="{}_{}_{}".format(side, template, target))
		recess = test_facial_recess(seg_data, seg_header, ann_data, ann_header, cached=ann_mesh_dir, prefix="{}_{}_{}".format(side, template, target))
		eac_dura = test_eac_dura_distance(seg_data, seg_header, ann_data, ann_header, cached=ann_mesh_dir, prefix="{}_{}_{}".format(side, template, target))

		print('malleus manubrium length: {}'.format(ossicles[0]))
		print('incus short process length: {}'.format(ossicles[1]))
		print('incus long process length: {}'.format(ossicles[2]))
		print('incus angle: {}'.format(ossicles[3]))
		print('facial nerve angles: {}'.format(facial))
		print('facial recess angle: {}'.format(recess[0]))
		print('facial recess area: {}'.format(recess[1]))
		print('facial recess span: {}'.format(recess[2]))
		print('eac-dura distance: {}'.format(eac_dura))

		output_dict['scan'].append(side + ' ' + target)
		output_dict['malleus manubrium length'].append(ossicles[0])
		output_dict['incus short process length'].append(ossicles[1])
		output_dict['incus long process length'].append(ossicles[2])
		output_dict['incus angle'].append(ossicles[3])
		output_dict['facial nerve genu1 angle'].append(facial[0])
		output_dict['facial nerve genu2 angle'].append(facial[1])
		output_dict['facial recess angle'].append(recess[0])
		output_dict['facial recess area'].append(recess[1])
		output_dict['facial recess span'].append(recess[2])
		output_dict['eac-dura distance'].append(eac_dura)	

	if include_other_side:

		for target in opposite_scan_id:
			seg_path = adjust_file_path(pred_dir, "Segmentation %s %s %s"%(side, template, target), ".seg.nrrd", downsample=downsample, flip=True)
			ann_path = adjust_file_path(pred_dir, "%s %s %s"%(side, template, target), ".seg.nrrd", downsample=downsample, is_annotation=True, flip=True)

			print('reading segmentations')
			seg_data, seg_header = nrrd.read(seg_path)
			seg_data = convert_to_one_hot(seg_data, seg_header)
			print('reading annotations')
			ann_data, ann_header = nrrd.read(ann_path)
			ann_data = convert_to_one_hot(ann_data, ann_header)
			
			ossicles = test_ossicle_calculations(seg_data, seg_header, ann_data, ann_header, cached=ann_mesh_dir, prefix="{}_{}_{}_flipped".format(side, template, target))
			facial = test_facial_angles(seg_data, seg_header, ann_data, ann_header, cached=ann_mesh_dir, prefix="{}_{}_{}_flipped".format(side, template, target))
			recess = test_facial_recess(seg_data, seg_header, ann_data, ann_header, cached=ann_mesh_dir, prefix="{}_{}_{}_flipped".format(side, template, target))
			eac_dura = test_eac_dura_distance(seg_data, seg_header, ann_data, ann_header, cached=ann_mesh_dir, prefix="{}_{}_{}_flipped".format(side, template, target))

			print('malleus manubrium length: {}'.format(ossicles[0]))
			print('incus short process length: {}'.format(ossicles[1]))
			print('incus long process length: {}'.format(ossicles[2]))
			print('incus angle: {}'.format(ossicles[3]))
			print('facial nerve angles: {}'.format(facial))
			print('facial recess angle: {}'.format(recess[0]))
			print('facial recess area: {}'.format(recess[1]))
			print('facial recess span: {}'.format(recess[2]))
			print('eac-dura distance: {}'.format(eac_dura))

			output_dict['scan'].append(other_side + ' ' + target)
			output_dict['malleus manubrium length'].append(ossicles[0])
			output_dict['incus short process length'].append(ossicles[1])
			output_dict['incus long process length'].append(ossicles[2])
			output_dict['incus angle'].append(ossicles[3])
			output_dict['facial nerve genu1 angle'].append(facial[0])
			output_dict['facial nerve genu2 angle'].append(facial[1])
			output_dict['facial recess angle'].append(recess[0])
			output_dict['facial recess area'].append(recess[1])
			output_dict['facial recess span'].append(recess[2])
			output_dict['eac-dura distance'].append(eac_dura)

	if write:
		output_df = pd.DataFrame(output_dict).set_index('scan')
		output_df_path = os.path.join(ann_mesh_dir, "RT 153 Propagated 138-195" + ".csv")
		output_df.to_csv(output_df_path)
