"""
mask.py
"""
import ants 
import numpy as np
import nrrd


def flip_image(ants_image, axis=0, single_components=False): 
	data = ants_image.numpy(single_components=single_components)
	flipped_data = np.flip(data, axis=axis)
	if flipped_data.dtype == 'int16':
		flipped_data = flipped_data.astype('float32')
		
	return ants_image.new_image_like(flipped_data)


def get_bounding_box(np_array, as_string=True): 

	print("-- getting bounding boxes")

	if len(np_array.shape) == 3: 
		np_array = [np_array]

	bounding_boxes = []
	for idx, array in enumerate(np_array): 
		print("-- bounding box for segment %d" % idx)
		x = np.any(array, axis=(1, 2))
		y = np.any(array, axis=(0, 2))
		z = np.any(array, axis=(0, 1))

		xmin, xmax = np.where(x)[0][[0, -1]]
		ymin, ymax = np.where(y)[0][[0, -1]]
		zmin, zmax = np.where(z)[0][[0, -1]]

		bounds = "%d %d %d %d %d %d" % (xmin, xmax, ymin, ymax, zmin, zmax)
		print(bounds)
		bounding_boxes.append(bounds)

	return bounding_boxes


def create_mask(mask_file): 
	"""create a mask from the provided segmentation nrrd file. 

	Expect a file path corresponding to tensor of shape (x_dim, y_dim, z_dim, n_channel) 
	where n_channel dimension corresponds to one hot vector encoding of segmentation 

	Checks if segments are overlapping during the creation of the mask file
	
	Args:
	    mask_file (str): provide the path to the file that will be used to create a mask
	
	Returns:
	    ANTsImage: the output mask 
	"""
	mask_image_raw = ants.image_read(mask_file)
	mask_image_np = mask_image_raw.numpy()
	mask_image_compressed = np.sum(mask_image_np, axis=-1)

	if np.any(mask_image_compressed > 1): 
		print('segments were overlapping during creation of mask image')
		print(mask_image_compressed[mask_image_compressed > 1])
		mask_image_compressed[mask_image_compressed > 1] == 1

	mask_image_new = ants.from_numpy(mask_image_compressed, 
									spacing=mask_image_raw.spacing, 
									origin=mask_image_raw.origin,
									direction=mask_image_raw.direction,
									has_components=False
									)

	return mask_image_new


def invert_mask(mask_image): 

	mask_raw = mask_image.numpy()
	temp = np.ones(mask_raw.shape)
	temp -= mask_raw

	return mask_image.new_image_like(temp)


def main(): 
	"""test the create_mask() function
	
	Returns:
	    None: does not return anything
	"""
	# mask_file = "/Users/alex/Documents/segmentations/Segmentation RT 146.seg.nrrd"
	# create_mask(mask_file)

	data, header = nrrd.read("/Users/alex/Documents/segmentations/Segmentation RT 153.seg.nrrd")
	bboxes = get_bounding_box(data)

	for idx, extent_str in enumerate(bboxes): 
		print(extent_str)
		print(header['Segment%d_Extent'%(idx)])

	return


if __name__ == "__main__": 
	main()
