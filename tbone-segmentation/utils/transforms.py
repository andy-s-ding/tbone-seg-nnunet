def apply_transform_to_image(transform, target_image, template_image, interpolator='genericLabel'): 
	"""apply_transform_to_image

	Apply transformations to an image. The image could either be a multichannel segmentation (0, 1) in each channel
	or a single channel segmentation (0: num indices)

	Transform should be in the order of an ants fwd transform, i.e. [deformation path, affine path]
	
	Args:
	    transform (list): list of file paths to transformations to be applied 
	    target_image (ANTsImage): the image with spatial metadata to be applied after transformation
	    template_image (ANTsImage): the image that will be transformed
	    interpolator (str, optional): the interpolation method
	
	Returns:
	    predicted_targets_image (ANTsImage): the transformed template image
	"""

	# directly apply the transformation if the channels do not have to be split
	if template_image.components == 1: 
		return ants.apply_transforms(target_image, template_image, transform, interpolator=interpolator, verbose=True)

	# split the channels and apply transformation to each individually 
	individual_segments = ants.split_channels(template_image)
	print(len(individual_segments))
	predicted_targets = []
	for i, segment in enumerate(individual_segments): 

		print("applying transformation to channel %d"%i)

		print(segment.components)
		print(segment.pixeltype)
		print(segment.shape)

		predicted_target = ants.apply_transforms(target_image, segment, transform, interpolator=interpolator, verbose=True)
		predicted_targets.append(predicted_target)

	predicted_targets_image = ants.merge_channels(predicted_targets)

	return predicted_targets_image