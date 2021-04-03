function generateSSMTransforms()

base_dir = '/Volumes/Extreme SSD/ANTs-registration';
image_dir = fullfile(base_dir, 'images');
downsampled_dir = fullfile(image_dir, 'downsampled');
transform_dir = fullfile(base_dir, 'transforms');
ssm_H5_transform_dir = fullfile(base_dir, 'ssm_H5_transforms');
ssm_transform_dir = fullfile(base_dir, 'ssm_transforms');
ssm_meta_dir = fullfile(ssm_transform_dir, 'meta');
if not(exist(ssm_transform_dir, 'dir'))
    fprintf('Creating SSM transforms folder\n')
    mkdir(base_dir, 'ssm_transforms');
end

num_transforms = 100;
side = "RT";
downsample_size = 60;
template = 153;

transform_meta_path = fullfile(ssm_meta_dir, sprintf('%s %d inverse-downsample%d.mat', side, template, downsample_size));
transform_H5_path = fullfile(ssm_H5_transform_dir, sprintf('%s %d inverse-downsample%d.h5', side, template, downsample_size));

pcaMean = h5read(transform_H5_path,'/model/mean');
coeff = h5read(transform_H5_path,'/model/pcaBasis');
variance = h5read(transform_H5_path, '/model/pcaVariance');
meta = load(transform_meta_path);

for id=1:num_transforms
    fprintf('Generating deform %d\n', id)
    transformFromSSM(side, template, downsample_size, coeff, variance, pcaMean, meta, ssm_transform_dir, id)
end
end