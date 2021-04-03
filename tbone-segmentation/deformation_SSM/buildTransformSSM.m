function buildTransformSSM()

clear all;
addpath('./NIfTI_20140122/');

%% 0. Housekeeping
base_dir = '/Volumes/Extreme SSD/ANTs-registration';
image_dir = fullfile(base_dir, 'images');
downsampled_dir = fullfile(image_dir, 'downsampled');
transform_dir = fullfile(base_dir, 'transforms');
ssm_H5_transform_dir = fullfile(base_dir, 'ssm_H5_transforms');
ssm_transform_dir = fullfile(base_dir, 'ssm_transforms');
ssm_meta_dir = fullfile(ssm_transform_dir, 'meta');
if not(exist(ssm_H5_transform_dir, 'dir'))
    fprintf('Creating SSM H5 transforms folder\n')
    mkdir(base_dir, 'ssm_H5_transforms');
end
if not(exist(ssm_transform_dir, 'dir'))
    fprintf('Creating SSM transforms folder\n')
    mkdir(base_dir, 'ssm_transforms');
end
if not(exist(ssm_meta_dir, 'dir'))
    fprintf('Creating SSM transforms meta folder\n')
    mkdir(ssm_transform_dir, 'meta');
end

side = "RT";
downsample_size = 60;
template = 153;

side_files = dir(fullfile(transform_dir, sprintf('%s %d *-inverse-downsample%d.nii.gz', side, template, downsample_size)));
other_side_files = dir(fullfile(transform_dir, sprintf('%s %d *-inverse-downsample%d-flipped.nii.gz', side, template, downsample_size)));

num_transforms = length(side_files) + length(other_side_files);
vectors = zeros(downsample_size^3*3,num_transforms);

for i=1:length(side_files)
    if i==1
        name = fullfile(transform_dir, side_files(i).name);
        deform_meta = load_nii(fullfile(transform_dir, side_files(i).name));
    end
    curr_transform = niftiread(fullfile(transform_dir, side_files(i).name));
    vectors(:,i) = reshape(curr_transform, [numel(curr_transform), 1]);
end
for i=1:length(other_side_files)
    curr_transform = niftiread(fullfile(transform_dir, other_side_files(i).name));
    vectors(:,length(side_files)+i) = reshape(curr_transform, [numel(curr_transform), 1]);
end

[coeff,score,latent] = pca(vectors');
pcaMean = mean(vectors,2);

transform_H5_path = fullfile(ssm_H5_transform_dir, sprintf('%s %d inverse-downsample%d.h5', side, template, downsample_size));
writeTransformH5(transform_H5_path, pcaMean, coeff, latent);

transform_meta_path = fullfile(ssm_meta_dir, sprintf('%s %d inverse-downsample%d.mat', side, template, downsample_size));
save(transform_meta_path, '-struct', 'deform_meta');

end