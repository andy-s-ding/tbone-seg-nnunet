function buildTransformSSM_test()

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

% Test split: ['RT_143', 'LT_148', 'LT_171', 'LT_147', 'LT_143', 'RT_147']

side_files = dir(fullfile(transform_dir, sprintf('%s %d *-inverse-downsample%d.nii.gz', side, template, downsample_size)));
other_side_files = dir(fullfile(transform_dir, sprintf('%s %d *-inverse-downsample%d-flipped.nii.gz', side, template, downsample_size)));

% Exclude any datasets in the test set
exclude_side = ["143", "147"];
exclude_other_side = ["148", "171", "147", "143"];
side_names = string(vertcat(side_files(:).name));
other_side_names = string(vertcat(other_side_files(:).name));
side_names = exclude_indices(side_names, exclude_side);
other_side_names = exclude_indices(other_side_names, exclude_side);

num_transforms = length(side_names) + length(other_side_names);
vectors = zeros(downsample_size^3*3,num_transforms);

for i=1:length(side_names)
    if i==1
        deform_meta = load_nii(char(fullfile(transform_dir, side_names(i))));
    end
    curr_transform = niftiread(fullfile(transform_dir, side_names(i)));
    vectors(:,i) = reshape(curr_transform, [numel(curr_transform), 1]);
end
for i=1:length(other_side_names)
    curr_transform = niftiread(fullfile(transform_dir, other_side_names(i)));
    vectors(:,length(side_names)+i) = reshape(curr_transform, [numel(curr_transform), 1]);
end

[coeff,score,latent] = pca(vectors');
pcaMean = mean(vectors,2);

transform_H5_path = fullfile(ssm_H5_transform_dir, sprintf('%s %d inverse-downsample%d.h5', side, template, downsample_size));
writeTransformH5(transform_H5_path, pcaMean, coeff, latent);

transform_meta_path = fullfile(ssm_meta_dir, sprintf('%s %d inverse-downsample%d.mat', side, template, downsample_size));
save(transform_meta_path, '-struct', 'deform_meta');

end

function names = exclude_indices(names, exclude_side)
    excluded_indices = [];
    for i=1:length(names(:,1))
        curr_name_array = split(names(i,:));
        curr_name = split(curr_name_array(3), '-');
        curr_index = string(cell2mat(curr_name(1)));
        excluded = any(exclude_side == curr_index);
        if excluded
            excluded_indices = [i, excluded_indices];
        end
    end
    for i=1:length(excluded_indices)
        index = excluded_indices(i);
        names(index,:) = [];
    end
end