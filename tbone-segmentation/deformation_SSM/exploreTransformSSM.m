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
downsample_size = 100;
template = 153;

transform_H5_path = fullfile(ssm_H5_transform_dir, sprintf('%s %d inverse-downsample%d.h5', side, template, downsample_size));

mu = h5read(transform_H5_path,'/model/mean');
P = h5read(transform_H5_path,'/model/pcaBasis');
P = P';
% size(P)

pcaVariance = h5read(transform_H5_path, '/model/pcaVariance')

cum_pca_var = ones(length(pcaVariance),1);
for i=1:length(pcaVariance)
    cum_pca_var(i) = sum(pcaVariance(1:i))/sum(pcaVariance);
end

figure;
plot(pcaVariance/sum(pcaVariance), 'marker', 'o');
set(gca, 'XTick', 1:length(pcaVariance));
title('Variance Explained by Deformation Field SSM Components')
xlabel('Principal Component Number') 
ylabel('Percentage of Variance Explained')
% saveas(gcf, pca_var_path, 'png');
hold off

figure;
plot(cum_pca_var, 'marker', 'o');
set(gca, 'XTick', 1:length(pcaVariance));
title('Cumulative Variance Explained by Deformation Field SSM Components')
xlabel('Principal Component Number') 
ylabel('Cumulative Percentage of Variance Explained')
% saveas(gcf, pca_cum_var_path, 'png');
hold off