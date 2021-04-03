function transformFromSSM(side, template, downsample_size, coeff, variance, pcaMean, meta, save_dir, id)

addpath('./NIfTI_20140122/');

deform_path = fullfile(save_dir, sprintf('%s %d deform%d-inverse-downsample%d.nii.gz', side, template, id, downsample_size));

weights = randn(length(variance),1);
deform = cast(reshape(pcaMean + coeff*(sqrt(variance).*weights), [downsample_size downsample_size downsample_size 1 3]), 'single');

deform_nii = make_nii(deform);
glmax = deform_nii.hdr.dime.glmax;
glmin = deform_nii.hdr.dime.glmin;
deform_nii.hdr = meta.hdr;
deform_nii.hdr.dime.glmax = glmax;
deform_nii.hdr.dime.glmin = glmin;
deform_nii.original = deform_nii.hdr;

save_nii(deform_nii, deform_path)

end