function writeTransformH5(filename,vmean,coeff,variance)
% Write SSM parameters into a .h5 file format
% vmean:    mean surface (3N*1)
% coeff:    PCA coefficients
% variance: PCA variance
% v:        example surface vertices matrix (N*3)
% f:        example surface faces matrix (?*3)

if exist(filename,'file')==2
    delete(filename);
end
h5create(filename,'/model/mean',length(vmean),'Datatype','single');
h5write(filename,'/model/mean',single(vmean));
h5create(filename,'/model/pcaBasis',size(coeff),'Datatype','single');
h5write(filename,'/model/pcaBasis',single(coeff));
h5create(filename,'/model/pcaVariance',length(variance),'Datatype','single');
h5write(filename,'/model/pcaVariance',single(variance));

h5create(filename,'/version/majorVersion',1,'Datatype','int32');
h5write(filename,'/version/majorVersion',0);
h5create(filename,'/version/minorVersion',1,'Datatype','int32');
h5write(filename,'/version/minorVersion',9);