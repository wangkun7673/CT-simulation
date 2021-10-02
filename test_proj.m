% this program is used to test jacob_ray_projection.cu file
clear
clc
% system parameters setting
ParaSet;
% phantom generation
phan = single(phantom3dAniso('modified shepp-logan', param.nVoxelX));
% figure, imshow(phan(:,:,128), [])
projData = single(zeros(param.nDetV, param.nDetU, param.nAngle));
for ii=1:param.nAngle
    projData(:,:,ii) = Ax_mex(param, ii-1, phan);
end
figure, hold on
for ii=1:param.nAngle
    imshow(projData(:,:,ii), [])
    pause(0.3);
end