clear
clc
% Lpath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64';
% compile .cu file
system('nvcc -c jacob_ray_projection.cu')
% compile .cpp file with .obj
mex -g Ax_mex.cpp jacob_ray_projection.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64"