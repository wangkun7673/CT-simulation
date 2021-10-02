#ifndef _GEO_H_
#define _GEO_H_

struct Geometry
{
    int bSysType;// scanning mode, cone-beam or fan-beam, default value is 1 which means cone-beam scanning mode
    float fDso;// distance from x-ray source to object
    float fDsd;// distance from x-ray source to detector
    int nAngle;// the amount of rotation angle
    int nDetU;// the resolution of detector in u direction
    int nDetV;// the resolution of detector in v direction
    float fDetUnit;// the pixel size of detector
    int nVoxelX;// the resolution in x direction in x direction
    int nVoxelY;// the resolution in y direction in y direction
    int nVoxelZ;// the resolution in z direction in z direction
    float fVoxel;// the voxel size of reconstructed volume
    int bGPU;// whether support Nvidia gpu and only Nvidia gpu
};

struct Point3D
{
	float x;
	float y;
	float z;
};

#endif