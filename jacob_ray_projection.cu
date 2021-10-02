#include <cuda_runtime_api.h>
#include <cuda.h>
#include <math.h>
#include "geometry.h"
#include "jacob_ray_projection.h"
#include "mex.h"

/*
#define cudaCheckErrors(msg)\
do {\
		cudaError_t __err = cudaGetLastError();\
		if(__err != cudaSuccess){\
			mexPrintf("%s\n", msg);\
			mexErrMsgIdAndTxt("CBCT:CUDA:Ax_mex", cudaGetErrorString(__err));
		}\
}while(0)
*/

// Declare the texture reference
texture<float, cudaTextureType3D, cudaReadModeElementType> tex_phan;


#define MAXTHREADS 1024
#define PI 3.14159265359

// void computeDeltas_Jacob(Geometry geo, Point3D* uvOrigin, Point3D* deltaU, Point3D* deltaV, Point3D* source, int angleIndex);
// void computeDeltas_Jacob(Geometry geo, Point3D* detectPixel, Point3D* source, int angleIndex);

__global__ void kernelPixelDetector(Geometry geo, float* dProjection_phan, Point3D* detectPixel, Point3D source)
{
	// Point3D source: the position of the x-ray source
	// Point* detectPixel: the array which stores the positions of detector units
	unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned long idx = y * geo.nDetU + x; // the index of detector unit
	
	if((x>=geo.nDetU)|(y>=geo.nDetV))
		return;
	
	// get the coordinate XYZ of pixel UV in the detector

	// vector of x-ray
	Point3D ray;
	ray.x = detectPixel[idx].x - source.x;
	ray.y = detectPixel[idx].y - source.y;
	ray.z = detectPixel[idx].z - source.z;

	float axm, aym, azm;
	float axM, ayM, azM;
	
	// the x planes, y planes and z planes
	float x_planes_1  = -(float)geo.nVoxelX / 2 * geo.fVoxel;
	float x_planes_Nx =  (float)geo.nVoxelX / 2 * geo.fVoxel;

	float y_planes_1  = -(float)geo.nVoxelY / 2 * geo.fVoxel;
	float y_planes_Ny =  (float)geo.nVoxelY / 2 * geo.fVoxel;
	
	float z_planes_1  = -(float)geo.nVoxelZ / 2 * geo.fVoxel;
	float z_planes_Nz =  (float)geo.nVoxelZ / 2 * geo.fVoxel;

	// alpha_min and alpha_max
	axm = min((x_planes_1 - source.x) / ray.x, (x_planes_Nx - source.x) / ray.x);
	aym = min((y_planes_1 - source.y) / ray.y, (y_planes_Ny - source.y) / ray.y);
	azm = min((z_planes_1 - source.z) / ray.z, (z_planes_Nz - source.z) / ray.z);
	axM = max((x_planes_1 - source.x) / ray.x, (x_planes_Nx - source.x) / ray.x);
	ayM = max((y_planes_1 - source.y) / ray.y, (y_planes_Ny - source.y) / ray.y);
	azM = max((z_planes_1 - source.z) / ray.z, (z_planes_Nz - source.z) / ray.z);

	float am = max(max(axm, aym), azm);// choose the maximum value from these alpha_min 
	float aM = min(min(axM, ayM), azM);// choose the minimum value from these alpha_max
	
	if(am>=aM)
	{
		dProjection_phan[idx] = 0.0;
	}
	
	// Nx = geo.nVoxelX + 1; Ny = geo.nVoxelY + 1; Nz = geo.nVoxelZ + 1;
	float imin, imax, jmin, jmax, kmin, kmax;
	// for x
	if(source.x < detectPixel[idx].x)
	{
		imin = (am==axm)? 2               : ceil ((source.x + am*ray.x - x_planes_1) / geo.fVoxel);
		imax = (aM==axM)? (geo.nVoxelX+1) : floor((source.x + aM*ray.x - x_planes_1) / geo.fVoxel);
	}
	else
	{
		imax = (am==axm)?  geo.nVoxelX    : floor((source.x + am*ray.x - x_planes_1) / geo.fVoxel);
		imin = (aM==axM)?  1              : ceil ((source.x + aM*ray.x - x_planes_1) / geo.fVoxel); 
	}
	// for y
	if(source.y < detectPixel[idx].y)
	{
		jmin = (am==aym)? 2               : ceil ((source.y + am*ray.y - y_planes_1) / geo.fVoxel);
		jmax = (aM==ayM)? (geo.nVoxelY+1) : floor((source.y + aM*ray.y - y_planes_1) / geo.fVoxel);
	}
	else
	{
		jmax = (am==aym)?  geo.nVoxelY    : floor((source.y + am*ray.y - y_planes_1) / geo.fVoxel);
		jmin = (aM==ayM)?  1              : ceil ((source.y + aM*ray.y - y_planes_1) / geo.fVoxel); 
	}
	// for z
	if(source.z < detectPixel[idx].z)
	{
		kmin = (am==azm)? 2               : ceil ((source.z + am*ray.z - z_planes_1) / geo.fVoxel);
		kmax = (aM==azM)? (geo.nVoxelZ+1) : floor((source.z + aM*ray.z - z_planes_1) / geo.fVoxel);
	}
	else
	{
		kmax = (am==azm)?  geo.nVoxelZ    : floor((source.z + am*ray.z - z_planes_1) / geo.fVoxel);
		kmin = (aM==azM)?  1              : ceil ((source.z + aM*ray.z - z_planes_1) / geo.fVoxel); 
	}

	float ax, ay, az;
	long double lambda = 0.000000000001;
	ax = (source.x < detectPixel[idx].x)? (imin*geo.fVoxel+x_planes_1 - source.x)/(ray.x + lambda) : (imax*geo.fVoxel+x_planes_1 - source.x)/(ray.x + lambda);
	ay = (source.y < detectPixel[idx].y)? (jmin*geo.fVoxel+y_planes_1 - source.y)/(ray.y + lambda) : (jmax*geo.fVoxel+y_planes_1 - source.y)/(ray.y + lambda);
	az = (source.z < detectPixel[idx].z)? (kmin*geo.fVoxel+z_planes_1 - source.z)/(ray.z + lambda) : (kmax*geo.fVoxel+z_planes_1 - source.z)/(ray.z + lambda);
	
	int i,j,k;
	float aminc = min(min(ax, ay), az);
	i = (int)floor((source.x + (aminc + am)/2 * ray.x - x_planes_1) / geo.fVoxel);
	j = (int)floor((source.y + (aminc + am)/2 * ray.y - y_planes_1) / geo.fVoxel);
	k = (int)floor((source.z + (aminc + am)/2 * ray.z - z_planes_1) / geo.fVoxel);
	
	// initialize
	float ac = am;
	// update unit
	float axu, ayu, azu;
	axu = geo.fVoxel / abs(ray.x);
	ayu = geo.fVoxel / abs(ray.y);
	azu = geo.fVoxel / abs(ray.z);
	// direction of update
	float iu, ju, ku;
	iu = (source.x < detectPixel[idx].x)? 1 : -1;
	ju = (source.y < detectPixel[idx].y)? 1 : -1;
	ku = (source.z < detectPixel[idx].z)? 1 : -1;

	float maxlength = sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
	float sum_phan = 0.0;// the sum of intersection length with uranium phantom

	unsigned int Np = (imax - imin + 1) + (jmax - jmin + 1) + (kmax - kmin + 1);// the number of intersections
	for(unsigned int ii=0;ii<Np;ii++)
	{
		if(ax==aminc)
		{
			sum_phan += (ax - ac) * tex3D(tex_phan, i+0.5f, j+0.5f, k+0.5f);
			i            = i + iu;
			ac           = ax;
			ax          += axu;
		}
		else if(ay==aminc)
		{
			sum_phan += (ay - ac) * tex3D(tex_phan, i+0.5f, j+0.5f, k+0.5f);
			j            = j + ju;
			ac           = ay;
			ay          += ayu;
		}
		else if(az==aminc)
		{
			sum_phan += (az - ac) * tex3D(tex_phan, i+0.5f, j+0.5f, k+0.5f);
			k            = k + ku;
			ac           = az;
			az          += azu;
		}
		aminc = min(min(ax, ay), az);
	}
	// the projection data for the three materials
	dProjection_phan[idx] = sum_phan * maxlength;
}

int jacob_ray_projection(float const * const img_phan, Geometry geo, float* outProjections, int angleIndex)
{
	// we assume that there is only one gpu
	cudaArray *d_imagedata_phan = 0;// device array for uranium volume
	
	const cudaExtent extent = make_cudaExtent(geo.nVoxelX, geo.nVoxelY, geo.nVoxelZ);
	cudaChannelFormatDesc channelDesc_phan = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&d_imagedata_phan, &channelDesc_phan, extent);// allocate 3d array in device for phantom

	// cudaMemcpy3DParms for uranium phantom
	cudaMemcpy3DParms copyParams_phan = {0};
	copyParams_phan.srcPtr = make_cudaPitchedPtr((void*)img_phan, extent.width * sizeof(float), extent.width, extent.height);
	copyParams_phan.dstArray = d_imagedata_phan;
	copyParams_phan.extent = extent;
	copyParams_phan.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams_phan);

	
	// configure texture options for uranium phantom
	tex_phan.normalized = false;
	tex_phan.filterMode = cudaFilterModePoint;// we don't want interpolation
	tex_phan.addressMode[0] = cudaAddressModeBorder;
	tex_phan.addressMode[1] = cudaAddressModeBorder;
	tex_phan.addressMode[2] = cudaAddressModeBorder;

	cudaBindTextureToArray(tex_phan, d_imagedata_phan, channelDesc_phan);

	// image put into texture memory
	size_t num_bytes = geo.nDetU * geo.nDetV * sizeof(float);
	float *dProjection_phan;
	cudaMalloc((void**)&dProjection_phan, num_bytes);
	cudaMemset(dProjection_phan, 0, num_bytes);

	// source: the coordinate of the x-ray source
	// detectorPixel: the coordinate of the detector units and the amount of units is geo.Nu * geo.Nv
	Point3D source;
	Point3D* detectPixel = (Point3D*)malloc(geo.nDetU * geo.nDetV * sizeof(Point3D));
	
	int divU, divV;
	divU = 16;
	divV = 16;
	dim3 grid((geo.nDetU + divU - 1)/divU, (geo.nDetV + divV - 1)/divV, 1);
	dim3 block(divU, divV, 1);

	computeDeltas_Jacob(geo, detectPixel, &source, angleIndex);

	// copy detector coordinates to device memory
	Point3D *d_detPix;
	cudaMalloc((void**)&d_detPix, geo.nDetU * geo.nDetV * sizeof(Point3D));
	cudaMemcpy(d_detPix, detectPixel, geo.nDetU * geo.nDetV * sizeof(Point3D), cudaMemcpyHostToDevice);

	// kernelPixelDetector<<<grid, block>>>(geo, dProjection_uranium, dProjection_iron, dProjection_LiH, uvOrigin, deltaU, deltaV, source);
	kernelPixelDetector<<<grid, block>>>(geo, dProjection_phan, d_detPix, source);	

	cudaMemcpy(outProjections, dProjection_phan, num_bytes, cudaMemcpyDeviceToHost);
	
	// cudaCheckErrors();

	// free host memory
	free(detectPixel);	
	
	// unbind texture
	cudaUnbindTexture(tex_phan);

	// free device memory
	cudaFree(dProjection_phan);
	cudaFree(d_detPix);

	// free device array
	cudaFreeArray(d_imagedata_phan);

    return 0;
}

void computeDeltas_Jacob(Geometry geo, Point3D* detectPixel, Point3D* source, int angleIndex)
{
	// the initial position of the starting point
	Point3D S0;
	S0.x = geo.fDso;
	S0.y = 0.0;
	S0.z = 0.0;

	// transform 'degree' to 'rad'	
	float beta = angleIndex * PI / 180.0;

	// the position of the starting point after the rotation
	Point3D S1;
	S1.x = S0.x * cos(beta) - S0.y * sin(beta);
	S1.y = S0.x * sin(beta) + S0.y * cos(beta);
	S1.z = 0.0;
	source->x = S1.x;
	source->y = S1.y;
	source->z = S1.z;	

	// the setting of detector
	for(int i=0;i<geo.nDetU;i++)
	{
		for(int j=0;j<geo.nDetV;j++)
		{
			int index = j * geo.nDetU + i;
			// pTemp[index].x = -(geo.fDsd - geo.fDso);
			// pTemp[index].y = ;
			// pTemp[index].z = ;
			float x_temp = -(geo.fDsd - geo.fDso);// the coordinate of the detector unit in x axis
			float y_temp = (-(float)geo.nDetU / 2 + i + 0.5) * geo.fDetUnit;// the coordinate of the detector unit in y axis
			detectPixel[index].x = x_temp * cos(beta) - y_temp * sin(beta);// the coordinate after rotation
			detectPixel[index].y = x_temp * sin(beta) + y_temp * cos(beta);// the coordinate after rotation
			detectPixel[index].z = (-(float)geo.nDetV / 2 + j + 0.5) * geo.fDetUnit;// the coordinate of the detector unit in z axis
		}
	}
}