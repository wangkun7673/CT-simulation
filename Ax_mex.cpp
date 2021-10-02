#include "mex.h"
#include "geometry.h"
#include "jacob_ray_projection.h"
#include "matrix.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
    if(nrhs!=3)
    {
        mexErrMsgTxt("Invalid Number of Inputs");
    }
    
    mxArray *geometryMex = (mxArray*)prhs[0];
    const char *fieldnames[12];
    fieldnames[0] = "bSysType";
    fieldnames[1] = "fDso";
    fieldnames[2] = "fDsd";
    fieldnames[3] = "nAngle";
    fieldnames[4] = "nDetU";
    fieldnames[5] = "nDetV";
    fieldnames[6] = "fDetUnit";
    fieldnames[7] = "nVoxelX";
    fieldnames[8] = "nVoxelY";
    fieldnames[9] = "nVoxelZ";
    fieldnames[10] = "fVoxel";
    fieldnames[11] = "bGPU";
    
    double *bSysType, *fDso, *fDsd, *nAngle, *nDetU, *nDetV, *fDetUnit, *nVoxelX, *nVoxelY, *nVoxelZ, *fVoxel, *bGPU;
    mxArray *tmp;
    Geometry geo;
    for(int ifield=0;ifield<12;ifield++)
    {
        tmp = mxGetField(geometryMex, 0, fieldnames[ifield]);
        if(!tmp)
        {
            continue;
        }
        switch(ifield)
        {
            case 0:
                bSysType = (double *)mxGetData(tmp);
                geo.bSysType = (int)bSysType[0];
                break;
            case 1:
                fDso = (double *)mxGetData(tmp);
                geo.fDso = (float)fDso[0];
                break;
            case 2:
                fDsd = (double *)mxGetData(tmp);
                geo.fDsd = (float)fDsd[0];
                break;
            case 3:
                nAngle = (double *)mxGetData(tmp);
                geo.nAngle = (int)nAngle[0];
                break;
            case 4:
                nDetU = (double *)mxGetData(tmp);
                geo.nDetU = (int)nDetU[0];
                break;
            case 5:
                nDetV = (double *)mxGetData(tmp);
                geo.nDetV = (int)nDetV[0];
                break;
            case 6:
                fDetUnit = (double *)mxGetData(tmp);
                geo.fDetUnit = (float)fDetUnit[0];
                break;
            case 7:
                nVoxelX = (double *)mxGetData(tmp);
                geo.nVoxelX = (int)nVoxelX[0];
                break;
            case 8:
                nVoxelY = (double *)mxGetData(tmp);
                geo.nVoxelY = (int)nVoxelY[0];
                break;
            case 9:
                nVoxelZ = (double *)mxGetData(tmp);
                geo.nVoxelZ = (int)nVoxelZ[0];
                break;
            case 10:
                fVoxel = (double *)mxGetData(tmp);
                geo.fVoxel = (float)fVoxel[0];
                break;
            case 11:
                bGPU = (double *)mxGetData(tmp);
                geo.bGPU = (int)bGPU[0];
                break;
            default:
                mexErrMsgTxt("This should not happen. Weird");
                break;
        }
    }
    
    // 2nd input argument and obtain the index of the rotation angle
    double *pAngleInd = (double *)mxGetData(prhs[1]);
    int angleIndex = (int)pAngleInd[0];
    
    // 3rd input argument and obtain the volume data
    mxArray const * const phan = prhs[2];
    mwSize const numDims_phan = mxGetNumberOfDimensions(phan);
    
    float const * const img_phan = static_cast<float const*>(mxGetData(phan));
    const mwSize *size_img_phan = mxGetDimensions(phan);
    
    size_t num_bytes = geo.nDetU * geo.nDetV * sizeof(float);
    mwSize outsize[2];
    outsize[0] = geo.nDetU;
    outsize[1] = geo.nDetV;
    plhs[0] = mxCreateNumericArray(2, outsize, mxSINGLE_CLASS, mxREAL);
    float *outProjections = (float*)mxGetPr(plhs[0]);
    
    jacob_ray_projection(img_phan, geo, outProjections, angleIndex);
}
