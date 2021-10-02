#include "geometry.h"

#ifndef JACOB_RAY_PROJECTION_H
#define JACOB_RAY_PROJECTION_H

extern int jacob_ray_projection(float const * const img_phan, Geometry geo, float* outProjections, int angleIndex);
void computeDeltas_Jacob(Geometry geo, Point3D* detectPixel, Point3D* source, int angleIndex);

#endif