#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"

__device__ float3 dfX_0( const cv::cuda::PtrStep<float3> src,
		    int i, int j )
{
  return ( src(i+1,j)-src(i-1,j) )/ make_float3(2.0,2.0,2.0);
}

__device__ float3 dfY_0( const cv::cuda::PtrStep<float3> src,
		    int i, int j )
{
  return ( src(i,j+1)-src(i,j-1) )/2.0;
}

__device__ float3 dfX2( const cv::cuda::PtrStep<float3> src,
		   int i, int j )
{
  return ( src(i+1,j) - 
	   2.0* src(i,j) + 
	   src(i-1,j));
}

__device__ float3 dfY2( const cv::cuda::PtrStep<float3> src,
		   int i, int j )
{
  return ( src(i,j+1) - 
	   2.0* src(i,j) + 
	   src(i,j-1) );
}

__device__ float3 dfXY( const cv::cuda::PtrStep<float3> src,
		   int i, int j )
{
  return (  ( src(i+1,j+1) + src(i-1,j-1)-
	      (src(i-1,j+1) + src(i+1,j-1)))/4.0 );
}


__global__ void process(const cv::cuda::PtrStep<float3> src, 
                        cv::cuda::PtrStep<float3> dst, 
                            int rows, int cols, 
                            float tau, float alpha )
{
 
  const int j = blockDim.x * blockIdx.x + threadIdx.x;
  const int i = blockDim.y * blockIdx.y + threadIdx.y;

if ((j > 0) && (j < cols-1) && (i < rows-1) && (i>0))
    {
      float indicator, g11, g22, g12;
      float3 Ix;
      float3 Iy;
      float3 Ixx;
      float3 Iyy;
      float3 Ixy;
      float value;
      float alphag;
      float3 value3;
      float3 num;
      float3 denom;

	  Ix= dfX_0(src, i, j );
	
	  Iy= dfY_0(src, i, j );
	
	  Ixx = dfX2(src, i, j);

	  Iyy = dfY2(src, i, j);

	  Ixy = dfXY(src, i, j);

	  g11= 1.0 + Ix.x*Ix.x + Ix.y*Ix.y + Ix.z*Ix.z;
	  g12= Ix.x*Iy.x + Ix.y*Iy.y + Ix.z*Iy.z;
	  g22= 1.0 + Iy.x*Iy.x + Iy.y*Iy.y + Iy.z*Iy.z;
	
	  indicator= sqrt( (g11-g22)*(g11-g22) + 4.0 *g12*g12 );
	
	  value = sqrt(indicator)/tau;
	  value*=-value;
	  alphag= alpha*exp(value);

	  num = Ixx*Iy*Iy - 2.0*Ixy*Ix*Iy + Iyy*Ix*Ix;
	  denom = 1e-8 + Ix*Ix + Iy*Iy;

      
	  value3 = src(i,j) + make_float3(alphag,alphag,alphag)*num/denom;
	    
      clamp (value3, 0.0, 1.0);
    
      dst(i,j) = value3;
    }

}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA ( cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, 
                            float tau, float alpha  )
{
  const dim3 block(32, 8);
  const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

  process<<<grid, block>>>(src, dst, src.rows, src.cols, 
                           tau, alpha );

}

