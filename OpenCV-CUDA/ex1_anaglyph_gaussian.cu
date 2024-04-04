#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono> // for high_resolution_clock
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"

using namespace std;

__device__ void invert(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols)
{
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  if (dst_x < cols && dst_y < rows)
  {
    uchar3 val = src(dst_y, dst_x);
    dst(dst_y, dst_x).x = 255 - val.x;
    dst(dst_y, dst_x).y = 255 - val.y;
    dst(dst_y, dst_x).z = 255 - val.z;
  }
}

__device__ uchar3 trueAnaglyph(uchar3 left, uchar3 right)
{
  uchar3 result;
  result.z = left.z * 0.299 + left.y * 0.587 + left.x * 0.114;
  result.y = 0;
  result.x = right.z * 0.299 + right.y * 0.587 + right.x * 0.114;

  return result;
}
__device__ uchar3 greyAnaglyph(uchar3 left, uchar3 right)
{
  uchar3 result;
  result.z = left.z * 0.299 + left.y * 0.587 + left.x * 0.114;
  result.y = right.z * 0.299 + right.y * 0.587 + right.x * 0.114;
  result.x = right.z * 0.299 + right.y * 0.587 + right.x * 0.114;
  return result;
}

__device__ uchar3 colourAnaglyph(uchar3 left, uchar3 right)
{
  uchar3 result;
  result.x = right.x;
  result.y = right.y;
  result.z = left.z;
  return result;
}

__device__ uchar3 halfColourAnaglyph(uchar3 left, uchar3 right)
{
  uchar3 result;
  result.x = right.x;
  result.y = right.y;
  result.z = left.x * 0.299 + left.y * 0.587 + left.z * 0.114;
  return result;
}

__device__ uchar3 optimisedAnaglyph(uchar3 left, uchar3 right)
{
  uchar3 result;
  result.x = right.x;
  result.y = right.y;
  result.z = left.y * 0.7 + left.x * 0.3;
  return result;
}

__device__ void anaglyph(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols)
{
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  if (dst_x < cols / 2 && dst_y < rows)
  {
    uchar3 left = src(dst_y, dst_x);
    uchar3 right = src(dst_y, cols / 2 + dst_x);

    // dst(dst_y, dst_x) = trueAnaglyph(left, right);
    // dst(dst_y, dst_x) = greyAnaglyph(left, right);
    // dst(dst_y, dst_x) = colourAnaglyph(left, right);
    // dst(dst_y, dst_x) = halfColourAnaglyph(left, right);
    dst(dst_y, dst_x) = optimisedAnaglyph(left, right);
  }
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols)
{
  // invert(src, dst, rows, cols);
  anaglyph(src, dst, rows, cols);
}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void processCUDA(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst)
{
  const dim3 block(32, 8);
  const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

  process<<<grid, block>>>(src, dst, dst.rows, dst.cols);
}

int main(int argc, char **argv)
{
  cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

  cv::Mat h_img = cv::imread(argv[1]);
  cv::Mat h_result;

  cv::cuda::GpuMat d_img, d_result;

  d_img.upload(h_img);
  d_result.upload(h_img);
  int width = d_img.cols;
  int height = d_img.rows;

  cv::imshow("Original Image", h_img);

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 1; // 100000;

  for (int i = 0; i < iter; i++)
  {
    d_img.upload(h_img);
    processCUDA(d_img, d_result);
    d_result.download(h_result);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  // d_result.download(h_result);

  cv::imshow("Processed Image", h_result);

  cout << "Time: " << diff.count() << endl;
  cout << "Time/frame: " << diff.count() / iter << endl;
  cout << "IPS: " << iter / diff.count() << endl;

  cv::waitKey();

  return 0;
}
