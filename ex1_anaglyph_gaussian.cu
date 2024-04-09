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

// default values for arguments
bool enableAnaglyph = false;
__device__ int anaglyphValue = 1;

bool enableGaussian = false;
__device__ int gausKernel = 4;
__device__ float gausSigma = 2.0;

bool enableDenoising = false;
__device__ int denoisingNbhoodSize = 8;
__device__ float denoisingFactorRatio = 60;

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

__device__ float gaussian(float x, float sigma)
{
  return 1 / (sqrt(2 * M_PI) * sigma) * exp(-x * x / (2 * sigma * sigma));
}

__device__ uchar3 gaussianPixel(const cv::cuda::PtrStep<uchar3> src, int rows, int cols, int i, int j, bool isLeftImage, int kernelSize, float sigma)
{
  float counter = 0;

  // set limits differently for left image or right image
  int imageStart = isLeftImage ? 0 : cols / 2;
  int imageLimit = isLeftImage ? cols / 2 : cols;

  uchar3 result = make_uchar3(0, 0, 0);

  for (int w = -kernelSize / 2; w <= kernelSize / 2; w++)
  {
    for (int l = -kernelSize / 2; l <= kernelSize / 2; l++)
    {
      if (i + w >= 0 && i + w < rows && j + l >= imageStart && j + l < imageLimit)
      {
        float thisGaussian = gaussian(w, sigma) * gaussian(l, sigma);

        result.x += src(i + w, j + l).x * thisGaussian;
        result.y += src(i + w, j + l).y * thisGaussian;
        result.z += src(i + w, j + l).z * thisGaussian;

        counter += thisGaussian;
      }
      // result += prev_result;
    }
  }

  // normalise results with gaussian weights
  result.x = result.x / counter;
  result.y = result.y / counter;
  result.z = result.z / counter;

  return result;
}
__device__ float determinant(float matrix[3][3])
{
  return matrix[0][0] * matrix[1][1] * matrix[2][2] + matrix[0][1] * matrix[1][2] * matrix[2][0] + matrix[0][2] * matrix[1][0] * matrix[2][1] - matrix[0][2] * matrix[1][1] * matrix[2][0] - matrix[0][1] * matrix[1][0] * matrix[2][2] - matrix[0][0] * matrix[1][2] * matrix[2][1];
}

__device__ uchar3 denoisingProcess(int i, int j, const cv::cuda::PtrStep<uchar3> src, int rows, int cols, bool isLeftImage, int nbhoodSize, float factorRatio, int baseGaussianKernel)
{
  float counter = 0;

  // set limits differently for left image or right image
  int imageStart = isLeftImage ? 0 : cols / 2;
  int imageLimit = isLeftImage ? cols / 2 : cols;

  float covariance[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

  // max neighbourhood number is 10 because of array size
  if (nbhoodSize > 10)
    nbhoodSize = 10;
  const int nbHoodArea = nbhoodSize * nbhoodSize;
  float redValues[100];
  float greenValues[100];
  float blueValues[100];
  float redMean = 0;
  float greenMean = 0;
  float blueMean = 0;

  int index = 0;

  for (int w = -nbhoodSize / 2; w <= nbhoodSize / 2; w++)
  {
    for (int l = -nbhoodSize / 2; l <= nbhoodSize / 2; l++)
    {
      if (i + w >= 0 && i + w < rows && j + l >= imageStart && j + l < imageLimit)
      {
        // store values of each cvolour for covariance calcualtion
        redValues[index] = src(i + w, j + l).z;
        greenValues[index] = src(i + w, j + l).y;
        blueValues[index] = src(i + w, j + l).x;
      }
      else
      {
        // mirror the main pixel if out of bounds
        redValues[index] = src(i, j).z;
        greenValues[index] = src(i, j).y;
        blueValues[index] = src(i, j).x;
      }

      // add values of each colour for mean calculation
      redMean += redValues[index];
      greenMean += greenValues[index];
      blueMean += blueValues[index];

      index++;
    }
  }

  // calculate mean
  redMean /= nbHoodArea;
  greenMean /= nbHoodArea;
  blueMean /= nbHoodArea;

  // cout << "Mean: " << redMean << " " << greenMean << " " << blueMean << endl;

  // populate covariance matrix
  for (int i = 0; i < nbHoodArea; i++)
  {
    covariance[0][0] += (redValues[i] - redMean) * (redValues[i] - redMean) / nbHoodArea;
    covariance[0][1] += (redValues[i] - redMean) * (greenValues[i] - greenMean) / nbHoodArea;
    covariance[0][2] += (redValues[i] - redMean) * (blueValues[i] - blueMean) / nbHoodArea;
    covariance[1][0] += (greenValues[i] - greenMean) * (redValues[i] - redMean) / nbHoodArea; // redundant ;
    covariance[1][1] += (greenValues[i] - greenMean) * (greenValues[i] - greenMean) / nbHoodArea;
    covariance[1][2] += (greenValues[i] - greenMean) * (blueValues[i] - blueMean) / nbHoodArea;
    covariance[2][0] += (blueValues[i] - blueMean) * (redValues[i] - redMean) / nbHoodArea;     // redundant ?
    covariance[2][1] += (blueValues[i] - blueMean) * (greenValues[i] - greenMean) / nbHoodArea; // redundant ;?
    covariance[2][2] += (blueValues[i] - blueMean) * (blueValues[i] - blueMean) / nbHoodArea;
  }

  // calculate determinant
  float det = determinant(covariance);
  det = std::abs(det);

  // cout << "Covariance: " << covariance << endl;
  // cout << "Det " << det << endl;

  // calculate gaussian
  // float gaussianValue = baseGaussianKernel + exp(-det / factorRatio);
  // float gaussianValue = factorRatio / (baseGaussianKernel + det);
  // float gaussianValue = factorRatio / (baseGaussianKernel + 1000 * pow(det, 3));
  float gaussianValue = factorRatio / (baseGaussianKernel + max(0.0f, log(det)));

  // cout << "Gaussian: " << gaussianValue << endl;
  uchar3 result = gaussianPixel(src, rows, cols, i, j, isLeftImage, gaussianValue, 2.0);

  return result;
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, bool enableAnaglyph, bool enableGaussian, bool enableDenoising)
{

  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  if (dst_x < cols / 2 && dst_y < rows)
  {
    uchar3 left = src(dst_y, dst_x);
    uchar3 right = src(dst_y, cols / 2 + dst_x);

    if (enableDenoising)
    {
      int nbhoodSize = denoisingNbhoodSize;
      float factorRatio = denoisingFactorRatio;
      int baseGaussianKernel = 1;

      dst(dst_y, dst_x) = denoisingProcess(dst_y, dst_x, src, rows, cols, true, nbhoodSize, factorRatio, baseGaussianKernel);
      dst(dst_y, cols / 2 + dst_x) = denoisingProcess(dst_y, cols / 2 + dst_x, src, rows, cols, false, nbhoodSize, factorRatio, baseGaussianKernel);

      left = dst(dst_y, dst_x);
      right = dst(dst_y, cols / 2 + dst_x);
    }
    else if (enableGaussian)
    {
      int kernelSize = gausKernel;
      float sigma = gausSigma;

      dst(dst_y, dst_x) = gaussianPixel(src, rows, cols, dst_y, dst_x, true, kernelSize, sigma);
      dst(dst_y, cols / 2 + dst_x) = gaussianPixel(src, rows, cols, dst_y, cols / 2 + dst_x, false, kernelSize, sigma);

      left = dst(dst_y, dst_x);
      right = dst(dst_y, cols / 2 + dst_x);
    }

    if (enableAnaglyph)
    {
      switch (anaglyphValue)
      {
      case 1:
        dst(dst_y, dst_x) = trueAnaglyph(left, right);
        break;
      case 2:
        dst(dst_y, dst_x) = greyAnaglyph(left, right);
        break;
      case 3:
        dst(dst_y, dst_x) = colourAnaglyph(left, right);
        break;
      case 4:
        dst(dst_y, dst_x) = halfColourAnaglyph(left, right);
        break;
      case 5:
        dst(dst_y, dst_x) = optimisedAnaglyph(left, right);
        break;
      }
    }
  }
}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void processCUDA(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst)
{
  const dim3 block(16, 16);
  const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

  process<<<grid, block>>>(src, dst, src.rows, src.cols, enableAnaglyph, enableGaussian, enableDenoising);
}

// default values for arguments
int hostAnaglyphValue = 1;

int hostGausKernel = 4;
float hostGausSigma = 2.0;

int hostDenoisingNbhoodSize = 8;
float hostDenoisingFactorRatio = 60;

int main(int argc, char **argv)
{
  if (argc > 2)
  {
    for (int i = 2; i < argc; i++)
    {
      string arg = argv[i];
      if (arg == "-gaussian" || arg == "-g")
      {
        enableGaussian = true;
        if (i + 2 < argc && std::isdigit(*argv[i + 1]) && std::isdigit(*argv[i + 2])) // Ensure we have two numerical values after the argument
        {
          hostGausKernel = std::stod(argv[i + 1]); // Convert the next argument to a double
          hostGausSigma = std::stod(argv[i + 2]);  // Convert the argument after that to a double
          i += 2;                                  // Skip the next two arguments since we just processed them
        }
        cudaMemcpyToSymbol(gausKernel, &hostGausKernel, sizeof(int));
        cudaMemcpyToSymbol(gausSigma, &hostGausSigma, sizeof(float));
      }
      else if (arg == "-anaglyph" || arg == "-a")
      {
        enableAnaglyph = true;
        if (i + 1 < argc && std::isdigit(*argv[i + 1])) // Ensure we have a value after the argument
        {
          hostAnaglyphValue = std::stod(argv[i + 1]); // Convert the next argument to a double
          i++;                                        // Skip the next argument since we just processed it
        }
        cudaMemcpyToSymbol(anaglyphValue, &hostAnaglyphValue, sizeof(int));
      }
      else if (arg == "-denoising" || arg == "-d")
      {
        enableDenoising = true;
        if (i + 2 < argc && std::isdigit(*argv[i + 1]) && std::isdigit(*argv[i + 2])) // Ensure we have two numerical values after the argument
        {
          int hostDenoisingNbhoodSize = std::stod(argv[i + 1]);    // Convert the next argument to a double
          float hostDenoisingFactorRatio = std::stod(argv[i + 2]); // Convert the argument after that to a double
          i += 2;                                                  // Skip the next two arguments since we just processed them
        }
        cudaMemcpyToSymbol(denoisingNbhoodSize, &hostDenoisingNbhoodSize, sizeof(int));
        cudaMemcpyToSymbol(denoisingFactorRatio, &hostDenoisingFactorRatio, sizeof(float));
      }

      else if (arg == "-h" || arg == "--help")
      {
        std::cout << "Usage: program imagePath [-g gaussianKernel gaussianSigma] [-a anaglyphValue] [-d denoisingNbhoodSize denoisingFactorRatio]\n";
        std::cout << "Options:\n";
        std::cout << "  -g, --gaussian     Enable Gaussian filter with specified kernel and sigma\n";
        std::cout << "                      kernel: size of the Gaussian kernel (int, default: 4)\n";
        std::cout << "                      sigma: standard deviation of the Gaussian distribution (double, default: 2.0)\n";
        std::cout << "  -a, --anaglyph     Enable Anaglyph with specified value\n";
        std::cout << "                      anaglyphValue: value for the Anaglyph effect (int, default: 1)\n";
        std::cout << "                        1: True Anaglyph\n";
        std::cout << "                        2: Grey Anaglyph\n";
        std::cout << "                        3: Color Anaglyph\n";
        std::cout << "                        4: Half-Color Anaglyph\n";
        std::cout << "                        5: Optimised Anaglyph\n";
        std::cout << "  -d, --denoising    Enable Denoising with specified neighborhood size and factor ratio\n";
        std::cout << "                      neighbourhood size: size of the neighborhood for denoising (int, default: 8)\n";
        std::cout << "                      factor ratio: factor ratio for denoising (double, default: 60)\n";
        std::cout << "  -h, --help         Display this help message and exit\n";
        return 0;
      }
    }
  }

  cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

  cv::Mat h_img = cv::imread(argv[1]);
  cv::Mat h_result;

  h_result = h_img;

  cv::cuda::GpuMat d_img, d_result;

  d_img.upload(h_img);
  d_result.upload(h_result);
  int width = d_img.cols;
  int height = d_img.rows;

  cv::imshow("Original Image", h_img);

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 1;

  for (int i = 0; i < iter; i++)
  {
    d_img.upload(h_img);
    processCUDA(d_img, d_result);
    d_result.download(h_result);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  // d_result.download(h_result);

  if (enableAnaglyph)
    h_result = h_result(cv::Rect(0, 0, h_img.cols / 2, h_img.rows)).clone();

  cv::imshow("Processed Image", h_result);

  cout << "Time: " << diff.count() << endl;
  cout << "Time/frame: " << diff.count() / iter << endl;
  cout << "IPS: " << iter / diff.count() << endl;

  cv::waitKey();

  return 0;
}
