#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>  // for high_resolution_clock
#include <fstream> // for file operations

using namespace std;

cv::Vec3b trueAnaglyph(cv::Vec3b pixelLeft, cv::Vec3b pixelRight)
{
  cv::Vec3b result;

  result[2] = pixelLeft[2] * 0.299 + pixelLeft[1] * 0.587 + pixelLeft[0] * 0.114;
  result[1] = 0;
  result[0] = pixelRight[2] * 0.299 + pixelRight[1] * 0.587 + pixelRight[0] * 0.114;

  return result;
}

cv::Vec3b greyAnaglyph(cv::Vec3b pixelLeft, cv::Vec3b pixelRight)
{
  cv::Vec3b result;

  result[2] = pixelLeft[2] * 0.299 + pixelLeft[1] * 0.587 + pixelLeft[0] * 0.114;
  result[1] = pixelRight[2] * 0.299 + pixelRight[1] * 0.587 + pixelRight[0] * 0.114;
  result[0] = pixelRight[2] * 0.299 + pixelRight[1] * 0.587 + pixelRight[0] * 0.114;

  return result;
}

cv::Vec3b colourAnaglyph(cv::Vec3b pixelLeft, cv::Vec3b pixelRight)
{
  cv::Vec3b result;

  result[2] = pixelLeft[2];
  result[1] = pixelRight[1];
  result[0] = pixelRight[0];

  return result;
}

cv::Vec3b halfColourAnaglyph(cv::Vec3b pixelLeft, cv::Vec3b pixelRight)
{
  cv::Vec3b result;

  result[2] = pixelLeft[2] * 0.299 + pixelLeft[1] * 0.587 + pixelLeft[0] * 0.114;
  result[1] = pixelRight[1];
  result[0] = pixelRight[0];

  return result;
}

cv::Vec3b optimisedAnaglyph(cv::Vec3b pixelLeft, cv::Vec3b pixelRight)
{
  cv::Vec3b result;

  result[2] = pixelLeft[1] * 0.7 + pixelLeft[0] * 0.3;
  result[1] = pixelRight[1];
  result[0] = pixelRight[0];

  return result;
}

float gaussian(float x, float sigma)
{
  return 1 / (sqrt(2 * M_PI) * sigma) * exp(-x * x / (2 * sigma * sigma));
}

cv::Vec3b gaussianProcess(int i, int j, cv::Mat_<cv::Vec3b> source, bool isLeftImage, int kernelSize = 4, float sigma = 2.0)
{
  float counter = 0;

  // set limits differently for left image or right image
  int imageStart = isLeftImage ? 0 : source.cols / 2;
  int imageLimit = isLeftImage ? source.cols / 2 : source.cols;

  cv::Vec3b result;
  cv::Vec3b prev_result = source(i, j) * gaussian(0, sigma) * gaussian(0, sigma);

  for (int w = -kernelSize / 2; w <= kernelSize / 2; w++)
  {
    for (int l = -kernelSize / 2; l <= kernelSize / 2; l++)
    {
      if (i + w >= 0 && i + w < source.rows && j + l >= imageStart && j + l < imageLimit)
      {
        float thisGaussian = gaussian(w, sigma) * gaussian(l, sigma);
        prev_result = source(i + w, j + l) * thisGaussian;
        counter += thisGaussian;
        result += prev_result;
      }
      // result += prev_result;
    }
  }

  // normalise results with gaussian weights
  return result / counter;
}

cv::Vec3b denoisingProcess(int i, int j, cv::Mat_<cv::Vec3b> source, bool isLeftImage, int nbhoodSize = 8, float factorRatio = 60, int baseGaussianKernel = 1)
{
  float counter = 0;

  // set limits differently for left image or right image
  int imageStart = isLeftImage ? 0 : source.cols / 2;
  int imageLimit = isLeftImage ? source.cols / 2 : source.cols;

  cv::Mat_<float> covariance = cv::Mat_<float>::zeros(3, 3);

  int nbHoodArea = nbhoodSize * nbhoodSize;
  float redValues[nbHoodArea];
  float greenValues[nbHoodArea];
  float blueValues[nbHoodArea];
  float redMean = 0;
  float greenMean = 0;
  float blueMean = 0;

  int index = 0;

  for (int w = -nbhoodSize / 2; w <= nbhoodSize / 2; w++)
  {
    for (int l = -nbhoodSize / 2; l <= nbhoodSize / 2; l++)
    {
      if (i + w >= 0 && i + w < source.rows && j + l >= imageStart && j + l < imageLimit)
      {
        // store values of each cvolour for covariance calcualtion
        redValues[index] = source(i + w, j + l)[2];
        greenValues[index] = source(i + w, j + l)[1];
        blueValues[index] = source(i + w, j + l)[0];
      }
      else
      {
        // mirror the main pixel if out of bounds
        redValues[index] = source(i, j)[2];
        greenValues[index] = source(i, j)[1];
        blueValues[index] = source(i, j)[0];
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

  for (int i = 0; i < nbHoodArea; i++)
  {
    covariance(0, 0) += (redValues[i] - redMean) * (redValues[i] - redMean);
    covariance(0, 1) += (redValues[i] - redMean) * (greenValues[i] - greenMean);
    covariance(0, 2) += (redValues[i] - redMean) * (blueValues[i] - blueMean);
    covariance(1, 0) += (greenValues[i] - greenMean) * (redValues[i] - redMean);
    covariance(1, 1) += (greenValues[i] - greenMean) * (greenValues[i] - greenMean);
    covariance(1, 2) += (greenValues[i] - greenMean) * (blueValues[i] - blueMean);
    covariance(2, 0) += (blueValues[i] - blueMean) * (redValues[i] - redMean);
    covariance(2, 1) += (blueValues[i] - blueMean) * (greenValues[i] - greenMean);
    covariance(2, 2) += (blueValues[i] - blueMean) * (blueValues[i] - blueMean);
  }

  covariance /= nbHoodArea;
  // covariance /= 255;

  // calculate determinant
  float det = cv::determinant(covariance);
  det = std::abs(det);

  // cout << "Covariance: " << covariance << endl;
  // cout << "Det " << det << endl;

  // calculate gaussian
  // float gaussianValue = baseGaussianKernel + exp(-det / factorRatio);
  // float gaussianValue = factorRatio / (baseGaussianKernel + 1000 * pow(det, 3));
  float gaussianValue = factorRatio / (baseGaussianKernel + max(0.0f, log(det)));
  // float gaussianValue = factorRatio * pow(det, 0.5) + baseGaussianKernel;

  // cout << "Gaussian: " << gaussianValue << endl;

  // if (gaussianValue < 0.0 || gaussianValue > 30)
  //   cout << "Gaussian: " << gaussianValue << " \t Det " << det << endl;
  cv::Vec3b result = gaussianProcess(i, j, source, isLeftImage, gaussianValue, 2.0);
  // cv::Vec3b result = cv::Vec3b(150 * det, 0, 0);
  // cv::Vec3b result = cv::Vec3b(0, 30 * gaussianValue, 0);

  return result;
}

// default values for arguments
bool enableAnaglyph = false;
int anaglyphValue = 1;

bool enableGaussian = false;
int gausKernel = 4;
float gausSigma = 2.0;

bool enableDenoising = false;
int denoisingNbhoodSize = 8;
float denoisingFactorRatio = 60;

int main(int argc, char **argv)
{
  cv::Mat_<cv::Vec3b> source = cv::imread(argv[1], cv::IMREAD_COLOR);

  const int res_cols = source.cols / 2;
  cv::Mat_<cv::Vec3b> destination(source.rows, res_cols);

  cv::imshow("Source Image", source);

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 1;

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
          gausKernel = std::stod(argv[i + 1]); // Convert the next argument to a double
          gausSigma = std::stod(argv[i + 2]);  // Convert the argument after that to a double
          i += 2;                              // Skip the next two arguments since we just processed them
        }
      }
      else if (arg == "-anaglyph" || arg == "-a")
      {
        enableAnaglyph = true;
        if (i + 1 < argc && std::isdigit(*argv[i + 1])) // Ensure we have a value after the argument
        {
          anaglyphValue = std::stod(argv[i + 1]); // Convert the next argument to a double
          i++;                                    // Skip the next argument since we just processed it
        }
      }
      else if (arg == "-denoising" || arg == "-d")
      {
        enableDenoising = true;
        if (i + 2 < argc && std::isdigit(*argv[i + 1]) && std::isdigit(*argv[i + 2])) // Ensure we have two numerical values after the argument
        {
          denoisingNbhoodSize = std::stod(argv[i + 1]);  // Convert the next argument to a double
          denoisingFactorRatio = std::stod(argv[i + 2]); // Convert the argument after that to a double
          i += 2;                                        // Skip the next two arguments since we just processed them
        }
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

  for (int it = 0; it < iter; it++)
  {
#pragma omp parallel for
    for (int i = 0; i < source.rows; i++)
      for (int j = 0; j < res_cols; j++)
      {
        cv::Vec3b pixelLeft;
        cv::Vec3b pixelRight;

        for (int c = 0; c < 3; c++)
        {

          if (enableGaussian)
          {
            pixelLeft = gaussianProcess(i, j, source, true, gausKernel, gausSigma);
            pixelRight = gaussianProcess(i, j + res_cols, source, false, gausKernel, gausSigma);
            break;
          }
          else if (enableDenoising)
          {
            pixelLeft = denoisingProcess(i, j, source, true, denoisingNbhoodSize, denoisingFactorRatio);
            pixelRight = denoisingProcess(i, j + res_cols, source, false, denoisingNbhoodSize, denoisingFactorRatio);
            break;
          }
          else
          {
            pixelLeft[c] = source(i, j)[c];
            pixelRight[c] = source(i, j + res_cols)[c];
          }
        }

        // #pragma omp critical
        if (enableAnaglyph)
          switch (anaglyphValue)
          {
          case 1:
            destination(i, j) = trueAnaglyph(pixelLeft, pixelRight);
            break;
          case 2:
            destination(i, j) = greyAnaglyph(pixelLeft, pixelRight);
            break;
          case 3:
            destination(i, j) = colourAnaglyph(pixelLeft, pixelRight);
            break;
          case 4:
            destination(i, j) = halfColourAnaglyph(pixelLeft, pixelRight);
            break;
          case 5:
            destination(i, j) = optimisedAnaglyph(pixelLeft, pixelRight);
            break;
          }
        else // if no anaglyph
             // destination(i, j) = pixelLeft;
          destination(i, j) = pixelRight;
      }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  cv::imshow("Processed Image", destination);

  cout << "Total time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
  cout << "IPS: " << iter / diff.count() << endl;

  cv::waitKey();
  return 0;
}
