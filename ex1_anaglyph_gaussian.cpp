#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono> // for high_resolution_clock

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

cv::Vec3b gaussianProcess(int i, int j, cv::Mat_<cv::Vec3b> source, bool isLeftImage, int kernelSize = 10, float sigma = 2.0)
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

cv::Vec3b covarianceGaussianProcess(int i, int j, cv::Mat_<cv::Vec3b> source, bool isLeftImage, int nbhoodSize = 5, float factorRatio = 9, int baseGaussianKernel = 1)
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
    covariance(1, 0) += (greenValues[i] - greenMean) * (redValues[i] - redMean); // redundant?
    covariance(1, 1) += (greenValues[i] - greenMean) * (greenValues[i] - greenMean);
    covariance(1, 2) += (greenValues[i] - greenMean) * (blueValues[i] - blueMean);
    covariance(2, 0) += (blueValues[i] - blueMean) * (redValues[i] - redMean);     // redundant?
    covariance(2, 1) += (blueValues[i] - blueMean) * (greenValues[i] - greenMean); // redundant?
    covariance(2, 2) += (blueValues[i] - blueMean) * (blueValues[i] - blueMean);
  }

  covariance /= nbHoodArea;

  // calculate determinant
  float det = cv::determinant(covariance);

  // cout << "Covariance: " << covariance << endl;
  // cout << "Determinant: " << det << endl;

  // calculate gaussian
  float gaussianValue = baseGaussianKernel + exp(-det / factorRatio);
  // float gaussianValue = factorRatio * pow(det, 0.5) + baseGaussianKernel;

  // cout << "Gaussian: " << gaussianValue << endl;
  cv::Vec3b result = gaussianProcess(i, j, source, isLeftImage, gaussianValue, 2.0);

  return result;
}

bool enableGaussian = false;
bool enbleCovarianceGaussian = true;
bool enableAnaglyph = true;

int main(int argc, char **argv)
{
  cv::Mat_<cv::Vec3b> source = cv::imread(argv[1], cv::IMREAD_COLOR);

  const int res_cols = source.cols / 2;
  cv::Mat_<cv::Vec3b> destination(source.rows, res_cols);

  cv::imshow("Source Image", source);

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 500;

  // for (int it = 0; it < iter; it++)
  // {
#pragma omp parallel for
  for (int i = 0; i < source.rows; i++)
    for (int j = 0; j < res_cols; j++)
    {
      cv::Vec3b pixelLeft;
      cv::Vec3b pixelRight;

      for (int c = 0; c < 3; c++)
      {

        if (enbleCovarianceGaussian)
        {
          pixelLeft = covarianceGaussianProcess(i, j, source, true);
          pixelRight = covarianceGaussianProcess(i, j + res_cols, source, false);
          break;
        }
        else if (enableGaussian)
        {
          pixelLeft = gaussianProcess(i, j, source, true);
          pixelRight = gaussianProcess(i, j + res_cols, source, false);
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
        // destination(i, j) = trueAnaglyph(pixelLeft, pixelRight);
        // destination(i, j) = greyAnaglyph(pixelLeft, pixelRight);
        destination(i, j) = colourAnaglyph(pixelLeft, pixelRight);
      // destination(i, j) = halfColourAnaglyph(pixelLeft, pixelRight);
      // destination(i, j) = optimisedAnaglyph(pixelLeft, pixelRight);
      else // if no anaglyph
        // destination(i, j) = pixelLeft;
        destination(i, j) = pixelRight;
    }
  // }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - begin;

  cv::imshow("Processed Image", destination);

  cout << "Total time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
  cout << "IPS: " << iter / diff.count() << endl;

  cv::waitKey();
  return 0;
}
