#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono> // for high_resolution_clock

using namespace std;

int main(int argc, char **argv)
{

  cv::Mat_<cv::Vec3b> source = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat_<cv::Vec3b> destination(source.rows, source.cols);

  cv::imshow("Source Image", source);

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 500;

  for (int it = 0; it < iter; it++)
  {
#pragma omp parallel for
    for (int i = 0; i < source.rows; i++)
      for (int j = 0; j < source.cols; j++)
        for (int c = 0; c < 3; c++)
        {
          destination(i, j)[c] = 255.0 * tan((255 - source(i, j)[c]) / 255.0);
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
