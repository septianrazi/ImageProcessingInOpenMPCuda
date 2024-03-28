#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono> // for high_resolution_clock
#include <omp.h>  // OMP API

using namespace std;

int main(int argc, char **argv)
{

  cv::Mat_<uchar> source = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  cv::Mat_<uchar> destination(source.rows, source.cols);

  cv::imshow("Source Image", source);

  const int iter = 200;

  for (int t = 1; t < 48; t++)
  {
    omp_set_num_threads(t);
    auto begin = chrono::high_resolution_clock::now();

    for (int it = 0; it < iter; it++)
    {
#pragma omp parallel for
      for (int i = 0; i < source.rows; i++)
        // #pragma omp parallel for
        for (int j = 0; j < source.cols; j++)
          destination(i, j) = 255.0 * cos((255 - source(i, j)) / 255.0);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cv::imshow("Processed Image", destination);

    cout << "Number of threads: " << t << endl;
    cout << "   Total time: " << diff.count() << " s" << endl;
    cout << "   Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "   IPS: " << iter / diff.count() << endl;
    cout << endl;
  }
  cv::waitKey();

  return 0;
}
