#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void startCUDA ( cv::cuda::GpuMat& src,cv::cuda::GpuMat& dst );

int main( int argc, char** argv )
{
  cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Resized Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

  cv::Mat h_img = cv::imread(argv[1]);
  cv::cuda::GpuMat d_img,d_result;
  cv::Mat h_result;


  d_img.upload(h_img);
  int width= d_img.cols;
  int height = d_img.size().height;

  cv::imshow("Original Image", d_img);
  
  auto begin = chrono::high_resolution_clock::now();
  const int iter = 10000;
  
  for (int i=0;i<iter;i++)
    {
      cv::cuda::resize(d_img,d_result,cv::Size(width, height), cv::INTER_CUBIC);
      startCUDA ( d_img,d_result );
      cv::imshow("Resized Image", d_result);
    }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;

  cv::imshow("Resized Image", d_result);

  cout << diff.count() << endl;
  cout << diff.count()/iter << endl;
  cout << iter/diff.count() << endl;
  
  cv::waitKey();
  return 0;
  
  return 0;
}
