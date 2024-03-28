#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;
using namespace cv;

void startCUDA ( cv::cuda::GpuMat& src,cv::cuda::GpuMat& dst, 
                            float tau, float alpha );

// ./imagePDEcuda grenouille.jpg result.png 500 .10 .05

int main( int argc, char** argv )
{
  cv::Mat_<Vec3f> h_imaRGB = cv::imread(argv[1]);
  cv::Mat_<Vec3f> h_result  ( h_imaRGB.rows, h_imaRGB.cols ); 

  for (int i=0;i<h_imaRGB.rows;i++)
    for (int j=0;j<h_imaRGB.cols;j++)
      for (int c=0;c<3;c++)
	    h_imaRGB(i,j)[c] /= 255.0;

  cv::cuda::GpuMat d_imaRGB,d_result;

  d_imaRGB.upload ( h_imaRGB );
  d_result.upload ( h_result );

  int iter = atoi(argv[3]);
  float tau = atof(argv[4]);
  float alpha =  atof(argv[5]);
  
  auto begin = chrono::high_resolution_clock::now();
  
  for (int i=0;i<iter/2;i++)
    {
      startCUDA ( d_imaRGB,d_result, tau, alpha );
      startCUDA ( d_result, d_imaRGB, tau, alpha );
    }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;
  
  cout << diff.count() << endl;
  cout << diff.count()/iter << endl;
  cout << iter/diff.count() << endl;
  
cout << d_imaRGB.cols << endl;

  d_imaRGB.download(h_imaRGB);
  
  for (int i=0;i<h_imaRGB.rows;i++)
    for (int j=0;j<h_imaRGB.cols;j++)
      for (int c=0;c<3;c++)
	    h_imaRGB(i,j)[c] *= 255.0;
  
  imwrite ( argv[2], h_imaRGB );

  return 0;
}
