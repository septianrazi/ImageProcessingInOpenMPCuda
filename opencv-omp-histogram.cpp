#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void computeHistogram ( const cv::Mat_<uchar>& src, cv::Mat_<int>& histo )
{ 
  histo.create ( 1, 256 );
  for (int i=0;i<256;i++)
    histo(0,i) = 0;
 
#pragma omp parallel for
  for (int i=0;i<src.rows;i++)
    for (int j=0;j<src.cols;j++)
// #pragma omp critical
// #pragma omp atomic
      histo(0,src(i,j))++;
}

int main( int argc, char** argv )
{
  cv::Mat_<uchar> source = cv::imread ( argv[1], cv::IMREAD_GRAYSCALE);
  cv::Mat_<int> histogram;
  
  cv::imshow("Source Image", source );

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 1000;

  for (int it=0;it<iter;it++)
    computeHistogram ( source, histogram );

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;

  cout << "Total time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count()/iter << " s" << endl;
  cout << "IPS: " << iter/diff.count() << endl;
    
	for (int i=0;i<histogram.cols;i++)
		cout << i << " : " << histogram (0,i) << endl;

  cv::waitKey();
  return 0;
}
