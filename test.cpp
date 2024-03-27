#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::Mat img = cv::imread("test.tif", cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    cv::namedWindow("Window", cv::WINDOW_NORMAL);
    cv::imshow("Window", img);

    cv::waitKey(0);
    return 0;
}