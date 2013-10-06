#include <stdio.h>
#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        return -1;
    }

    Mat img1 = imread(argv[1]);
    if(img1.empty())
    {
        printf("Can't read image\n");
        return -1;
    }

	namedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", img1 );                   // Show our image inside it.

    waitKey(0);
}