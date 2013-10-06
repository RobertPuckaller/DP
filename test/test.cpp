#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
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

	// detect keypoints
	FastFeatureDetector detector(15);
	vector<KeyPoint> keypoints1;
	detector.detect(img1, keypoints1);

	// draw keypoints
	Mat imgKeypoints1;
	drawKeypoints( img1, keypoints1, imgKeypoints1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	// show detected (drawn) keypoints
	imshow("Keypoints 1", imgKeypoints1 );

    waitKey(0);
}