#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if(argc != 3)
    {
		cerr << "program <train.txt> <val.txt>" << endl;
        return -1;
    }

	vector<string> train, val;
	vector<int> train_evaluation, val_evaluation;
	string path(argv[1]);
	path = path.substr(0, path.find("ImageSets")) + "JPEGImages\\";

	// read train file
	ifstream file(argv[1], ios::in);
	string file_name;
	const int max_images = 100;
	int contains, count = 0;

	if(!file)
		cerr << "Can't open " << endl;

	while( file >> file_name >> contains)
	{
		count++;
		if (count > max_images) break;
		train.push_back(path + file_name + ".jpg");
		train_evaluation.push_back(contains);
	}

	file.close();

	/*
	// read validate file
	file.open(argv[2], ios::in);
	if(!file)
	cerr << "Can't open " << endl;

	count = 0;
	while( file >> file_name >> contains)
	{
	count++;
	if (count > max_images) break;
	val.push_back(file_name + ".jpg");
	val_evaluation.push_back(contains);
	}

	file.close();
	*/

	FastFeatureDetector detector(15);
	vector<KeyPoint> keypoints;
	Ptr<DescriptorExtractor > extractor(new SiftDescriptorExtractor());
	Mat img;
	Mat descriptors;
	Mat training_descriptors;

	// computing descriptors
	for(int pos = 0; pos < train.size(); pos++)
	{
		cout << pos << endl;
		img = imread(train[pos]);
		if(img.empty())
		{
			printf("Can't read image\n");
			return -1;
		}
		detector.detect(img, keypoints); // detect keypoints
		extractor->compute(img, keypoints, descriptors); // compute descriptors
		training_descriptors.push_back(descriptors);
	}

	cout << training_descriptors.rows << endl;

	// creating vocabulary
	BOWKMeansTrainer bowtrainer(20); //num clusters
	bowtrainer.add(training_descriptors);
	Mat vocabulary = bowtrainer.cluster();

	ofstream myfile;
	myfile.open("vocabulary.txt");
	myfile << vocabulary;
	myfile.close();
	
	// creating histograms
	Mat response_hist;
	map<int,Mat> classes_training_data;
	Ptr<DescriptorMatcher > matcher(new FlannBasedMatcher());
	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	for(int pos = 0; pos < train.size(); pos++)
	{
		cout << pos << endl;
		img = imread(train[pos]);
		if(img.empty())
		{
			printf("Can't read image\n");
			return -1;
		}
		detector.detect(img, keypoints); // detect keypoints
		bowide.compute(img, keypoints, response_hist);
		if (classes_training_data.count(train_evaluation[pos]) == 0)
		{ //not yet created...create class 1 or -1
			classes_training_data[train_evaluation[pos]].create(0,response_hist.cols,response_hist.type());
		}
		classes_training_data[train_evaluation[pos]].push_back(response_hist);
   }
    /*
	// detect keypoints
	FastFeatureDetector detector(15);
	vector<KeyPoint> keypoints1;
	detector.detect(img1, keypoints1);

	// draw keypoints
	Mat imgKeypoints1;
	drawKeypoints( img1, keypoints1, imgKeypoints1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	// compute descriptors
    SiftDescriptorExtractor extractor;
    Mat descriptors1;
    extractor.compute(img1, keypoints1, descriptors1);

	// show detected (drawn) keypoints
	imshow("Keypoints 1", imgKeypoints1 );

    waitKey(0);
	*/
}