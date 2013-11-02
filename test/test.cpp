#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/ml/ml.hpp"
#include "timer.h"
#include "progressor.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Timer t("Image classification");

    if (argc != 3)
    {
		cerr << "program <train.txt> <val.txt>" << endl;
        return -1;
    }

	vector<string> train, val, train_evaluation, val_evaluation;
	string path(argv[1]);
	path = path.substr(0, path.find("ImageSets")) + "JPEGImages\\";

	// read train file
	ifstream file(argv[1], ios::in);
	string file_name, contains;
	const int max_images_train = 200;
	const int max_images_val = 50;
	int count = 0;

	if (!file)
		cerr << "Can't open " << endl;

	while (file >> file_name >> contains)
	{
		count++;
		if (count > max_images_train) break;
		train.push_back(path + file_name + ".jpg");
		train_evaluation.push_back(contains);
	}
	file.close();
	t.report("reading train data");

	// read validate file
	file.open(argv[2], ios::in);
	if (!file)
	cerr << "Can't open " << endl;

	count = 0;
	while (file >> file_name >> contains)
	{
		count++;
		if (count > max_images_val) break;
		val.push_back(path + file_name + ".jpg");
		val_evaluation.push_back(contains);
	}	
	file.close();
	t.report("reading validation data");

	FastFeatureDetector detector(15);
	vector<KeyPoint> keypoints;
	Ptr<DescriptorExtractor > extractor(new SiftDescriptorExtractor());
	Mat img;
	Mat descriptors;
	Mat training_descriptors;

	// computing descriptors
	Progressor progress(3);
	for (size_t pos = 0; pos < 3; pos++)
	{
		img = imread(train[pos]);
		if (img.empty())
		{
			printf("Can't read image\n");
			return -1;
		}
		detector.detect(img, keypoints); // detect keypoints
		extractor->compute(img, keypoints, descriptors); // compute descriptors
		training_descriptors.push_back(descriptors);
		progress.reportNext("computing descriptors...");
	}
	t.report("computing descriptors");

	ofstream myfile;
	myfile.open("training_descriptors.out");
	myfile << training_descriptors;
	myfile.close();
	t.report("writing training descriptors");

	// creating vocabulary
	cout << "clustering...";
	BOWKMeansTrainer bowtrainer(1000); // num clusters
	bowtrainer.add(training_descriptors);
	Mat vocabulary = bowtrainer.cluster();
	cout << "                \r" << flush;
	t.report("clustering");

	// ofstream myfile;
	myfile.open("vocabulary.out");
	myfile << vocabulary;
	myfile.close();
	t.report("writing vocabulary");
	
	// creating histograms
	Mat response_hist;
	map<string,Mat> classes_training_data;
	vector<string> classes_names;
	Ptr<DescriptorMatcher > matcher(new FlannBasedMatcher());
	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	progress.reset(3);
	for (size_t pos = 0; pos < 3; pos++)
	{
		img = imread(train[pos]);
		if (img.empty())
		{
			printf("Can't read image\n");
			return -1;
		}
		detector.detect(img, keypoints); // detect keypoints
		bowide.compute(img, keypoints, response_hist);
		if (classes_training_data.count(train_evaluation[pos]) == 0)
		{ // not yet created...create class 1 or -1
			classes_training_data[train_evaluation[pos]].create(0,response_hist.cols,response_hist.type());
			classes_names.push_back(train_evaluation[pos]);
		}
		classes_training_data[train_evaluation[pos]].push_back(response_hist);
		progress.reportNext("creating histograms...");
	}
	cout << "                \r" << flush;
	t.report("creating histograms");

	myfile.open("training_data1.out");
	myfile << classes_training_data["1"];
	myfile.close();
	t.report("writing class 1 histograms");

	myfile.open("training_data-1.out");
	myfile << classes_training_data["-1"];
	myfile.close();
	t.report("writing class -1 histograms");

	map<string,CvSVM> classes_classifiers;

	for (size_t i = 0; i < classes_names.size(); i++)
	{
		string class_ = classes_names[i];

		Mat samples(0,response_hist.cols,response_hist.type());
		Mat labels(0,1,CV_32FC1);

		// copy class samples and label
		samples.push_back(classes_training_data[class_]);
		Mat class_label = Mat::ones(classes_training_data[class_].rows, 1, CV_32FC1);
		labels.push_back(class_label);

		// copy rest samples and label
		for (map<string,Mat>::iterator it1 = classes_training_data.begin(); it1 != classes_training_data.end(); ++it1) {
			string not_class_ = (*it1).first;
			if (not_class_.compare(class_) == 0) continue; //skip class itself
			samples.push_back(classes_training_data[not_class_]);
			class_label = Mat::zeros(classes_training_data[not_class_].rows, 1, CV_32FC1);
			labels.push_back(class_label);
		}

		Mat samples_32f;
		samples.convertTo(samples_32f, CV_32F);
		if (samples.rows == 0) continue; //phantom class?!
		classes_classifiers[class_].train(samples_32f,labels);		
	}
	t.report("classifier training");

	// classification
	myfile.open("results.out");
	progress.reset(val.size());
	for (size_t pos = 0; pos < val.size(); pos++)
	{
		myfile << pos << endl;
		img = imread(val[pos]);
		if (img.empty())
		{
			printf("Can't read image\n");
			return -1;
		}
		detector.detect(img, keypoints); // detect keypoints
		bowide.compute(img, keypoints, response_hist);

		for (map<string,CvSVM>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
			float res = (*it).second.predict(response_hist,false);
			myfile << "class: " << (*it).first << ", response: " << res << endl;
		}
		myfile << "------------------------" << endl;
		progress.reportNext("writing results...");
	}
	t.report("writing results");
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
	*/
    waitKey(0);
	
}