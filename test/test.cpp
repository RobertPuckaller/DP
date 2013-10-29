#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
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
	const int max_images_val = 20;
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

	FastFeatureDetector detector(15);
	vector<KeyPoint> keypoints;
	Ptr<DescriptorExtractor > extractor(new SiftDescriptorExtractor());
	Mat img;
	Mat descriptors;
	Mat training_descriptors;

	// computing descriptors
	cout << "Computing descriptors..." << endl;
	for (size_t pos = 0; pos < train.size(); pos++)
	{
		cout << pos << " ";
		img = imread(train[pos]);
		if (img.empty())
		{
			printf("Can't read image\n");
			return -1;
		}
		detector.detect(img, keypoints); // detect keypoints
		extractor->compute(img, keypoints, descriptors); // compute descriptors
		training_descriptors.push_back(descriptors);
	}

	cout << endl << training_descriptors.rows << endl;

	ofstream myfile;
	myfile.open("training_descriptors.txt");
	myfile << training_descriptors;
	myfile.close();

	// creating vocabulary
	cout << "Clustering..." << endl;
	BOWKMeansTrainer bowtrainer(20); //num clusters
	bowtrainer.add(training_descriptors);
	Mat vocabulary = bowtrainer.cluster();

	//ofstream myfile;
	myfile.open("vocabulary.txt");
	myfile << vocabulary;
	myfile.close();
	
	// creating histograms
	Mat response_hist;
	map<string,Mat> classes_training_data;
	vector<string> classes_names;
	Ptr<DescriptorMatcher > matcher(new FlannBasedMatcher());
	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	cout << "Creating histograms..." << endl;
	for (size_t pos = 0; pos < train.size(); pos++)
	{
		cout << pos << " ";
		img = imread(train[pos]);
		if (img.empty())
		{
			printf("Can't read image\n");
			return -1;
		}
		detector.detect(img, keypoints); // detect keypoints
		bowide.compute(img, keypoints, response_hist);
		if (classes_training_data.count(train_evaluation[pos]) == 0)
		{ //not yet created...create class 1 or -1
			classes_training_data[train_evaluation[pos]].create(0,response_hist.cols,response_hist.type());
			classes_names.push_back(train_evaluation[pos]);
		}
		classes_training_data[train_evaluation[pos]].push_back(response_hist);
	}
	cout << endl;

	myfile.open("training_data1.txt");
	myfile << classes_training_data["1"];
	myfile.close();

	myfile.open("training_data-1.txt");
	myfile << classes_training_data["-1"];
	myfile.close();

	map<string,CvSVM> classes_classifiers;

	for (size_t i = 0; i < classes_names.size(); i++)
	{
		string class_ = classes_names[i];

		Mat samples(0,response_hist.cols,response_hist.type());
		Mat labels(0,1,CV_32FC1);

		//copy class samples and label
		samples.push_back(classes_training_data[class_]);
		Mat class_label = Mat::ones(classes_training_data[class_].rows, 1, CV_32FC1);
		labels.push_back(class_label);

		//copy rest samples and label
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

	for (size_t pos = 0; pos < val.size(); pos++)
	{
		cout << pos << endl;
		img = imread(val[pos]);
		if (img.empty())
		{
			printf("Can't read image\n");
			return -1;
		}
		detector.detect(img, keypoints); // detect keypoints
		bowide.compute(img, keypoints, response_hist);

		float minf = FLT_MAX; string minclass;
		for (map<string,CvSVM>::iterator it = classes_classifiers.begin(); it != classes_classifiers.end(); ++it) {
			float res = (*it).second.predict(response_hist,false);
			cout << "class: " << (*it).first << ", response: " << res << endl;
		}
		cout << "------------------------" << endl;
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