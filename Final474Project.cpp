// This is the ELEC 474 final course project/take home exam. Steven Crosby (#20011059)

// C++ Standard Libraries
#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>

// OpenCV Import
#include <opencv2/opencv.hpp>

// Use OpenCV namespace
using namespace cv;
using namespace std;

// Create a struct to hold indices of two well-matched (overlapping) images & lists of the keypoints of matched features on both images
struct Pair
{
	int id1, id2;
	vector<Point2f> keyl1, keyl2;
};

// Function declarations
vector<String> getImages(String path);
vector<Pair> pickOverlap(vector<String> listOfImages, int multiplier, int thresh, int inlierThresh, int maxMatches, int n);
int pickBase(vector<String> listOfImages, vector<Pair> overlapping);
Mat panorama(vector<String> listOfImages, vector<Pair> overlapping, int base, int padding_top, int padding_side, int toggle_2d);
Mat fixLight(Mat image);

int main()
{
	vector<String> listOfImages;
	vector<Pair> overlapped;
	int baseIdx;
	Mat pano;

	listOfImages = getImages("office2/*.jpg");
	overlapped = pickOverlap(listOfImages, 1, 100, 170, 10, 2000);
	baseIdx = pickBase(listOfImages, overlapped);
	pano = panorama(listOfImages, overlapped, baseIdx, 1000, 4000, 1);
	imwrite("Office_Pano_2D.jpg", pano);
}

// This function retrieves all images from a folder and returns them in a vector of strings of the file names.
vector<String> getImages(String path)
{
	String filePath = path;
	vector<String> listOfImages;
	glob(filePath, listOfImages, false);
	
	return listOfImages;
}

// This function selects the images that overlap & returns a vector of pairs that contain the overlapping indices
vector<Pair> pickOverlap(vector<String> listOfImages, int multiplier, int thresh, int inlierThresh, int maxMatches, int n)
{
	RNG rng((uint64)-1);

	vector<Pair> goodPairs;
	vector<int> hisList1, hisList2, matchedList; // History Lists (of all checked)

	for (int j = 0; j < listOfImages.size(); j++) matchedList.push_back(0);

	int numMatched = 1;
	int nowMatch = 0;
	int idx1;
	while (numMatched < (listOfImages.size()*multiplier))
	{
		if (nowMatch > listOfImages.size() - 1)
			nowMatch = 0;
		idx1 = nowMatch;
		for (int i = 0; i < listOfImages.size(); i++)
		{
			// Randomly select 1 images from the folder
			int idx2 = (int)rng.uniform(0, (int)listOfImages.size());
			if (idx1 == idx2) continue;

			// Check if this set of 2 images has already been checked, and if so, skip
			int flag = 0;
			int flag1 = 0;
			int flag2 = 0;
			for (int j = 0; j < hisList1.size(); j++)
			{
				if (((hisList1[j] == idx1) && (hisList2[j] == idx2)) || ((hisList2[j] == idx1) && (hisList1[j] == idx2)))
				{
					flag = 1;
					//cout << "History repeated " << idx1 << ", " << idx2 << endl;
					break;
				}
			}

			if (flag == 1) continue;
			//if ((flag1 == 1) & (flag2 == 1))
			if ((matchedList[idx1] == maxMatches) && (matchedList[idx2] == maxMatches))
			{
				cout << listOfImages[idx1] << " & " << listOfImages[idx2] << " already matched " << maxMatches << " times. Moving on..." << endl;
				continue;
			}

			// Put the indices of the current images onto the lists to be checked for repetition in further loops
			hisList1.push_back(idx1);
			hisList2.push_back(idx2);

			// Retrieve and resize the images
			Mat image1 = imread(listOfImages[idx1]);
			Mat image2 = imread(listOfImages[idx2]);
			resize(image1, image1, Size(), 0.25, 0.25, INTER_NEAREST);
			resize(image2, image2, Size(), 0.25, 0.25, INTER_NEAREST);

			// Detect matches in the two images
			vector<KeyPoint> keypoints1, keypoints2;
			Mat descriptors1, descriptors2;

			int nfeatures = n;
			float scaleFactor = 1.2f;
			int nlevels = 8;
			int edgeThreshold = 31;
			int firstLevel = 0;
			int WTA_K = 2;
			int patchSize = 31;

			// Feature detector
			Ptr<FeatureDetector> detector = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, ORB::HARRIS_SCORE, patchSize);
			detector->detectAndCompute(image1, Mat(), keypoints1, descriptors1);
			detector->detectAndCompute(image2, Mat(), keypoints2, descriptors2);

			// Match features
			vector<DMatch> matches;
			Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
			matcher->match(descriptors1, descriptors2, matches, Mat());

			// Sort matches by score
			sort(matches.begin(), matches.end());

			// Remove not good matches
			const int numGoodMatches = matches.size() * 0.1f;
			matches.erase(matches.begin() + numGoodMatches, matches.end());

			vector<Point2f> kp1List, kp2List, kp2Trans;
			for (int q = 0; q < matches.size(); q++)
			{
				kp1List.push_back(keypoints1[matches[q].queryIdx].pt);
				kp2List.push_back(keypoints2[matches[q].trainIdx].pt);
			}

			Mat imgTransed, h;

			// Find homography
			h = findHomography(kp2List, kp1List, RANSAC);
			perspectiveTransform(kp2List, kp2Trans, h);

			int inliers = 0;

			for (int q = 0; q < kp1List.size(); q++)
			{
				float piecex, piecey, diff;
				int kp1x = kp1List[q].x;
				int kp2x = kp2Trans[q].x;
				int kp1y = kp1List[q].y;
				int kp2y = kp2Trans[q].y;

				piecex = pow((abs(kp1x - kp2x)), 2);
				piecey = pow((abs(kp1y - kp2y)), 2);
				diff = sqrt(piecex + piecey);

				if (diff <= thresh) inliers++;
			}

			if (inliers >= inlierThresh)
			{
				cout << listOfImages[idx1] << " & " << listOfImages[idx2] << " OVERLAP w/ inliers: " << inliers << endl;
				Pair current;
				current.id1 = idx1;
				current.id2 = idx2;
				current.keyl1 = kp1List;
				current.keyl2 = kp2List;
				goodPairs.push_back(current);
				matchedList.at(idx1) = matchedList[idx1] + 1;
				matchedList.at(idx2) = matchedList[idx2] + 1;
				break;
			}
			else cout << listOfImages[idx1] << " & " << listOfImages[idx2] << " **DO NOT** overlap. Inliers: " << inliers << endl;

			// Make a vector, goodPairs, of pairs of corresponding well-matched images
		}
		numMatched++;
		nowMatch++;
	}

	return goodPairs;
}

// This looks at the overlapping pairs and finds the most common image, and returns its index as a 'base' in the panorama
int pickBase(vector<String> listOfImages, vector<Pair> overlapping)
{
	int countMax = 0;
	int base = -1;
	for (int i = 0; i < listOfImages.size(); i++)
	{
		int count = 0;
		for (int j = 0; j < overlapping.size(); j++)
		{
			if ((overlapping[j].id1 == i) | (overlapping[j].id2 == i)) count++;
		}
		if (count > countMax)
		{
			base = i;
			countMax = count;
		}
	}
	cout << "Base is image w/ index " << base << endl;
	return base;
}

// This stitches images previously deemed 'good matches' together to create the panorama
Mat panorama(vector<String> listOfImages, vector<Pair> overlapping, int base, int padding_top, int padding_side, int toggle_2d)
{
	RNG rng((uint64)-1);
	Mat image1 = imread(listOfImages[base]); //image1 is the CURRENT BASE; "base" to start, and then the current pano for future iterations
	Mat image2;
	vector<int> alreadyDone;
	alreadyDone.push_back(base);
	int lastAdded = base;
	resize(image1, image1, Size(), 0.25, 0.25, INTER_NEAREST);
	//image1 = fixLight(image1);
	copyMakeBorder(image1, image1, padding_top, padding_top, padding_side, padding_side, BORDER_CONSTANT, Scalar(0));
	Mat next_base = image1.clone();
	int out_of_options = 1;
	int nowAdd;
	int keepGoing = 0;
	//int hisIdx[20];
	array<int, 30> hisIdx;
	int going = 0;
	int upoint = 0;

	Mat h1(3, 3, CV_64FC3);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			h1.at<double>(i, j) = 1.0;
	}
		
	vector<Mat> history;
	for (int j = 0; j < listOfImages.size(); j++)
	{
		Mat temp;
		history.push_back(temp);
		hisIdx[j] = 0;
	}
	int q = 0;
	int ecount = 0;

	hisIdx[base] = 1;
	history.at(base) = h1;
	int end = 0;

	while (q < (listOfImages.size() - 1))
	{
		vector<Point2f> pointsBase, pointsTrans;
		Mat img2Transed;
		keepGoing = 0;
		cout << "Looking for image to stitch to " << lastAdded << endl;
		for (int r = 0; r < overlapping.size(); r++)
		{
			if (overlapping[r].id1 == lastAdded)
			{
				nowAdd = overlapping[r].id2;
				keepGoing = 1;
				q++;
				upoint = 0;
				cout << "Found image to stitch: " << nowAdd << endl;

				for (int i = 0; i < overlapping[r].keyl1.size(); i++)
				{
					pointsBase.push_back(overlapping[r].keyl1[i]);
					pointsTrans.push_back(overlapping[r].keyl2[i]);
				}
				overlapping.erase(overlapping.begin() + r);
				ecount = 0;
				break;
			}
			else if (overlapping[r].id2 == lastAdded)
			{
				nowAdd = overlapping[r].id1;
				keepGoing = 1;
				q++;
				upoint = 0;
				cout << "Found image to stitch: " << nowAdd << endl;
				for (int i = 0; i < overlapping[r].keyl1.size(); i++)
				{
					pointsBase.push_back(overlapping[r].keyl2[i]);
					pointsTrans.push_back(overlapping[r].keyl1[i]);
				}
				overlapping.erase(overlapping.begin() + r);
				ecount = 0;
				break;
			}
		}
		
		if (keepGoing == 0)
		{
			if (ecount == 0)
			{
				cout << "Found no image to stitch! Try with base, " << base << endl;
				nowAdd = base;
				ecount++;
			}
			else if (upoint < listOfImages.size()) // used to be ecount < listOfImages.size()
			{
				int yo = 0;
				//cout << "in elseif. upoint is " << upoint << endl;
				for (int u = upoint; u < listOfImages.size(); u++)
				{
					if (hisIdx[u] != 0)
					{
						nowAdd = u;
						upoint = u+1;
						ecount++;
						cout << "Found no image to stitch! Try with past image, " << u << endl;
						yo = 1;
						break;
					}
					if (yo == 1) break;
					yo = 2;
				}
				if (yo == 2)
				{
					cout << "yo" << endl;
					break;
				}
					
			}
			else
			{
				cout << "Still found no image to stitch! Ending program." << endl;
				break;
			}

		}
		if (hisIdx[nowAdd] != 0)
		{
			cout << "Image " << nowAdd << " already stitched" << endl;
			img2Transed = history[nowAdd];
			keepGoing = 0;
		}
		else
		{
			image2 = imread(listOfImages[nowAdd]);
			resize(image2, image2, Size(), 0.25, 0.25, INTER_NEAREST);

			copyMakeBorder(image2, image2, padding_top, padding_top, padding_side, padding_side, BORDER_CONSTANT, Scalar(0));

			for (int i = 0; i < pointsBase.size(); i++)
			{
				pointsBase[i].x += padding_side;
				pointsBase[i].y += padding_top;
				pointsTrans[i].x += padding_side;
				pointsTrans[i].y += padding_top;
			}

			if(lastAdded != base)
				perspectiveTransform(pointsBase, pointsBase, history.at(lastAdded));

			Mat img1Transed, h;

			// Find homography
			//h = findHomography(pointsTrans, pointsBase, RANSAC);

			if (toggle_2d == 1)
			{
				h = estimateAffine2D(pointsTrans, pointsBase);
				warpAffine(image2, img2Transed, h, img2Transed.size(), 1, 0, 0.1);
			}
			else
			{
				// Find homography
				h = findHomography(pointsTrans, pointsBase, RANSAC);
				warpPerspective(image2, img2Transed, h, image2.size(), 1, 0, 0);
			}

			Mat imgPan;
			img2Transed = fixLight(img2Transed);

			for (int r = 0; r < image1.rows; r++)
			{
				for (int c = 0; c < image1.cols; c++)
				{
					if ((image1.at<Vec3b>(r, c)[0] == 0) && (image1.at<Vec3b>(r, c)[1] == 0) && (image1.at<Vec3b>(r, c)[2] == 0))
					{
						image1.at<Vec3b>(r, c)[0] = img2Transed.at<Vec3b>(r, c)[0];
						image1.at<Vec3b>(r, c)[1] = img2Transed.at<Vec3b>(r, c)[1];
						image1.at<Vec3b>(r, c)[2] = img2Transed.at<Vec3b>(r, c)[2];
					}
				}
			}
			hisIdx[nowAdd] = 1;
			history.at(nowAdd) = h;
			//q++;
		}
		lastAdded = nowAdd;
		next_base = img2Transed.clone();
	}

	cout << q+1 << " images stitched of " << listOfImages.size() << endl;
	namedWindow("Panorama", WINDOW_KEEPRATIO);
	imshow("Panorama", image1);
	waitKey();

	history.clear();
	listOfImages.clear();
	overlapping.clear();
	alreadyDone.clear();

	return image1;
}

Mat fixLight(Mat image)
{
	Mat yuv;
	cvtColor(image, yuv, COLOR_BGR2YUV);
	vector<Mat> channels;
	split(yuv, channels);

	equalizeHist(channels[0], channels[0]);

	Mat result;
	merge(channels, yuv);
	cvtColor(yuv, result, COLOR_YUV2BGR);
	return result;
}