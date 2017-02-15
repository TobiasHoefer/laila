//
//  main.cpp
//  texture_features
//
//  Created by Tobias Höfer on 06/02/2017.
//  Copyright © 2017 Tobias Höfer. All rights reserved.
//


#include "opencv2/opencv.hpp"
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;



int display_img(Mat& src, string name){
    //create a window
    namedWindow(name, CV_WINDOW_NORMAL);
    
    //resize image
    resize(src, src, Size(1000,1000));
    
    //display the image
    imshow(name, src);
    
    //wait infinite time for a keypress
    waitKey(0);
    destroyWindow(name);
    return 0;
}


int variance(Mat& src, int m, bool cuda_support){
    short unsigned int channel_count = src.channels();
    Mat chan[3];
    split(src, chan);
    Mat output[3];
    Mat merged_result;
    
    unsigned int rows = src.rows;
    unsigned int cols = src.cols;
    
    //Number of pixel in a kernel m*m
    int M = (float) (m * m);

    //iterate over all channels(B,G,R)
    for (int c=0; c < channel_count; c++){
        vector<uchar> result(rows*cols);
        unsigned int r = 0;
        //TODO:Delete vector ?
        vector<uchar> environment(M);
        unsigned int e;
        double M_corner;
        
        //Iterating through bitmap of a single channel
        for(int i=0; i < rows; i++){
            for(int j=0; j < cols; j++){
                
                //kernel convolution for given size m
                double mean = 0;
                e = 0;
                M_corner = M;
                for(int x=i-(m-1)/2;x<=i+(m-1)/2;x++){
                    for(int y=j-(m-1)/2;y<=j+(m-1)/2;y++){
                        if (x >= 0 && y >= 0) {
                            mean += (double)chan[c].at<uchar>(x,y);
                            environment[e] = chan[c].at<uchar>(x,y);
                        }else{
                            environment[e] = -1;
                            M_corner -= 1;
                        }
                        e++;
                    }
                }
                mean = mean/M_corner;
                //variance
                double variance=0;
                for(int l=0;l<M;l++){
                    if(environment[l] >= 0){
                        variance += pow(environment[l] - mean,2);
                    }
                }
                double v = 1.0/((double)M_corner-1.0) * variance;
                v = v > 255.0 ? 255.0 : v;
                result[r] = v;
                r++;
            }
        }
        output[c] = Mat (rows,cols, CV_8UC1);
        memcpy(output[c].data, result.data(), result.size()*sizeof(uchar));
        result.clear();
    }
    display_img(output[0], "Blue");
    display_img(output[1], "Green");
    display_img(output[2], "Red");
    vector<Mat> channels;
    channels.push_back(output[0]);
    channels.push_back(output[1]);
    channels.push_back(output[2]);
    merge(channels, merged_result);
    display_img(merged_result, "RGB_Result");
    return 0;
}

int edge_density(Mat& src, int m, char basis, bool cuda_support){
    short unsigned int channel_count = src.channels();
    Mat chan[3];
    unsigned int rows = src.rows;
    unsigned int cols = src.cols;

    
    //Apply Gaussian Blurr to reduce the noice
    GaussianBlur(src, src, Size(5,5), 0, 0, BORDER_DEFAULT);
    //display_img(src, "gaussian_blurr");
    
    //Sobel edge detection as basis
    if (basis == 'S'){
        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;
    }
    //Laplacian edge detection as basis
    if (basis == 'L'){
        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;
        int kernel_size = 3;
        Mat src_gray, dst, abs_dst;

        /// Convert the image to grayscale
        cvtColor(src, src_gray, CV_BGR2GRAY);
        //display_img(src_gray, "grayscale");
        
        Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
        convertScaleAbs(dst, abs_dst);
        display_img(abs_dst, "Laplacian");
    }
//    split(src, chan);
//
//    split(src, chan);
//    Mat output[3];
//    Mat merged_result;
//    
//    unsigned int rows = src.rows;
//    unsigned int cols = src.cols;
//    
//    //Number of pixel in a kernel m*m
//    int M = (float) (m * m);
//    
//    //iterate over all channels(B,G,R)
//    for (int c=0; c < channel_count; c++){
//        vector<uchar> result(rows*cols);
//        unsigned int r = 0;
//        //TODO:Delete vector ?
//        vector<uchar> environment(M);
//        unsigned int e;
//        double M_corner;
//        
//        //Iterating through bitmap of a single channel
//        for(int i=0; i < rows; i++){
//            for(int j=0; j < cols; j++){
//                
//                //kernel convolution for given size m
//                double mean = 0;
//                e = 0;
//                M_corner = M;
//                for(int x=i-(m-1)/2;x<=i+(m-1)/2;x++){
//                    for(int y=j-(m-1)/2;y<=j+(m-1)/2;y++){
//                        if (x >= 0 && y >= 0) {
//                            mean += (double)chan[c].at<uchar>(x,y);
//                            environment[e] = chan[c].at<uchar>(x,y);
//                        }else{
//                            environment[e] = -1;
//                            M_corner -= 1;
//                        }
//                        e++;
//                    }
//                }
//                mean = mean/M_corner;
//                //variance
//                double variance=0;
//                for(int l=0;l<M;l++){
//                    if(environment[l] >= 0){
//                        variance += pow(environment[l] - mean,2);
//                    }
//                }
//                double v = 1.0/((double)M_corner-1.0) * variance;
//                v = v > 255.0 ? 255.0 : v;
//                result[r] = v;
//                r++;
//            }
//        }
//        output[c] = Mat (rows,cols, CV_8UC1);
//        memcpy(output[c].data, result.data(), result.size()*sizeof(uchar));
//        result.clear();
//    }
//    display_img(output[0], "Blue");
//    display_img(output[1], "Green");
//    display_img(output[2], "Red");
//    vector<Mat> channels;
//    channels.push_back(output[0]);
//    channels.push_back(output[1]);
//    channels.push_back(output[2]);
//    merge(channels, merged_result);
//    display_img(merged_result, "RGB_Result");
    return 0;
}



int main(int argc, const char * argv[]) {
    Mat img = imread("/Users/tobiashofer/Desktop/test.tif", CV_LOAD_IMAGE_UNCHANGED);
    if(img.empty()){
            cout << "Error: Image cannot be loaded" << endl;
            system("pause");
            return -1;
        }
    //variance(img, 5, true);
    edge_density(img, 3, 'L', false);
    return 0;
}
