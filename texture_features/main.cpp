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



int display_img(Mat input_img, string name){
    
    //create a window
    namedWindow(name, CV_WINDOW_AUTOSIZE);
    
    //display the image
    imshow(name, input_img);
    
    //wait infinite time for a keypress
    waitKey(0);
    
    //destroy the window
    destroyWindow(name);
    
    return 0;
}


int variance(Mat input_img, int m, float norm, bool cuda_support){
    
    short unsigned int channel_count = input_img.channels();
    Mat chan[3];
    Mat output[3];
    Mat mergedResult;
    split(input_img, chan);
    unsigned int rows = input_img.rows;
    unsigned int cols = input_img.cols;
    vector<uchar> result(rows*cols);
    norm = 255.0 / norm;
    
    
    //Number of pixel in a kernel m*m
    int M = (float) (m * m);

    
    //iterate over all channels(B,G,R) or channels(H,S,V)
    for (int c=0; c < channel_count; c++){
        unsigned int r = 0;
        //int *environment = new int[M];
        int environment[9];
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
        //output[c] = Mat(rows,cols, CV_8U, result);
    }
    display_img(output[0], "Blue");
    display_img(output[1], "Green");
    display_img(output[2], "Red");
    return 0;
}




int main(int argc, const char * argv[]) {
    Mat img = imread("/Users/tobiashofer/Desktop/test.tif", CV_LOAD_IMAGE_UNCHANGED);
    if(img.empty()){
            cout << "Error: Image cannot be loaded" << endl;
            system("pause");
            return -1;
        }
    variance(img, 3, 1, true);
    return 0;
}
