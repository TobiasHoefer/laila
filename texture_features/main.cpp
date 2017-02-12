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
    int *result = new int[rows*cols];
    
    norm = 255.0 / norm;
    
    
    //Number of pixel in a kernel m*m
    int M = (float) (m * m);

    
    //iterate over all channels(B,G,R) or channels(H,S,V)
    for (int c=0; c < channel_count; c++){
        unsigned int r = 0;
        //int *environment = new int[M];
        int environment[9];
        unsigned int e;
        int M_corner;
        
        //Iterating through bitmap of a single channel
        for(int i=0; i < rows; i++){
            for(int j=0; j < cols; j++){
                
                //kernel convolution for given size m
                int mean = 0;
                e = 0;
                M_corner = M;
                for(int x=i-(m-1)/2;x<=i+(m-1)/2;x++){
                    for(int y=j-(m-1)/2;y<=j+(m-1)/2;y++){
                        if (x >= 0 && y >= 0) {
                            mean += chan[c].at<uchar>(x,y);
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
                int variance=0;
                for(int l=0;l<M;l++){
                    if(environment[l] >= 0){
                        variance += pow(environment[l] - mean,2);
                    }
                }
                int v = 1.0/((double)M_corner-1.0) * variance;
                v = v > 255 ? 255 : v;
                result[r] = v;
                //cout<<result[r]<<endl;
                r++;
            }
        }
        output[c] = Mat(rows,cols, CV_8U, result);
    }
    
    delete [] result;
    //delete [] environment;
    
    //cout<< chan[0]<<endl<<endl;
   // cout<< output[0] <<endl <<endl;

//    display_img(chan[0], "Blue");
    display_img(output[0], "Blue");
    display_img(output[1], "Green");
    display_img(output[2], "Red");
    return 0;
}




int main(int argc, const char * argv[]) {
    
    Mat img = imread("/Users/tobiashofer/Desktop/lena.tiff", CV_LOAD_IMAGE_UNCHANGED);
    if(img.empty()){
            cout << "Error: Image cannot be loaded" << endl;
            system("pause");
            return -1;
        }
    
    //display_img(img);
    variance(img, 7, 1, true);
    
    
        
        return 0;
}
