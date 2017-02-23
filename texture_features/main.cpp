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



int display_img(Mat src, string name, bool resize_img=true){
    if (resize_img){
        //create a window
        namedWindow(name, CV_WINDOW_NORMAL);
        //resize image
        resize(src, src, Size(900,900));
    }else{
        //create a window
        namedWindow(name, CV_WINDOW_AUTOSIZE);
        
    }
    
    
    //display the image
    imshow(name, src);
    
    //wait infinite time for a keypress
    waitKey(0);
    destroyWindow(name);
    return 0;
}


Mat variance(Mat& src, int m, bool cuda_support){
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
        vector<uchar> tex_window(M);
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
                        if (x >= 0 && x <= rows && y >= 0 && y <= cols) {
                            mean += (double)chan[c].at<uchar>(x,y);
                            tex_window[e] = chan[c].at<uchar>(x,y);
                        }else{
                            tex_window[e] = -1;
                            M_corner -= 1;
                        }
                        e++;
                    }
                }
                mean = mean/M_corner;
                //variance
                double variance=0;
                for(int l=0;l<M;l++){
                    if(tex_window[l] >= 0){
                        variance += pow(tex_window[l] - mean,2);
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
    return merged_result;
}

Mat edge_density(Mat& src, int m, char basis, bool cuda_support){
    //kernel for edge detection must be odd
    int kernel_size = 3;
    
    Mat output, src_gray, dst, abs_dst, grad;
    
    //Apply Gaussian Blurr to reduce the noice
    GaussianBlur(src, src, Size(5,5), 0, 0, BORDER_DEFAULT);
    //display_img(src, "gaussian_blurr");
    
    // Convert the image to grayscale
    cvtColor(src, src_gray, CV_BGR2GRAY);
    //display_img(src_gray, "grayscale");
    
    //Sobel edge detection as basis
    if (basis == 'S'){
        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;

        Mat grad_x, grad_y, abs_grad_y, abs_grad_x;
        //gradient X
        Sobel(src_gray, grad_x, ddepth, 1, 0, kernel_size, scale, delta, BORDER_DEFAULT);
        //gradient Y
        Sobel(src_gray, grad_y, ddepth, 0, 1, kernel_size, scale, delta, BORDER_DEFAULT);
        //convert partial results to CV_8U
        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);
        //approximate the gradient
        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
        display_img(grad, "Sobel");
        
    }
    //Laplacian edge detection as basis
    if (basis == 'L'){
        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;

        Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
        //abs_dst = absolute_value(dst)
        convertScaleAbs(dst, grad);
        display_img(grad, "Laplacian");
    }
  
    //Number of pixel in a kernel m*m
    int M = (float) (m * m);
    unsigned int rows = grad.rows;
    unsigned int cols = grad.cols;
    vector<uchar> result(rows*cols);
    unsigned int r = 0;
    //TODO:Delete vector ?
    int edge_count = 0;
    double M_corner;
    
    //Iterating through bitmap of a single channel
    for(int i=0; i < rows; i++){
        for(int j=0; j < cols; j++){
            //kernel convolution for given size m
            M_corner = M;
            for(int x=i-(m-1)/2;x<=i+(m-1)/2;x++){
                for(int y=j-(m-1)/2;y<=j+(m-1)/2;y++){
                    if (x >= 0 && x <= rows && y >= 0 && y <= cols) {
                        int pixel = grad.at<uchar>(x,y);
                        if(pixel >= 30){
                            edge_count++;
                        }
                    }else{
                        M_corner -= 1;
                    }
                }
            }
            //edge_density value between [0, 1]
            double density = 1.0/((double)M_corner-1.0) * edge_count;
            result[r] = density*255;
            edge_count = 0;
            r++;
        }
    }
    output = Mat (rows,cols, CV_8UC1);
    memcpy(output.data, result.data(), result.size()*sizeof(uchar));
    display_img(output, "edge_density");
    return output;
}

Mat autocorrelation(Mat& src, int m, int dx, int dy, bool cuda_support){
    Mat src_gray, output;
    /// Convert the image to grayscale
    cvtColor(src, src_gray, CV_BGR2GRAY);

    unsigned int rows = src.rows;
    unsigned int cols = src.cols;
    
    //M: pixel in a kernel m*m
    int M = (float) (m * m);
    
    //auto corr
    vector<uchar> result(rows*cols);
    //sum quadratic diff from mean
    vector<uchar> result_(rows*cols);
    //sum diff from mean
    vector<uchar> result__(rows*cols);
    unsigned int r = 0;
    //TODO:Delete vector ?
    vector<uchar> tex_window(M);
    vector<uchar> tex_window_(M);

    unsigned int e;
    double M_corner;
    
    //Iterating through bitmap of a single channel
    for(int i=0; i < rows; i++){
        for(int j=0; j < cols; j++){
            //kernel convolution for given size m
            double mean = 0;
            e = 0;
            M_corner = M;
            //texture window U
            for(int x=i-(m-1)/2;x<=i+(m-1)/2;x++){
                for(int y=j-(m-1)/2;y<=j+(m-1)/2;y++){
                    if (x >= 0 && x <= rows && y >= 0 && y <= cols) {
                        mean += (double)src_gray.at<uchar>(x,y);
                        tex_window[e] = src_gray.at<uchar>(x,y);
                    }else{
                        tex_window[e] = -1;
                        M_corner -= 1;
                    }
                    e++;
                }
            }
            mean = mean/M_corner;
            double mean_ = 0;
            e = 0;
            M_corner = M;
            //texture window U': U --> vec[dx, dy]  U'
            for(int x=(i-(m-1)/2) + dx; x<=(i+(m-1)/2) + dx; x++){
                for(int y=(j-(m-1)/2) + dy; y<=(j+(m-1)/2) + dy; y++){
                    if (x >= 0 && x <= rows && y >= 0 && y <= cols) {
                        mean_ += (double)src_gray.at<uchar>(x,y);
                        tex_window_[e] = src_gray.at<uchar>(x,y);
                    }else{
                        tex_window_[e] = -1;
                        M_corner -= 1;
                    }
                    e++;
                }
            }
            mean_ = mean_/M_corner;
            
            //auto corr
            int numerator = 0;
            int denominator = 0;
            int denominator_ = 0;
            for(int l=0;l<M;l++){
                if(tex_window[l] >= 0 && tex_window_[l] >= 0){
                    numerator += (tex_window[l] - mean) * (tex_window_[l] - mean_);
                    denominator += pow(tex_window[l] - mean, 2);
                    denominator_ += pow(tex_window_[l] - mean_, 2);
                }
            }
            double autocorrelation = 0;
            autocorrelation = numerator / sqrt(denominator * denominator_);
            //Mapping to grayscale
            autocorrelation = 127 * (autocorrelation + 1);
            result[r] = autocorrelation;
            r++;
        }
    }
    output = Mat (rows,cols, CV_8UC1);
    memcpy(output.data, result.data(), result.size()*sizeof(uchar));
    display_img(output, "auto_corr");
    return output;
}

/**
    Converts BGR Imgae to HSV Image.
    Be aware that the OpenCV HSV values are:
        H from 0 - 180
        S and V from 0 - 255
    OpenCV is unable to show HSV Images normally, this distorts the color
    because they are being interpreted as BGR.
*/
Mat convert_bgr_to_hsv(Mat img){
    Mat result;
    cvtColor(img, result, CV_BGR2HSV);
    return result;
}



Mat get_gaussian_pyramid(Mat src, int lvl, bool auto_expand=true){
    // Convert the image to grayscale
    cvtColor(src, src, CV_BGR2GRAY);
    //Make sure its divisible by 2^n
    Mat result = src;
    for (int i =0; i < lvl; i++){
        pyrDown(result, result);
    }
    if (auto_expand){
        for (int i =0; i < lvl; i++){
            pyrUp(result, result);
        }
    }
    return result;
}

Mat get_laplacian_pyramid(Mat src, int lvl, bool auto_expand=true){
    Mat result;
    Mat g,G;
    G = get_gaussian_pyramid(src, lvl, false);
    pyrDown(G, g);
    pyrUp(g, g);
    result = G - g;
    if (auto_expand){
        for (int i =0; i < lvl; i++){
            pyrUp(result, result);
        }
    }
    return result;
}


int main(int argc, const char * argv[]) {
    Mat img = imread("/Users/tobiashofer/Desktop/test.tif", CV_LOAD_IMAGE_UNCHANGED);
    if(img.empty()){
            cout << "Error: Image cannot be loaded" << endl;
            system("pause");
            return -1;
        }
    
    
    //TODO create Roi, so that the Image size is part of 2^n (recommende for Laplacian ) ???
    resize(img, img, Size(2048, 2048));
    //display_img(img, "reimg");
  
    Mat l0, l2, l3;
    l0 = get_laplacian_pyramid(img, 0);
    l2 = get_laplacian_pyramid(img, 1);
    l3 = get_laplacian_pyramid(img, 3);
    
    display_img(l0, "l0");
    display_img(l2, "l2");
    display_img(l3, "l3");
    
    
    //Testing Tesxture Features on original Image
    Mat result_var, result_edge_dens, result_auto, result;
    result_var = variance(img, 5, false);
    Mat test = edge_density(img, 5, 'L', false);
    result_edge_dens = edge_density(img, 5, 'S', false);
    result_auto = autocorrelation(img, 7, 0, 5, false);
    
    display_img(result_var, "var_org");
    display_img(result_edge_dens, "edge_dens_org");
    display_img(result_auto, "auto_org");
    
    
    //Testing Tesxture Features on L33
    Mat result_var_l, result_edge_dens_l, result_auto_l;
    result_var_l = variance(img, 5, false);
    result_edge_dens_l = edge_density(img, 5, 'S', false);
    result_auto_l = autocorrelation(img, 7, 0, 5, false);
    
    display_img(result_var_l, "var_l");
    display_img(result_edge_dens_l, "edge_dens_l");
    display_img(result_auto_l, "auto_l");

    
    
    
//    imwrite("/Users/tobiashofer/Desktop/result_var.tif", result_var);
//    imwrite("/Users/tobiashofer/Desktop/result_edge_dens.tif", result_edge_dens);
//    imwrite("/Users/tobiashofer/Desktop/result_auto.tif", result_auto);
 
    
    
//    Mat g1, g2, g3;
//    g1 = get_gaussian_pyramid(img, 1);
//    g2 = get_gaussian_pyramid(img, 4);
//    g3 = get_gaussian_pyramid(img, 5);
//    
//    display_img(img, "g0", false);
//    display_img(g1, "g1", false);
//    display_img(g2, "g2", false);
//    display_img(g3, "g3", false);

    
    

    
    
//    Mat hsv;
//    
//    hsv = convert_bgr_to_hsv(img);
//    display_img(hsv, "hsv");
//    cvtColor(hsv, hsv, CV_HSV2BGR);
//    display_img(hsv, "hsv");

    return 0;
}



