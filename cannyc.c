// Memory Usage Reduction by using only the pixels required
// Reduction of time in Edge Hysteresis Stage


#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#define SOBELX 1
#define SOBELY 0

int image_height = 0, image_width = 0;
int bordlen = 4;
int bord_image_height = 0, bord_image_width = 0;

float minmax(float *img_array, int option);
void gaussianfilter(int *src, float *dst);
void sobelfilter(float *src, float *dst, int direction);
void imgwrite(char image_name[]);
void imgread(char name[], int *img_data);
void getdim(char name[]);
void matmul(float *src1, float *src2, float *dst);
void getgrad(float *src1, float *src2, float *mag, float *dir);
void nonmaxsup(float *mag, float *dir, float *dst);
void doublethresh(float *nms, int *doublethr, int thresh1, int thresh2);
void edgehyst(int *doublthr, int *cannyfinal);
void mark(int h, int *visitedmap, int *dt, int *dst);
void borderrep(int *img, int *borderimg);
void deborderrep(float *borderimg, float *img);

int main()
{
    char image[] = "lena.jpg";
    imgwrite(image);

    char image_txt[] = "lena.txt";
    getdim(image_txt);
    int img_data[image_height*image_width];
    imgread(image_txt, img_data);

    bord_image_height = image_height + bordlen;
    bord_image_width = image_width + bordlen;

    int *borderimg = (int *)malloc(bord_image_height*bord_image_width*sizeof(int));
    float *gaussian_image = (float *)malloc(bord_image_height*bord_image_width*sizeof(float));
    float *sobel_x = (float *)malloc(bord_image_height*bord_image_width*sizeof(float));
    float *sobel_y = (float *)malloc(bord_image_height*bord_image_width*sizeof(float));
    float *sobel_xdeb = (float *)malloc(image_height*image_width*sizeof(float));
    float *sobel_ydeb = (float *)malloc(image_height*image_width*sizeof(float));
    float *gradmag = (float *)malloc(image_height*image_width*sizeof(float));
    float *graddir = (float *)malloc(image_height*image_width*sizeof(float));
    float *nms_image = (float *)malloc(image_height*image_width*sizeof(float));
    int *doublethr = (int *)malloc(image_height*image_width*sizeof(int));
    int *cannyfinal = (int *)malloc(image_height*image_width*sizeof(int));

    // Border Replicate
    borderrep(img_data, borderimg);
    // Convolution with Gaussian Kernel of size 5*5 and sigma = 1.4
    gaussianfilter(borderimg, gaussian_image);
    // Convolution with Sobel Kernel of size 3*3 in x- and y- directions
    sobelfilter(gaussian_image, sobel_x, SOBELX);
    sobelfilter(gaussian_image, sobel_y, SOBELY);
    //Debordering
    deborderrep(sobel_x, sobel_xdeb);
    deborderrep(sobel_y, sobel_ydeb);
    // Getting Gradient Magnitude and Directions
    getgrad(sobel_xdeb, sobel_ydeb, gradmag, graddir);
    // Non Maximum Suppression for Edge Localisation
    nonmaxsup(gradmag, graddir, nms_image);
    // Double Thresholding to classify weak and strong edges
    doublethresh(nms_image, doublethr, 30, 70);
    // Edge Tracking Algorithm
    edgehyst(doublethr, cannyfinal);

    IplImage *test = cvCreateImage(cvSize(image_width, image_height), IPL_DEPTH_8U, 1);
    uchar *test_data = (uchar *)test->imageData;

    for (int i = 0; i < image_height; i++)
    {
        for (int j = 0; j < image_width; j++)
        {
            *(test_data + i*image_width + j) = cannyfinal[j + i*image_width];
        }
    }

    cvNamedWindow("Out", CV_WINDOW_NORMAL);
    cvShowImage("Out", test);
    cvWaitKey(0);
    cvSaveImage("inter.jpg", test, 0);
    return 0;

}

float minmax(float *img_array, int option)
{
    float val = 0;
    if (option)
    {
        for (int i = 0; i < image_height*image_width; i++)
            val = val > img_array[i] ? val : img_array[i];
    }
    else
    {
        for (int i = 0; i < image_height*image_width; i++)
            val = val < img_array[i]? val : img_array[i];
    }
    return val;
}


void gaussianfilter(int *src, float *dst)
{

    float gaussian[25] = {0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
                          0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
                          0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
                          0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
                          0.003765, 0.015019, 0.023792, 0.015019, 0.003765
                         };

    int kernel_size = 5;

    for (int l = 0; l < bord_image_height*bord_image_width; l++)
    {
        if (l >= bord_image_width*image_height )
            dst[l] = dst[l - bord_image_width];
        else if ((l%bord_image_width) >= (bord_image_width-bordlen) && (l%bord_image_width) < bord_image_width)
            dst[l] = dst[l-1];
        else
        {
            int sum = 0;
            for (int i = 0; i < kernel_size; i++)
            {
                for (int j = 0; j < kernel_size; j++)
                {
                    sum = sum + src[l + j + bord_image_width*i]*gaussian[j + kernel_size*i];
                }
            }
            dst[l] = sum;
        }
    }
}

void sobelfilter(float *src, float *dst, int direction)
{
    int kernel_size = 3;
    for (int l = 0; l < bord_image_height*bord_image_width; l++)
    {
        if (l >= bord_image_width*image_height )
            dst[l] = dst[l - bord_image_width];
        else if ((l%bord_image_width) >= (bord_image_width-bordlen) && (l%bord_image_width) < bord_image_width)
            dst[l] = dst[l-1];
        else
        {
            float sum = 0;
            if (direction)
            {
                int sobel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
                //int sobel[5][5] = {1, 2, 0, -2, -1, 4, 8, 0, -8, -4, 6, 12, 0, -12, -6, 4, 8, 0, -8, -4, 1, 2, 0, -2, -1};
                for (int i = 0; i < kernel_size; i++)
                {
                    for (int j = 0; j < kernel_size; j++)
                    {
                        sum = sum + src[l + j + bord_image_width*i]*sobel[j + kernel_size*i];
                    }
                }
                dst[l] = sum;
            }
            else
            {
                int sobel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
                //int sobel[5][5] = {-1, -4, -6, -4, -1, -2, -8, -12, -8, -2, 0, 0, 0, 0, 0, 2, 8, 12, 8, 2, 1, 4, 6, 4, 1};
                for (int i = 0; i < kernel_size; i++)
                {
                    for (int j = 0; j < kernel_size; j++)
                    {
                        sum = sum + src[l + j + bord_image_width*i]*sobel[j + kernel_size*i];
                    }
                }
                dst[l] = sum;
            }
        }
    }
}

void matmul(float *src1, float *src2, float *dst)
{
    for (int i = 0; i < image_height*image_width; i++)
        dst[i] = src1[i] * src2[i];
}


void imgwrite(char image_name[])
{
    IplImage *input_image;
    char image_text[20];
    strcpy(image_text, image_name);
    int m = 0;
    while(image_text[m] != '.')
    {
        m++;
    }
    image_text[m+1] = 't';
    image_text[m+2] = 'x';
    image_text[m+3] = 't';
    image_text[m+4] = '\0';

    input_image = cvLoadImage(image_name, 0);

    FILE *input_image_data_text;
    input_image_data_text = fopen(image_text, "w");

    uchar *input_image_data = (uchar *) input_image->imageData;

    fprintf(input_image_data_text, "%d\n", input_image->height);
    fprintf(input_image_data_text, "%d\n", input_image->width);

    int i = 0;
    while(i < input_image->width*input_image->height)
    {
        fprintf(input_image_data_text, "%d\n", input_image_data[i]);
        i++;
    }

    fclose(input_image_data_text);
}


void imgread(char image_text[], int *img_data)
{
    FILE *temp;
    temp = fopen(image_text, "r");

    fscanf(temp, "%d", &image_height);
    fscanf(temp, "%d", &image_width);

    int k = 0;
    while(!(k == image_width*image_height))
    {
        fscanf(temp, "%d", &img_data[k]);
        k++;
    }
    fclose(temp);
}

void getdim(char image_text[])
{
    FILE *temp;
    temp = fopen(image_text, "r");

    fscanf(temp, "%d", &image_height);
    fscanf(temp, "%d", &image_width);

    fclose(temp);
}


void getgrad(float *src1, float *src2, float *mag, float *dir)
{
    for (int i = 0; i < image_height*image_width; i++)
    {
        if (src1[i] == 0)
            src1[i] = 0.0001;

        mag[i] = sqrt(pow(src1[i], 2) + pow(src2[i], 2));
        dir[i] = atan(src2[i]/src1[i]) * 180/3.142;
        if (dir[i] > 67.5 && dir[i] < 90)
            dir[i] = 90;
        else if (dir[i] > 22.5 && dir[i] <= 67.5)
            dir[i] = 45;
        else if (dir[i] > -22.5 && dir[i] <= 22.5)
            dir[i] = 0;
        else if (dir[i] > -67.5 && dir[i] <= -22.5)
            dir[i] = -45;
        else
            dir[i] = -90;
    }
}


void nonmaxsup(float *mag, float *dir, float *dst)
{
    for (int i = 0; i < image_height*image_width; i++)
    {
        switch((int)dir[i])
        {
        case 90:
        {
            if (mag[i] >= mag[i-image_width] && mag[i] >= mag[i+image_width])
                dst[i] = mag[i];
            else
                dst[i] = 0;
            break;
        }
        case 45:
        {
            if (mag[i] >= mag[i-image_width+1] && mag[i] >= mag[i+image_width-1])
                dst[i] = mag[i];
            else
                dst[i] = 0;
            break;
        }
        case 0:
        {
            if (mag[i] >= mag[i+1] && mag[i] >= mag[i-1])
                dst[i] = mag[i];
            else
                dst[i]= 0;
            break;
        }
        case -45:
        {
            if (mag[i] >= mag[i+image_width+1] && mag[i] >= mag[i-image_width-1])
                dst[i] = mag[i];
            else
                dst[i] = 0;
            break;
        }
        case -90:
        {
            if (mag[i] >= mag[i-image_width] && mag[i] >= mag[i+image_width])
                dst[i] = mag[i];
            else
                dst[i] = 0;
            break;
        }
        }

    }

    float maxval = minmax(dst, 1);
    for (int l = 0; l < image_height*image_width; l++)
        dst[l] = (dst[l]/maxval)*255;
}

void doublethresh(float *nms, int *dt, int thresh1, int thresh2)
{
    for (int i = 0; i < image_height*image_width; i++)
    {
        if (nms[i] > thresh2)
            dt[i] = 255;
        else if (nms[i] < thresh1)
            dt[i] = 0;
        else
            dt[i] = 128;
    }
}



void edgehyst(int *dt, int *dst)
{

    int *visitedmap = (int *)malloc(image_width*image_height*sizeof(int));

    for (int i = 0; i < image_height*image_width; i++)
    {
        if (dt[i] == 0)
        {
            visitedmap[i] = 1;
            dst[i] = dt[i];
        }
        else if (dt[i] == 255)
            dst[i] = dt[i];
    }

    for (int i = 0; i < image_height*image_width; i++)
    {
        if(dt[i] == 255)
            mark(i, visitedmap, dt, dst);
    }

    for (int k = 0; k < image_height*image_width; k++)
            dst[k] = dst[k] * visitedmap[k];
}


void mark(int h, int *visitedmap, int *dt, int *dst)
{
    if (visitedmap[h] == 1)
        return;

    visitedmap[h] = 1;

//1
    if (dt[h-image_width-1] == 128)
    {
        mark(h-image_width-1, visitedmap, dt, dst);
        visitedmap[h-image_width-1] = 1;
        dst[h-image_width-1] = 255;
    }

//2
    if (dt[h-image_width] == 128)
    {
        mark(h-image_width, visitedmap, dt, dst);
        visitedmap[h-image_width] = 1;
        dst[h-image_width] = 255;
    }

//3
    if (dt[h-image_width+1] == 128)
    {
        mark(h-image_width+1, visitedmap, dt, dst);
        visitedmap[h-image_width+1] = 1;
        dst[h-image_width+1] = 255;
    }

//4
    if (dt[h-1] == 128)
    {
        mark(h-1, visitedmap, dt, dst);
        visitedmap[h-1] = 1;
        dst[h-1] = 255;
    }

//5
    if (dt[h+1] == 128)
    {
        mark(h+1, visitedmap, dt, dst);
        visitedmap[h+1] = 1;
        dst[h+1] = 255;
    }

//6
    if (dt[h+image_width-1] == 128)
    {
        mark(h+image_width-1, visitedmap, dt, dst);
        visitedmap[h+image_width-1] = 1;
        dst[h+image_width-1] = 255;
    }

//7
    if (dt[h+image_width] == 128)
    {
        mark(h+image_width, visitedmap, dt, dst);
        visitedmap[h+image_width] = 1;
        dst[h+image_width] = 255;
    }

//8
    if (dt[h+image_width+1] == 128)
    {
        mark(h+image_width+1, visitedmap, dt, dst);
        visitedmap[h+image_width+1] = 1;
        dst[h+image_width+1] = 255;
    }

    return;
}


void borderrep(int *img, int *borderimg)
{

    int j = 0;
    for (int i = 0; i < (bord_image_height)*(bord_image_width); i++)
    {

        if (i%bord_image_width == 0 && i > 0)
            j++;
        if (i >= image_height*bord_image_width)
            borderimg[i] = borderimg[i-bord_image_width];
        else if (i >= image_width && i < bord_image_width)
            borderimg[i] = borderimg[i-1];
        else if (i%(j*bord_image_width+image_width) < bordlen && i >= bord_image_width)
            borderimg[i] = borderimg[i-1];
        else
            borderimg[i] = img[(i%bord_image_width)+(j*image_width)];

    }
}

void deborderrep(float *borderimg, float *deborderimg)
{
    int j = 0;
    for (int i = 0; i < image_height*image_width; i++)
    {
        if (i%image_width == 0 && i > 0)
            j++;
        deborderimg[i] = borderimg[i + j*bordlen];
    }
}
