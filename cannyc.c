#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#define SOBELX 1
#define SOBELY 0

int image_height = 0, image_width = 0;

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
void mark(int h, int *map, int *visitedmap, int *dt);

int flag = 0;

int main()
{
//char image[] = "lena.jpg";
//imgwrite(image);

    char image_txt[] = "lena.txt";
    getdim(image_txt);
    int img_data[image_height*image_width];
    imgread(image_txt, img_data);

    float *gaussian_image = (float *)malloc(image_width*image_height*sizeof(float));
    float *sobel_x = (float *)malloc(image_width*image_height*sizeof(float));
    float *sobel_y = (float *)malloc(image_width*image_height*sizeof(float));
    float *gradmag = (float *)malloc(image_width*image_height*sizeof(float));
    float *graddir = (float *)malloc(image_width*image_height*sizeof(float));
    float *nms_image = (float *)malloc(image_width*image_height*sizeof(float));
    int *doublethr = (int *)malloc(image_width*image_height*sizeof(int));
    int *cannyfinal = (int *)malloc(image_width*image_height*sizeof(int));

    // Convolution with Gaussian Kernel of size 5*5 and sigma = 1.4
    gaussianfilter(img_data, gaussian_image);
    // Convolution with Sobel Kernel of size 3*3 in x- and y- directions
    sobelfilter(gaussian_image, sobel_x, SOBELX);
    sobelfilter(gaussian_image, sobel_y, SOBELY);
    // Getting Gradient Magnitude and Directions
    getgrad(sobel_x, sobel_y, gradmag, graddir);
    // Non Maximum Suppression for Edge Localisation
    nonmaxsup(gradmag, graddir, nms_image);
    // Double Thresholding to classify weak and strong edges
    doublethresh(nms_image, doublethr, 30, 70);
    // Edge Tracking Algorithm
    edgehyst(doublethr, cannyfinal);

    for (int i = image_width*(image_height-4); i < image_height*image_width; i++)
        cannyfinal[i] = 0;

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

    for (int l = 0; l < image_height*image_width; l++)
    {
        if (l > image_width*(image_height-(kernel_size-1))-1 || (l%image_width) == (image_width-1) ||
                (l%image_width) == (image_width-2) || (l%image_width) == (image_width-3) || (l%image_width) == (image_width-4))
            dst[l] = src[l];
        else
        {
            int sum = 0;
            for (int i = 0; i < kernel_size; i++)
            {
                for (int j = 0; j < kernel_size; j++)
                {
                    sum = sum + src[l + j + image_width*i]*gaussian[j + kernel_size*i];
                }
            }
            dst[l] = sum;
        }
    }

}

void sobelfilter(float *src, float *dst, int direction)
{
    int kernel_size = 3;
    for (int l = 0; l < image_height*image_width; l++)
    {
        if (l > image_width*(image_height-(kernel_size-1))-1 || (l%image_width) == (image_width-1) ||
                (l%image_width) == (image_width-2))
            dst[l] = src[l];
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
                        sum = sum + src[l + j + image_width*i]*sobel[j + kernel_size*i];
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
                        sum = sum + src[l + j + image_width*i]*sobel[j + kernel_size*i];
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
    int *map = (int *)malloc(image_width*image_height*sizeof(int));

    for (int i = 0; i < image_height*image_width; i++)
    {
        if (dt[i] == 255 || dt[i] == 0)
        {
            visitedmap[i] = 1;
            dst[i] = dt[i];
        }
    }

    for (int i = 0; i < image_height*image_width; i++)
    {
        if(dt[i] == 128)
            mark(i, map, visitedmap, dt);


        for (int g = 0; g < image_height*image_width; g++)
        {
            if (flag)
            {
                if (map[g] == 200)
                {
                    dst[g] = 255;
                    map[g] = 0;
                }
            }
            else if (!flag)
            {
                if (map[g] == 200)
                {
                    dst[g] = 0;
                    map[g] = 0;
                }
            }

        }
        flag = 0;
    }
}


void mark(int h, int *map, int *visitedmap, int *dt)
{
    if (visitedmap[h] == 1)
        return;

    map[h] = 200;
    visitedmap[h] = 1;

//1
    if (dt[h-image_width-1] == 128)
    {
        mark(h-image_width-1, map, visitedmap, dt);
        visitedmap[h-image_width-1] = 1;
        map[h-image_width-1] = 200;
    }
    else if (dt[h-image_width-1] == 255)
    {
        flag = 1;
    }

//2
    if (dt[h-image_width] == 128)
    {
        mark(h-image_width, map, visitedmap, dt);
        visitedmap[h-image_width] = 1;
        map[h-image_width] = 200;
    }
    else if (dt[h-image_width] == 255)
    {
        flag = 1;
    }

//3
    if (dt[h-image_width+1] == 128)
    {
        mark(h-image_width+1, map, visitedmap, dt);
        visitedmap[h-image_width+1] = 1;
        map[h-image_width+1] = 200;
    }
    else if (dt[h-image_width+1] == 255)
    {
        flag = 1;
    }

//4
    if (dt[h-1] == 128)
    {
        mark(h-1, map, visitedmap, dt);
        visitedmap[h-1] = 1;
        map[h-1] = 200;
    }
    else if (dt[h-1] == 255)
    {
        flag = 1;
    }

//5
    if (dt[h+1] == 128)
    {
        mark(h+1, map, visitedmap, dt);
        visitedmap[h+1] = 1;
        map[h+1] = 200;
    }
    else if (dt[h+1] == 255)
    {
        flag = 1;
    }

//6
    if (dt[h+image_width-1] == 128)
    {
        mark(h+image_width-1, map, visitedmap, dt);
        visitedmap[h+image_width-1] = 1;
        map[h+image_width-1] = 200;
    }
    else if (dt[h+image_width-1] == 255)
    {
        flag = 1;
    }

//7
    if (dt[h+image_width] == 128)
    {
        mark(h+image_width, map, visitedmap, dt);
        visitedmap[h+image_width] = 1;
        map[h+image_width] = 200;
    }
    else if (dt[h+image_width] == 255)
    {
        flag = 1;
    }

//8
    if (dt[h+image_width+1] == 128)
    {
        mark(h+image_width+1, map, visitedmap, dt);
        visitedmap[h+image_width+1] = 1;
        map[h+image_width+1] = 200;
    }
    else if (dt[h+image_width+1] == 255)
    {
        flag = 1;
    }

    return;
}
