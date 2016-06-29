// Conversion from 2-D Matrix to 1-D Array allover
// Border Replication and Debordering
// Memory Usage Reduction by using only the pixels required (8 Rows of pixels processed at a time)
// Reduction of time in Edge Hysteresis Stage
// Change float type to int type allover


#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#define SOBELX 1
#define SOBELY 0

int image_height = 0, image_width = 0;
int bordlen = 10;
int bord_image_height = 0, bord_image_width = 0;

int minmax(int *img_array, int option);
void gaussianfilter(int *src, int *dst, int n);
void sobelfilter(int *src, int *dst, int direction, int n);
void imgwrite(char image_name[]);
void imgread(char name[], int *img_data);
void getdim(char name[]);
void matmul(int *src1, int *src2, int *dst);
void getgrad(int *src1, int *src2, int *mag, int *dir, int n);
void nonmaxsup(int *mag, int *dir, int *dst);
void doublethresh(int *nms, int *doublethr, int thresh1, int thresh2);
void edgehyst(int *doublthr, int *cannyfinal);
void mark(int h, int *visitedmap, int *dt, int *dst);
void borderrep(int *img, int *borderimg);
void deborderrep(int *borderimg, int *img, int n);

int main()
{
    char image[] = "doni.jpg";
    imgwrite(image);

    char image_txt[] = "doni.txt";
    getdim(image_txt);
    int img_data[image_height*image_width];
    imgread(image_txt, img_data);

    bord_image_height = image_height + bordlen;
    bord_image_width = image_width + bordlen;

    int *borderimg = (int *)malloc(bord_image_height*bord_image_width*sizeof(int));
    int *gaussian_outputbuf = (int *)malloc(5*bord_image_width*sizeof(int));
    int *sobel_x_outputbuf = (int *)malloc(3*bord_image_width*sizeof(int));
    int *sobel_y_outputbuf = (int *)malloc(3*bord_image_width*sizeof(int));
    int *sobel_xdeb = (int *)malloc(3*image_width*sizeof(int));
    int *sobel_ydeb = (int *)malloc(3*image_width*sizeof(int));
    int *gradmagbuf = (int *)malloc(3*image_width*sizeof(int));
    int *graddirbuf = (int *)malloc(3*image_width*sizeof(int));
    int *nms_image = (int *)malloc(image_height*image_width*sizeof(int));
    int *doublethr = (int *)malloc(image_height*image_width*sizeof(int));
    int *cannyfinal = (int *)malloc(image_height*image_width*sizeof(int));

    // Border Replicate
    borderrep(img_data, borderimg);
    // Convolution with Gaussian Kernel of size 5*5 and sigma = 1.4
    gaussianfilter(borderimg, gaussian_outputbuf, 5);
    // Convolution with Sobel Kernel of size 3*3 in x- and y- directions
    sobelfilter(gaussian_outputbuf, sobel_x_outputbuf, SOBELX, 3);
    sobelfilter(gaussian_outputbuf, sobel_y_outputbuf, SOBELY, 3);
    //Debordering
    deborderrep(sobel_x_outputbuf, sobel_xdeb, 3);
    deborderrep(sobel_y_outputbuf, sobel_ydeb, 3);
    // Getting Gradient Magnitude and Directions
    getgrad(sobel_xdeb, sobel_ydeb, gradmagbuf, graddirbuf, 3);
    // Non Maximum Suppression for Edge Localisation
    nonmaxsup(gradmagbuf, graddirbuf, nms_image);
    nonmaxsup(&gradmagbuf[image_width], &graddirbuf[image_width], &nms_image[image_width]);

    //-------------
    for (int n = 2; n < image_height; n++)
    {
        for (int i = 0; i < 4*bord_image_width; i++)
            gaussian_outputbuf[i] = gaussian_outputbuf[i + bord_image_width];
        gaussianfilter(&borderimg[(n+3)*bord_image_width], &gaussian_outputbuf[4*bord_image_width], 1);

        for (int i = 0; i < 2*bord_image_width; i++)
        {
            sobel_x_outputbuf[i] = sobel_x_outputbuf[i + bord_image_width];
            sobel_y_outputbuf[i] = sobel_y_outputbuf[i + bord_image_width];
        }

        for (int i = 0; i < 2*image_width; i++)
        {
            sobel_xdeb[i] = sobel_xdeb[i + image_width];
            sobel_ydeb[i] = sobel_ydeb[i + image_width];
            gradmagbuf[i] = gradmagbuf[i + image_width];
            graddirbuf[i] = graddirbuf[i + image_width];
        }
        sobelfilter(&gaussian_outputbuf[2*bord_image_width], &sobel_x_outputbuf[2*bord_image_width], SOBELX, 1);
        sobelfilter(&gaussian_outputbuf[2*bord_image_width], &sobel_y_outputbuf[2*bord_image_width], SOBELY, 1);

        deborderrep(&sobel_x_outputbuf[2*bord_image_width], &sobel_xdeb[2*image_width], 1);
        deborderrep(&sobel_y_outputbuf[2*bord_image_width], &sobel_ydeb[2*image_width], 1);

        getgrad(&sobel_xdeb[2*image_width], &sobel_ydeb[2*image_width], &gradmagbuf[2*image_width], &graddirbuf[2*image_width], 1);

        nonmaxsup(&gradmagbuf[image_width], &graddirbuf[image_width], &nms_image[n*image_width]);
    }
    int maxval = minmax(nms_image, 1);
    for (int i = 0; i < image_height*image_width; i++)
        nms_image[i] = (int)((((float)nms_image[i])/maxval)*255);

    // Double Thresholding to classify weak and strong edges
    doublethresh(nms_image, doublethr, 30, 50);
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
    //cvSaveImage("inter.jpg", test, 0);
    return 0;

}

int minmax(int *img_array, int option)
{
    int val = 0;
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


void gaussianfilter(int *src, int *dst, int n)
{

    float gaussian[25] = {0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
                          0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
                          0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
                          0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
                          0.003765, 0.015019, 0.023792, 0.015019, 0.003765
                         };

    int kernel_size = 5;

    for (int l = 0; l < n*bord_image_width; l++)
    {
        if ((l%bord_image_width) >= (bord_image_width-bordlen))
            dst[l] = dst[l-1];
        else
        {
            float sum = 0;
            for (int i = 0; i < kernel_size; i++)
            {
                for (int j = 0; j < kernel_size; j++)
                {
                    sum = sum + src[l + j + bord_image_width*i]*gaussian[j + kernel_size*i];
                }
            }
            dst[l] = (int)sum;
        }
    }
}

void sobelfilter(int *src, int *dst, int direction, int n)
{
    int kernel_size = 3;
    for (int l = 0; l < n*bord_image_width; l++)
    {
        if ((l%bord_image_width) >= (bord_image_width-bordlen))
            dst[l] = dst[l-1];
        else
        {
            int sum = 0;
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

void matmul(int *src1, int *src2, int *dst)
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


void getgrad(int *src1, int *src2, int *mag, int *dir, int n)
{
    for (int i = 0; i < n*image_width; i++)
    {
        if (src1[i] == 0)
            src1[i] = 0.0001;

        mag[i] = (int)sqrt(pow(src1[i], 2) + pow(src2[i], 2));
        float angle = atan(((float)src2[i])/src1[i]) * 180/3.142;
        if (angle > 67.5 && angle < 90)
            dir[i] = 90;
        else if (angle > 22.5 && angle <= 67.5)
            dir[i] = 45;
        else if (angle > -22.5 && angle <= 22.5)
            dir[i] = 0;
        else if (angle > -67.5 && angle <= -22.5)
            dir[i] = -45;
        else
            dir[i] = -90;
    }
}


void nonmaxsup(int *mag, int *dir, int *dst)
{
    for (int i = 0; i < image_width; i++)
    {
        switch((int)dir[i])
        {
        case 90:
        {
            if (mag[i] >= mag[i-image_width] && mag[i] >= mag[i+image_width])
                dst[i] = (int)mag[i];
            else
                dst[i] = 0;
            break;
        }
        case 45:
        {
            if (mag[i] >= mag[i-image_width+1] && mag[i] >= mag[i+image_width-1])
                dst[i] = (int)mag[i];
            else
                dst[i] = 0;
            break;
        }
        case 0:
        {
            if (mag[i] >= mag[i+1] && mag[i] >= mag[i-1])
                dst[i] = (int)mag[i];
            else
                dst[i]= 0;
            break;
        }
        case -45:
        {
            if (mag[i] >= mag[i+image_width+1] && mag[i] >= mag[i-image_width-1])
                dst[i] = (int)mag[i];
            else
                dst[i] = 0;
            break;
        }
        case -90:
        {
            if (mag[i] >= mag[i-image_width] && mag[i] >= mag[i+image_width])
                dst[i] = (int)mag[i];
            else
                dst[i] = 0;
            break;
        }
        }

    }
}

void doublethresh(int *nms, int *dt, int thresh1, int thresh2)
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

void deborderrep(int *borderimg, int *deborderimg, int n)
{
    int j = 0;
    for (int i = 0; i < n*image_width; i++)
    {
        if (i%image_width == 0 && i > 0)
            j++;
        deborderimg[i] = borderimg[i + j*bordlen];
    }
}
