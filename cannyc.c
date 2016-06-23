#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#define SOBELX 1
#define SOBELY 0

int image_height = 0, image_width = 0;

float minmax(float *img_array[], int option);
void gaussianfilter(int src[][image_width], float *dst[]);
void sobelfilter(float *src[], float *dst[], int direction);
void imgwrite(char image_name[]);
void imgread(char name[], int img_data[][image_width]);
void getdim(char name[]);
void matmul(float *src1[], float *src2[], float *dst[]);
void getgrad(float *src1[], float *src2[], float *mag[], float *dir[]);
void nonmaxsup(float *mag[], float *dir[], float *dst[]);
void doublethresh(float *nms[], int *doublethr[], int thresh1, int thresh2);
void edgehyst(int *doublthr[], int *cannyfinal[]);
void mark(int h, int g, int *map[], int *visitedmap[], int *dttemp[]);

int flag = 0;

int main()
{
 char image[] = "lena.jpg";
 imgwrite(image);

 char image_txt[] = "lena.txt";
 getdim(image_txt);
 int img_data[image_height][image_width];
 imgread(image_txt, img_data);

 float *gaussian_image[image_height];
 float *sobel_x[image_height];
 float *sobel_y[image_height];
 float *gradmag[image_height];
 float *graddir[image_height];
 float *nms_image[image_height];
 int *doublethr[image_height];
 int *cannyfinal[image_height];
 for (int i = 0; i < image_height; i++)
    {
     gaussian_image[i] = (float *)malloc(image_width*sizeof(float));
     sobel_x[i] = (float *)malloc(image_width*sizeof(float));
     sobel_y[i] = (float *)malloc(image_width*sizeof(float));
     gradmag[i] = (float *)malloc(image_width*sizeof(float));
     graddir[i] = (float *)malloc(image_width*sizeof(float));
     nms_image[i] = (float *)malloc(image_width*sizeof(float));
     doublethr[i] = (int *)malloc(image_width*sizeof(int));
     cannyfinal[i] = (int *)malloc(image_width*sizeof(int));
    }

 // Convolution with Gqaussian Kernel of size 5*5 and sigma = 1.4
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

 IplImage *input = cvLoadImage("lena.jpg", 0);
 IplImage *canny_opencv = cvCloneImage(input);
 cvCanny(input, canny_opencv, 150, 220, 3);

 //IplImage *test = cvCloneImage(input);
 //uchar *test_data = (uchar *)test->imageData;

 IplImage *test1 = cvCloneImage(input);
 uchar *test1_data = (uchar *)test1->imageData;
 for (int i = 0; i < image_height; i++)
    {
     for (int j = 0; j < image_width; j++)
        {
         //*(test_data + i*image_width + j) = doublethr[i][j];
         *(test1_data + i*image_width + j) = doublethr[i][j];
        }
    }

 cvSaveImage("lenadt.jpg", test1, 0);
 //cvSaveImage("lenacannyopencv.jpg", canny_opencv, 0);

 //cvNamedWindow("Test", CV_WINDOW_NORMAL);
 //cvNamedWindow("Test1", CV_WINDOW_NORMAL);
 cvShowImage("Input", input);
 //cvShowImage("Test", test);
 cvShowImage("Canny C Implementaion", test1);
 cvShowImage("Canny OpenCV", canny_opencv);
 cvWaitKey(0);
 return 0;

 }

 float minmax(float *img_array[], int option)
{
 float val = 0;
 if (option)
 {
 for (int i = 0; i <= image_height-1; i++)
    {
     for(int j = 0; j<= image_width-1; j++)
        {
         val = val > img_array[i][j] ? val : img_array[i][j];
        }
    }
  }
  else
  {
  for (int i = 0; i <= image_height-1; i++)
    {
     for(int j = 0; j<= image_width-1; j++)
        {
         val = val < img_array[i][j] ? val : img_array[i][j];
        }
    }
  }
 return val;
}


void gaussianfilter(int src[][image_width], float *dst[])
{

 float gaussian[5][5] = {{0.003765, 0.015019, 0.023792, 0.015019, 0.003765},
 {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
 {0.023792, 0.094907, 0.150342, 0.094907, 0.023792},
 {0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
 {0.003765, 0.015019, 0.023792, 0.015019, 0.003765}};

 int anchor;

 for (int l = 0; l < image_height; l++)
    {
     for (int m = 0; m < image_width; m++)
        {
         anchor = 5/2;
         if (l < anchor || m < anchor || l >  image_height-1-anchor || m > image_width-1-anchor)
         {
         dst[l][m] = src[l][m];
         }
         else
         {
          float sum = 0;
          for (int g = anchor; g >= -anchor; g--)
            {
             for (int h = anchor; h >= -anchor; h--)
             {
              sum = sum + (src[l-g][m-h]) * (gaussian[anchor-g][anchor-h]);
             }
            }
          dst[l][m] = sum;
         }
        }
    }
}

void sobelfilter(float *src[], float *dst[], int direction)
{
 int anchor;
 for (int l = 0; l < image_height; l++)
    {
     for (int m = 0; m < image_width; m++)
        {
         anchor = 3/2;
         if (l < anchor || m < anchor || l >  image_height-1-anchor || m > image_width-1-anchor)
         {
         dst[l][m] = 0;
         }
         else
         {
          float sum = 0;
          if (direction)
          {
           int sobel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
           //int sobel[5][5] = {1, 2, 0, -2, -1, 4, 8, 0, -8, -4, 6, 12, 0, -12, -6, 4, 8, 0, -8, -4, 1, 2, 0, -2, -1};
           for (int g = anchor; g >= -anchor; g--)
             {
              for (int h = anchor; h >= -anchor; h--)
              {
               sum = sum + src[l-g][m-h] * sobel[anchor-g][anchor-h];
              }
             }
           dst[l][m] = sum;
          }
          else
          {
           int sobel[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
           //int sobel[5][5] = {-1, -4, -6, -4, -1, -2, -8, -12, -8, -2, 0, 0, 0, 0, 0, 2, 8, 12, 8, 2, 1, 4, 6, 4, 1};
           for (int g = anchor; g >= -anchor; g--)
             {
              for (int h = anchor; h >= -anchor; h--)
              {
               sum = sum + src[l-g][m-h] * sobel[anchor-g][anchor-h];
              }
             }
           dst[l][m] = sum;
          }
         }
        }
    }
}

void matmul(float *src1[], float *src2[], float *dst[])
{
 for (int i = 0; i <= image_height-1; i++)
    {
     for (int j = 0; j<= image_width-1; j++)
        {
         dst[i][j] = src1[i][j] * src2[i][j];
        }
    }
}


void imgwrite(char image_name[])
{
 IplImage *input_image;
 char image_text[20];
 strcpy(image_text, image_name);
 int m = 0;
 while(image_text[m] != '.') {m++;}
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


void imgread(char image_text[], int img_data[][image_width])
{
 FILE *temp;
 temp = fopen(image_text, "r");

 fscanf(temp, "%d", &image_height);
 fscanf(temp, "%d", &image_width);

 int j = 0, k = 0;
 while(!((j+1)*k == image_width*image_height))
 {
  if(k == image_width)
   {j++;k = 0;}

  fscanf(temp, "%d", &img_data[j][k]);
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


void getgrad(float *src1[], float *src2[], float *mag[], float *dir[])
{
 for (int i = 0; i < image_height; i++)
    {
     for (int j = 0; j < image_width; j++)
        {
         if (src1[i][j] == 0)
            src1[i][j] = 0.0001;

         mag[i][j] = sqrt(pow(src1[i][j], 2) + pow(src2[i][j], 2));
         dir[i][j] = atan(src2[i][j]/src1[i][j]) * 180/3.142;
         if (dir[i][j] > 67.5 && dir[i][j] < 90)
            dir[i][j] = 90;
         else if (dir[i][j] > 22.5 && dir[i][j] <= 67.5)
            dir[i][j] = 45;
         else if (dir[i][j] > -22.5 && dir[i][j] <= 22.5)
            dir[i][j] = 0;
         else if (dir[i][j] > -67.5 && dir[i][j] <= -22.5)
            dir[i][j] = -45;
         else
            dir[i][j] = -90;
        }
    }
}


void nonmaxsup(float *mag[], float *dir[], float *dst[])
{
 for (int i = 0; i <= image_height-2; i++)
    {
     for (int j = 0; j <= image_width-2; j++)
        {
         switch((int)dir[i][j])
         {
         case 90:
            {
             if (mag[i][j] >= mag[i-1][j] && mag[i][j] >= mag[i+1][j])
                dst[i][j] = mag[i][j];
             else
                dst[i][j] = 0;
             break;
            }
          case 45:
            {
             if (mag[i][j] >= mag[i-1][j+1] && mag[i][j] >= mag[i+1][j-1])
                dst[i][j] = mag[i][j];
             else
                dst[i][j] = 0;
             break;
            }
          case 0:
            {
             if (mag[i][j] >= mag[i][j+1] && mag[i][j] >= mag[i][j-1])
                dst[i][j] = mag[i][j];
             else
                dst[i][j] = 0;
             break;
            }
          case -45:
            {
             if (mag[i][j] >= mag[i-1][j-1] && mag[i][j] >= mag[i+1][j+1])
                dst[i][j] = mag[i][j];
             else
                dst[i][j] = 0;
             break;
            }
          case -90:
            {
             if (mag[i][j] >= mag[i-1][j] && mag[i][j] >= mag[i+1][j])
                dst[i][j] = mag[i][j];
             else
                dst[i][j] = 0;
             break;
            }
         }

        }
    }

  float maxval = minmax(dst, 1);
  for (int l = 0; l < image_height; l++)
    {
     for (int m = 0; m < image_width; m++)
        {
         dst[l][m] = (dst[l][m]/maxval)*255;
        }
    }
}

void doublethresh(float *nms[], int *dt[], int thresh1, int thresh2)
{
 for (int i = 0; i < image_height; i++)
    {
     for (int j = 0; j < image_width; j++)
        {
          if (nms[i][j] > thresh2)
            dt[i][j] = 255;
          else if (nms[i][j] < thresh1)
            dt[i][j] = 0;
          else
            dt[i][j] = 128;
        }
    }
}



void edgehyst(int *dt[], int *dst[])
{

 int *visitedmap[image_height];
 int *map[image_height];
 int *dttemp[image_height];
 for (int i = 0; i < image_height; i++)
    {
     visitedmap[i] = (int *)malloc(image_width * sizeof(int));
     map[i] = (int *)malloc(image_width * sizeof(int));
     dttemp[i] = (int *)malloc(image_width * sizeof(int));
    }

 for (int i = 0; i < image_height; i++)
    {
     for (int j = 0; j < image_width; j++)
        {
         dttemp[i][j] = dt[i][j];
         if (dt[i][j] == 255 || dt[i][j] == 0)
            {
             visitedmap[i][j] = 1;
             dst[i][j] = dt[i][j];
            }

        }
    }

 for (int i = 1; i < image_height-1; i++)
    {
     for (int j = 1; j < image_width-1; j++)
        {
         if(dttemp[i][j] == 128)
            mark(i, j, map, visitedmap, dttemp);


         for (int g = 0; g < image_height; g++)
            {
             for (int h = 0; h < image_width; h++)
                {
                 if (flag)
                  {
                   if (map[g][h] == 200)
                     {
                      dst[g][h] = 255;
                      map[g][h] = 0;
                     }
                  }
                 else if (!flag)
                  {
                   if (map[g][h] == 200)
                     {
                      dst[g][h] = 0;
                      map[g][h] = 0;
                     }
                  }
                }
            }
          flag = 0;

        }
    }
}


void mark(int h, int g, int *map[], int *visitedmap[], int *dttemp[])
{
 if (visitedmap[h][g] == 1)
    return;


 if (dttemp[h][g] == 0)
    return;

 map[h][g] = 200;
 visitedmap[h][g] = 1;

 //1
 if (dttemp[h-1][g-1] == 128)
    {
     mark(h-1, g-1, map, visitedmap, dttemp);
     visitedmap[h-1][g-1] = 1;
     map[h-1][g-1] = 200;
    }
 else if (dttemp[h-1][g-1] == 255)
    {
     flag = 1;
    }

 //2
 if (dttemp[h-1][g] == 128)
    {
     mark(h-1, g, map, visitedmap, dttemp);
     visitedmap[h-1][g] = 1;
     map[h-1][g] = 200;
    }
 else if (dttemp[h-1][g] == 255)
    {
     flag = 1;
    }

 //3
 if (dttemp[h-1][g+1] == 128)
    {
     mark(h-1, g+1, map, visitedmap, dttemp);
     visitedmap[h-1][g+1] = 1;
     map[h-1][g+1] = 200;
    }
 else if (dttemp[h-1][g+1] == 255)
    {
     flag = 1;
    }

 //4
 if (dttemp[h][g-1] == 128)
    {
     mark(h, g-1, map, visitedmap, dttemp);
     visitedmap[h][g-1] = 1;
     map[h][g-1] = 200;
    }
 else if (dttemp[h][g-1] == 255)
    {
     flag = 1;
    }

 //5
 if (dttemp[h][g+1] == 128)
    {
     mark(h, g+1, map, visitedmap, dttemp);
     visitedmap[h][g+1] = 1;
     map[h][g+1] = 200;
    }
 else if (dttemp[h][g+1] == 255)
    {
     flag = 1;
    }

 //6
 if (dttemp[h+1][g-1] == 128)
    {
     mark(h+1, g-1, map, visitedmap, dttemp);
     visitedmap[h+1][g-1] = 1;
     map[h+1][g-1] = 200;
    }
 else if (dttemp[h+1][g-1] == 255)
    {
     flag = 1;
    }

 //7
 if (dttemp[h+1][g] == 128)
    {
     mark(h+1, g, map, visitedmap, dttemp);
     visitedmap[h+1][g] = 1;
     map[h+1][g] = 200;
    }
 else if (dttemp[h+1][g] == 255)
    {
     flag = 1;
    }

 //8
 if (dttemp[h+1][g+1] == 128)
    {
     mark(h+1, g+1, map, visitedmap, dttemp);
     visitedmap[h+1][g+1] = 1;
     map[h+1][g+1] = 200;
    }
 else if (dttemp[h+1][g+1] == 255)
    {
     flag = 1;
    }

 return;
}
