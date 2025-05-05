#include <stdio.h>
using namespace std;

#include <cmath>
#include <iostream>

// OpenCV includes
#include <opencv2/opencv.hpp>

//#include "opencv2/core.hpp"
//#include "opencv2/highgui.hpp"
using namespace cv;

//utworzenie okna o nazwie name w punkcie (x,y) (lewy górny róg okna)
void CreateWindowAt(const char* name, int x, int y)
{
    namedWindow(name, WINDOW_AUTOSIZE);
    moveWindow(name, x, y);
}

// wyświetlenie obrazu img w oknie o nazwie name położonego w punkcie (x,y) (lewy górny róg okna)
void ShowImageAt(const char* name, Mat img, int x, int y)
{
    CreateWindowAt(name, x, y);
    imshow(name, img);
}

// wczytanie obrazu z pliku name do macierzy img
int read_image(const char* name, Mat* img)
{
    *img = imread(name);
    if (!(*img).data)
    {
        cout << "Error! Cannot read source file. Press ENTER.";
        waitKey(); // czekaj na naciśnięcie klawisza
        return(-1);
    }
}

Mat srcImage;            // obraz wejściowy
Mat greyImage;           // obraz po konwersji do obrazu w odcieniach szarości

//funkcja konwertująca obraz src na obraz dst w odcieniach szarości
//void convertToGrey(Mat src, Mat dst)
//{
//    //pętla po wszystkich pikselach obrazu
//    for (int x = 0; x < src.cols; x++)
//        for (int y = 0; y < src.rows; y++)
//        {
//            //pobranie do zmiennej pixelColor wszystkich 3 składowych koloru piksela
//            Vec3b pixelColor = src.at<Vec3b>(y, x);
//            //konwersja na kolor szary; pixelColor[0] składowa B, pixelColor[1] składowa G, pixelColor[2] składowa R
//            int gray = (int)(0.299f * pixelColor[2] + 0.587f * pixelColor[1] + 0.114f * pixelColor[0]);
//            for (int i = 0; i < 3; i++) // for BGR elements
//                pixelColor[i] = gray;
//            //ustawienie obliczonej wartości piksela na obrazie wyjściowym
//            dst.at<Vec3b>(y, x) = pixelColor;
//        }
//}

// nowa funkcja, lepsza konwersja do 1-kanałowego obrazu (4. histogram)
void convertToGrey(Mat src, Mat& dst)
{
    cvtColor(src, dst, COLOR_BGR2GRAY);
}

// funkcja zmieniająca kontrast i jasność obrazu src i umieszczająca wynik na obrazie dst
// 1. operacje liniowe
void BrightnessAndContrast(Mat src, Mat dst, float A, int B)
{
    //pętla po wszystkich pikselach obrazu
    for (int x = 0; x < src.cols; x++)
        for (int y = 0; y < src.rows; y++)
        {
            Vec3b pixelColor = src.at<Vec3b>(y, x);
            for (int i = 0; i < 3; i++) // for BGR elements
                pixelColor[i] = std::min(255, std::max(0, (int)(A * (pixelColor[i] - 128) + 128 + B)));        //należy umieścić tutaj odpowiedni kod

            dst.at<Vec3b>(y, x) = pixelColor;
        }
}

// wartość jasności (B)
int brightness_value = 100;
//wartość kontrastu (A)
int alpha_value = 200;

// funkcja związana z suwakiem, wywoływana przy zmianie jego położenia
void BrightnessAndContrastCallBack(int pos, void* userdata)
{
    Mat* img = (Mat*)userdata;
    //wywołanie funkcji realizującej zmianę jasności i kontrastu BrightnessAndContrast
    BrightnessAndContrast(srcImage, *img, alpha_value / 100.0f, brightness_value - 200);
    imshow("Bright Image", *img);
}

// 2. potegowanie
float alpha_pow = 1.0f;  // wartość alfa dla potęgowania

void PowerLUT(Mat src, Mat& dst, float alpha)
{
    int Jmax = 255; // maksymalna wartość piksela
    Mat LUT(1, 256, CV_8U); // tablica LUT

    for (int i = 0; i < 256; i++)
    {
        float normalized = (float)i / Jmax;
        int newValue = (int)(Jmax * pow(normalized, alpha));
        LUT.at<uchar>(i) = std::min(255, std::max(0, newValue));
    }

    cv::LUT(src, LUT, dst);
}

// Funkcja callback dla suwaka alpha
void PowerCallBack(int pos, void* userdata)
{
    Mat* img = (Mat*)userdata;

    alpha_pow = 0.1f + (pos / 300.0f) * (3.0f - 0.1f);

    Mat poweredImage;
    PowerLUT(srcImage, poweredImage, alpha_pow); 
    imshow("Potegowanie", poweredImage);
}

// 3.1 maskowanie
void applyMask(Mat src, Mat mask, Mat& dst) {
    Mat floatMask;
    mask.convertTo(floatMask, CV_32F, 1.0 / 255.0);

    Mat srcFloat;
    src.convertTo(srcFloat, CV_32FC3);

    std::vector<Mat> channels;
    split(srcFloat, channels);

    for (int i = 0; i < 3; ++i) {
        channels[i] = channels[i].mul(floatMask);
    }

    merge(channels, dst);
    dst.convertTo(dst, CV_8UC3);
}

// 3.2 mieszanie
void blendImages(Mat img1, Mat img2, Mat& dst, float alpha) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        cout << "images are not complatible for blending" << endl;
        return;
    }
    addWeighted(img1, alpha, img2, 1.0f - alpha, 0.0, dst);
}

int blend_alpha_value = 25; // start value (0-100)

void BlendCallback(int pos, void* userdata) {
    std::pair<Mat*, Mat*>* imgs = (std::pair<Mat*, Mat*>*)userdata;
    float alpha = blend_alpha_value / 100.0f;

    Mat blended;
    blendImages(*imgs->first, *imgs->second, blended, alpha);
    imshow("Blended Image", blended);
}

// 4. histogram
void CountHistogram(Mat src, float histogram[256]) {
    for (int i = 0; i < 256; i++)
        histogram[i] = 0;
    for (int y = 0; y < src.rows; y++) {
        for(int x = 0; x < src.cols; x++) {
            uchar intensity = src.at<uchar>(y, x);
            histogram[intensity]++;
        }
    }
}

void EqualizeHistogram(Mat src, Mat& dst) {
    float histogram[256];
    CountHistogram(src, histogram);

    int totalPixels = src.rows * src.cols;

    float cdf[256] = { 0 };
    cdf[0] = histogram[0];

    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    uchar LUT[256];
    for (int i = 0; i < 256; i++) {
        LUT[i] = (uchar)(255.0f * (cdf[i] / totalPixels));
    }

    dst = src.clone();

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            uchar pixel = src.at<uchar>(y, x);
            dst.at<uchar>(y, x) = LUT[pixel];
        }
    }
}

int main()
{
    // wczytanie obrazu do srcImage
    int r = read_image("Samples/zebra.jpg", &srcImage);
    if (r == -1) return(-1);
    ShowImageAt("Source image", srcImage, 0, 0);

    // Grey image
    Mat greyImage;
    srcImage.copyTo(greyImage);
    convertToGrey(srcImage, greyImage);
    ShowImageAt("Grey image", greyImage, 450, 0);

    // 1. zmiana jasności i kontrastu
    Mat brightImage;
    //zainicjowanie obrazu brightImage obrazem srcImage
    srcImage.copyTo(brightImage);

    CreateWindowAt("Bright Image", 0, 335);
    //tworzenie suwaka o nazwie "Contrast" związanego z oknem "Bright Image", alpha_value wartość całkowita, która będzie zmieniana przez trackbar, 700 maksymalna wartość na suwaku,  
    // BrightnessAndContrastCallBack funkcja wywoływana przy każdej zmianie pozycji suwaka, brightImage dane przekazywane do callback
    createTrackbar("Contrast", "Bright Image", &alpha_value, 700, BrightnessAndContrastCallBack, &brightImage);
    createTrackbar("Brightness", "Bright Image", &brightness_value, 700, BrightnessAndContrastCallBack, &brightImage);
    //analogicznie dodaj TrackBar dla jasności (brightness) 
    BrightnessAndContrastCallBack(0, &brightImage);

    // 2. potegowanie
    CreateWindowAt("Potegowanie", 450, 335);
    createTrackbar("Alpha", "Potegowanie", &alpha_value, 300, PowerCallBack, &srcImage);

    // 3.1 maskowanie
    Mat maskImage = imread("Samples/maska.jpg", IMREAD_GRAYSCALE);
    if (!maskImage.data) {
        cout << "cannot read mask file. press ENTER";
        waitKey();
        return -1;
    }
    resize(maskImage, maskImage, srcImage.size());
    Mat maskedImage;
    applyMask(srcImage, maskImage, maskedImage);
    ShowImageAt("Masked Image", maskedImage, 900, 0);

    // 3.2 mieszanie
    CreateWindowAt("Blended Image", 900, 335);
    std::pair<Mat*, Mat* > imagesToBlend(&srcImage, &maskedImage);
    createTrackbar("Blend", "Blended Image", &blend_alpha_value, 100, BlendCallback, &imagesToBlend);
    BlendCallback(0, &imagesToBlend);

    // 4. histogram
    Mat greyImageEqualized;
    EqualizeHistogram(greyImage, greyImageEqualized);
    ShowImageAt("Equalized Grey Image", greyImageEqualized, 1350, 0);

    waitKey();
}