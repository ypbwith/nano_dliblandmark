
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <iostream>
#include <math.h>
#include <stddef.h>


using namespace cv;
using namespace std;


// Specifying minimum and maximum size parameters
#define MIN_FACE_SIZE 0
#define MAX_FACE_SIZE 400
#define SKIPFRAME 5
#define RATIO 1
#define AlarmLevel 0.2
#define AlarmCount 30
int alarmCount = 0;
double eyesClosedLevel;

struct leftEyePoint
{
    double x[6];
    double y[6];
};

struct rightEyePoint
{
    double x[6];
    double y[6];
};

struct eyeKeyPoint
{
    leftEyePoint leftEye;

    rightEyePoint rightEye;

};

eyeKeyPoint eyePoint;


void draw_polyline(cv::Mat &img, const dlib::full_object_detection& d, const int start, const int end, bool isClosed = false)
{
    std::vector <cv::Point> points;
    for (int i = start; i <= end; ++i)
    {
        points.push_back(cv::Point(d.part(i).x(), d.part(i).y()));
    }

    cv::polylines(img, points, isClosed, cv::Scalar(255, 0, 0), 1, 16);

}

void render_face(cv::Mat &img, const dlib::full_object_detection& d)
{
    DLIB_CASSERT
    (
        d.num_parts() == 68,
        "\n\t Invalid inputs were given to this function. "
        << "\n\t d.num_parts():  " << d.num_parts()
    );

    draw_polyline(img, d, 0, 16);           // Jaw line
    draw_polyline(img, d, 17, 21);          // Left eyebrow
    draw_polyline(img, d, 22, 26);          // Right eyebrow
    draw_polyline(img, d, 27, 30);          // Nose bridge
    draw_polyline(img, d, 30, 35, true);    // Lower nose
    draw_polyline(img, d, 36, 41, true);    // Left eye
    draw_polyline(img, d, 42, 47, true);    // Right Eye
    draw_polyline(img, d, 48, 59, true);    // Outer lip
    draw_polyline(img, d, 60, 67, true);    // Inner lip

}

void draw_Point(cv::Mat &img, const dlib::full_object_detection& d)
{
    std::vector <cv::Point> points;
    for (int i = 0 ;i <= 4; ++i)
    {
        //points.push_back(cv::Point(d.part(i).x(), d.part(i).y()));
        cv::circle(img, cv::Point(d.part(i).x(), d.part(i).y()), 4, cv::Scalar(255, 0, 0), 1, 16);
    }



}

int main(int argc, char **argv)
{
    std::vector<cv::Point2f> last_object;
    for (int i = 0; i < 68; ++i) {
        last_object.push_back(cv::Point2f(0.0, 0.0));
    }

    try
    {
        //---------------------------------shape pridector -----------------------------
        dlib::shape_predictor pose_model;
        dlib::deserialize("shape_predictor_5_face_landmarks.dat") >> pose_model;

        dlib::rectangle facePostion(0, 0, 0, 0);
        //-------------------------------------face detector-----------------------------
        // Load the Cascade Classifier Xml file
        CascadeClassifier faceCascade("cascade/haarcascade_lefteye_2splits.xml");

        // Create a VideoCapture object
        VideoCapture cap(0);

        // Check if camera opened successfully
        if (!cap.open(0)) return 0;

        Mat frame, frameBig, frameGray;
        //-----------------------------------face detector end------------------------------

        //----------------------------------------tracker------------------------------------
        // List of tracker types in OpenCV 3.2
        // NOTE : GOTURN implementation is buggy and does not work.
        const char *types[] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN" };
        std::vector <string> trackerTypes(types, std::end(types));

        // Create a tracker
        string trackerType = trackerTypes[2];
        Ptr<Tracker> tracker = Tracker::create(trackerType);

        // Define initial boundibg box
        Rect2d bbox(10, 10,10, 10);
         Rect2d facebox(10, 10,10, 10);

        int contframe = 0;
        //----------------------------tracker end------------------------------------------
        unsigned long im_mum=0;

        Mat grayscale_image, img_adaptive,img_EqualizeHist;

        // Creating vector to store the detected faces' parameters
        vector<Rect> faces;

        double eyesClosedLevel_filter[100];

        bool detectflag = 0;
         bool ok = 0  ;

        while (1)
        {
            // Reading each frame
            bool frameRead = cap.read(frameBig);

            // If frame not opened successfully
            if (!frameRead)
                break;

            // Fixing the scaling factor
            //float scale = 1024.0f / frameBig.cols;

            // Resizing the image
            //resize(frameBig, frame, Size(), scale, scale);
            cv::resize(frameBig, frame, cv::Size(), 1.0 / RATIO, 1.0 / RATIO);


            //cvtColor(frame, grayscale_image, CV_BGR2GRAY);

            //equalizeHist(grayscale_image,img_EqualizeHist);

            //adaptiveThreshold(img_EqualizeHist, img_adaptive, 255, adaptive_method, threadshould_type, block_size, offset);
            //imshow("facefilter", img_EqualizeHist);

            //dlib::cv_image<dlib::bgr_pixel> cimg(img_EqualizeHist);
            dlib::cv_image<dlib::bgr_pixel> cimg(frame);
            dlib::cv_image<dlib::bgr_pixel> cframeBig(frameBig);

            // Detect faces
            faceCascade.detectMultiScale(frame, faces, 1.1, 5, 0, Size(MIN_FACE_SIZE, MIN_FACE_SIZE), Size(MAX_FACE_SIZE, MAX_FACE_SIZE));

            contframe++;
            if(contframe > SKIPFRAME)
                contframe =0;

            //printf("contframe is :%d\n",contframe);

            if(contframe == 0)
            {
                // Loop over each detected face
                for (int i = 0; i < faces.size(); i++)
                {
                    // Dimension parameters for bounding rectangle for face
                    Rect faceRect = faces[i];
                    //rectangle(frame, faceRect, Scalar(255, 0, 0), 2, 1);

                    bbox = faceRect;
//                    // Calculating the dimension parameters for eyes from the dimensions parameters of the face
//                    Rect eyesRect = Rect(faceRect.x + 0.125*faceRect.width, faceRect.y + 0.25 * faceRect.height, 0.75 * faceRect.width,
//                        0.25 * faceRect.height);

//                    // Drawing the bounding rectangle around the face
//                    rectangle(frame, eyesRect, Scalar(128, 255, 0), 2);

                     tracker = Tracker::create(trackerType);
                     tracker->init(frame, bbox);
                     bool ok = tracker->update(frame, bbox);

                     detectflag = 1;
                     ok = 1;
                }
            }


            // Start timer
            //double timer = (double)getTickCount();
            ok = tracker->update(frame, bbox);

            // Calculating the dimension parameters for eyes from the dimensions parameters of the face
            //Rect eyesRect = Rect( bbox.x + 0.125* bbox.width,  bbox.y + 0.25 *  bbox.height, 0.75 * bbox.width,
             //   0.25 *  bbox.height);

            // Drawing the bounding rectangle around the face
            //rectangle(frame, eyesRect, Scalar(255, 255, 0), 2);

            // Calculate Frames per second (FPS)
            //float fps = getTickFrequency() / ((double)getTickCount() - timer);

              cv::Rect2d eyebox(int(10),int(10),int(240 ),int(240));
            cv::Mat eye_left(frameBig,eyebox  );
            cv::Mat eye_left_24x24;

            resize(eye_left, eye_left_24x24, cv::Size(24, 24), 0, 0, CV_INTER_LINEAR);

            cv::cvtColor(eye_left_24x24,eye_left_24x24,CV_BGR2GRAY);

            char im_str[sizeof("noeye/imc%d.jpg           ")];
            sprintf(im_str, "noeye/imc%d.jpg", im_mum++);
            cv::imwrite(im_str,eye_left_24x24);


            if (ok)
            {
                // Tracking success : Draw the tracked object
                //rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
                facePostion.set_left(bbox.x);
                facePostion.set_top(bbox.y);
                facePostion.set_right(bbox.x + bbox.width);
                facePostion.set_bottom(bbox.y + bbox.height);

                dlib::rectangle r(
                                   (long)(facePostion.left() * RATIO),
                                   (long)(facePostion.top() * RATIO),
                                   (long)(facePostion.right() * RATIO),
                                   (long)(facePostion.bottom() * RATIO)
                               );

               //Landmark detection on full sized image
               std::vector<dlib::full_object_detection> shapes;
               dlib::full_object_detection shape = pose_model(cframeBig, r);
                shapes.push_back(shape);
                const dlib::full_object_detection & d = shapes[0];
                //cout<<d.part(0).x()<<endl;
                draw_Point(frameBig, shape);

                //render_face(frameBig, shape);
                facebox.y = r.top();
                facebox.x = r.left();
                facebox.width = r.right() - facebox.x;
                facebox.height = r.bottom()  - facebox.y;
                cv::rectangle(frameBig, facebox, Scalar(255, 0, 0), 2, 1);

               //const dlib::full_object_detection & d = shapes[0];


 /*
               for (int i = 0; i < d.num_parts(); i++)
               {
                   last_object[i].x = d.part(i).x();
                   last_object[i].y = d.part(i).y();
               }

               Rect2d eyebox ;
               eyebox.width = abs (d.part(36).x() - d.part(39).x() );
               eyebox.height = eyebox.width;
               eyebox.x = d.part(36).x();
               eyebox.y = abs(d.part(36).y() - 0.5 * eyebox.width);

               //cout << eyebox<<endl;

               Mat img_clone =frameBig.clone();
               Rect2d  dstbox = eyesRect;

               bool dstbox_flag = 0;

               if(dstbox.x + dstbox.width<= 0)
               {
                   dstbox_flag =1;
               }
               else if (dstbox.x<=0)
               {

                   dstbox.width = dstbox.width + dstbox.x;

                   dstbox.x=0;
               }

               if(dstbox.y + dstbox.height<= 0)
               {
                   dstbox_flag =1;

               }
               else if (dstbox.y<=0)
               {
                   dstbox.height = dstbox.height + dstbox.y;
                   dstbox.y=0;
               }

               if(dstbox.x>img_clone.size().width)
               {
                  dstbox_flag =1;
               }
               else if( dstbox.x + dstbox.width>img_clone.size().width)
               {
                   dstbox.width = img_clone.size().width - dstbox.x;
               }

               if(dstbox.y + dstbox.height >img_clone.size().height)
               {
                   dstbox_flag =1;
               }
               else if( dstbox.y + dstbox.height >img_clone.size().height)
               {
                    dstbox.height = img_clone.size().width - dstbox.y;
               }
               if(dstbox_flag == 0)
               {
                   cv::Mat eye_left (img_clone,dstbox );
                   cv::Mat eye_left_24x24;

                   dstbox_flag == 0;

                   //resize(eye_left,eye_left_24x24,Size(64,64),0,0,CV_INTER_LINEAR);
                   char im_str[sizeof("eye_close/im%06d.jpg")];
                   sprintf(im_str,"eye_close/im%06d.jpg", im_mum);
                   //imwrite(im_str,eye_left_24x24); //c版本中的保存图片为cvSaveImage()函数，c++版本中直接与matlab的相似，imwrite()函数。
                   //imshow( "face", eye_left );
               }



              for (int i = 36; i <= 41; i++)
              {

                  eyePoint.leftEye.x[i - 36] = 0.01*shape.part(i).x() + 0.99*last_object[i].x;//predict_points[i].x;
                  eyePoint.leftEye.y[i - 36] = 0.01*shape.part(i).y() + 0.99*last_object[i].y;//predict_points[i].y;

                  //cv::circle(facefilter, cv::Point(eyePoint.leftEye.x[i - 36], eyePoint.leftEye.y[i - 36]), 2, cv::Scalar(255, 0, 0), -1);
              }

              for (int i = 42; i <= 47; i++)
              {

                  eyePoint.rightEye.x[i - 42] = 0.01*shape.part(i).x() + 0.99*last_object[i].x;//predict_points[i].x;
                  eyePoint.rightEye.y[i - 42] = 0.01*shape.part(i).y() + 0.99*last_object[i].y;//predict_points[i].y;

                  //cv::circle(facefilter, cv::Point(eyePoint.rightEye.x[i - 42], eyePoint.rightEye.y[i - 42]), 2, cv::Scalar(255, 0, 0), -1);
              }
    render_face(frame, shape);
//                facebox.y = r.top();
//                facebox.width = r.right() - facebox.x;
//                facebox.height = r.bottom() - facebox.y;
//                cv::rectangle(img, facebox, Scalar(255, 0, 0), 2, 1);

            }
            else
            {
                // Tracking failure detected.
                putText(frame, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
            }

*/

            //----------------------------------------------------------------------------------------------------

 /*          eyesClosedLevel =
               (
               (sqrt(pow(eyePoint.leftEye.x[1] - eyePoint.leftEye.x[5], 2) + pow(eyePoint.leftEye.y[1] - eyePoint.leftEye.y[5], 2)) +
                   sqrt(pow(eyePoint.leftEye.x[2] - eyePoint.leftEye.x[4], 2) + pow(eyePoint.leftEye.y[2] - eyePoint.leftEye.y[4], 2))
                   )
                   / (2 * sqrt(pow(eyePoint.leftEye.x[0] - eyePoint.leftEye.x[3], 2) + pow(eyePoint.leftEye.y[0] - eyePoint.leftEye.y[3], 2)))

                   +

                   (sqrt(pow(eyePoint.rightEye.x[1] - eyePoint.rightEye.x[5], 2) + pow(eyePoint.rightEye.y[1] - eyePoint.rightEye.y[5], 2)) +
                       sqrt(pow(eyePoint.rightEye.x[2] - eyePoint.rightEye.x[4], 2) + pow(eyePoint.rightEye.y[2] - eyePoint.rightEye.y[4], 2))
                       )
                   / (2 * sqrt(pow(eyePoint.rightEye.x[0] - eyePoint.rightEye.x[3], 2) + pow(eyePoint.rightEye.y[0] - eyePoint.rightEye.y[3], 2)))

                   ) / 2;

                   eyesClosedLevel_filter[0] = eyesClosedLevel ;
                   for (int i=14;i>=0;i--)
                   {
                       eyesClosedLevel += eyesClosedLevel_filter[i];

                       eyesClosedLevel_filter[i+1] = eyesClosedLevel_filter[i];

                   }

                   eyesClosedLevel =  eyesClosedLevel/15.0;



           if (eyesClosedLevel < AlarmLevel)
           {
               alarmCount++;
               if (alarmCount > AlarmCount)
               {
                   alarmCount = 0;

               }
           }
           else
           {
               alarmCount = 0;
           }


           char PutString[20]={0};
           sprintf(PutString, "eyeCloseLevel:%f\n", eyesClosedLevel);
           //printf(PutString);
           cv::putText(frame, PutString, cv::Point(250, 20),
               FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 2);


  */
            }
            // Display tracker type on frame
            putText(frame, trackerType + " Tracker", Point(100, 20),
                    FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);


//            sprintf(PutString, "FPS :%lf\n", fps);
//            // Display FPS on frame
//            putText(frame, PutString, Point(100, 50),
//                    FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50));

            // Display frame.
            imshow("Tracking", frameBig);

            // Exit if ESC pressed.
            int k = waitKey(1);
            if (k == 27)
            {
                break;
            }
        }
    }
    catch (exception& e)
    {
        cout << e.what() << endl;
    }


    return 0;
}
