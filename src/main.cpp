#include "../includes/auto_aim/Camera.h"
#include "../includes/auto_aim/Detector.h"

int main(int argc, char ** argv)
{
    std::vector<std::string> class_list;
    std::ifstream ifs("../classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
      class_list.push_back(line);
    }

    cv::dnn::Net net;
    net = cv::dnn::readNet("../best.onnx");

    Camera camera;
    camera.init();

    Detector det;
    cv::Mat frame;

    while (true){
        cv::Mat frame;
        camera.getImage(frame);
        std::vector<cv::Mat> detections;
        detections = det.pre_process(frame, net);
        cv::Mat img = det.post_process(frame, detections, class_list);
        cv::imshow("Output", frame);
        cv::waitKey(10);
    }
// //////////////////////////////////

//    Detector det;
//    cv::Mat frame;
//
//    frame = cv::imread("../sample.jpg");
//    std::vector<cv::Mat> detections;
//    detections = det.pre_process(frame, net);
//    cv::Mat img = det.post_process(frame, detections, class_list);
//    cv::imshow("Output", img);
//    cv::waitKey(0);


  return 0;

}


// to use cuda acceleration
// -net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
// -net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
