#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include <iostream>
#include <vector>
using Clock = std::chrono::high_resolution_clock;

// Ajustes de detección
const float CONF_THRESHOLD = 0.25f;
const float NMS_THRESHOLD  = 0.45f;
const cv::Size  INPUT_SIZE   = cv::Size(640, 640);
const std::vector<std::string> CLASS_NAMES = {
    /* 0: "А", 1: "Б", … 32: "Я" */
};

int main() {
    // Carga de modelo ONNX exportado de YOLOv8
    cv::dnn::Net net = cv::dnn::readNetFromONNX("best.onnx");
    // Usa CUDA si está disponible
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

    // Abre video
    cv::VideoCapture cap("input.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error al abrir video" << std::endl;
        return -1;
    }
    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps  = cap.get(cv::CAP_PROP_FPS);
    fps = (fps > 0 ? fps : 30.0);

    cv::VideoWriter writer("output_detect.mp4",
        cv::VideoWriter::fourcc('m','p','4','v'),
        fps, cv::Size(width, height));

    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        // Preprocesamiento
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.f, INPUT_SIZE, cv::Scalar(), true, false);
        net.setInput(blob);

        // Forward
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Postprocesamiento
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        float* data = reinterpret_cast<float*>(outputs[0].data);
        const int dimensions = outputs[0].size[2];   // 85
        const int rows       = outputs[0].size[1];   // N detecciones
        for (int i = 0; i < rows; ++i) {
            float conf = data[4];
            if (conf >= CONF_THRESHOLD) {
                // Encuentra la clase de mayor confianza
                cv::Mat scores(1, dimensions - 5, CV_32FC1, data + 5);
                cv::Point classIdPoint;
                double maxClassScore;
                minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);
                if (maxClassScore > CONF_THRESHOLD) {
                    float cx = data[0] * frame.cols;
                    float cy = data[1] * frame.rows;
                    float w  = data[2] * frame.cols;
                    float h  = data[3] * frame.rows;
                    int left   = static_cast<int>(cx - w / 2);
                    int top    = static_cast<int>(cy - h / 2);
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back(static_cast<float>(maxClassScore));
                    boxes.emplace_back(left, top, static_cast<int>(w), static_cast<int>(h));
                }
            }
            data += dimensions;
        }
        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);

        // Dibujar cuadros
        for (int idx : indices) {
            cv::Rect box = boxes[idx];
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            std::string label = CLASS_NAMES[classIds[idx]] +
                                ": " + cv::format("%.2f", confidences[idx]);
            cv::putText(frame, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0,255,0), 2);
        }

        // Mostrar y guardar
        cv::imshow("Detección YOLO", frame);
        writer.write(frame);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}