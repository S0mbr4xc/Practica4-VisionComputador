#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>       // GaussianFilter, MorphologyFilter
#include <opencv2/cudaimgproc.hpp>       // CannyEdgeDetector, cvtColor, equalizeHist
#include <filesystem>
#include <chrono>
#include <iostream>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Uso: " << argv[0] << " <carpeta_imágenes>" << std::endl;
        return -1;
    }
    std::string input_dir = argv[1];

    std::vector<std::string> files;
    for (auto& p : fs::directory_iterator(input_dir)) {
        if (p.is_regular_file())
            files.push_back(p.path().string());
    }
    if (files.empty()) {
        std::cerr << "No se hallaron imágenes en " << input_dir << std::endl;
        return -1;
    }

    double cpu_total = 0.0, gpu_total = 0.0;
    int count = 0;

    // Crear objetos de filtro CUDA\
    cv::Ptr<cv::cuda::Filter> gauss_gpu = cv::cuda::createGaussianFilter(
        CV_8UC3, CV_8UC3, cv::Size(5,5), 1.5);
    cv::Ptr<cv::cuda::Filter> erode_gpu = cv::cuda::createMorphologyFilter(
        cv::MORPH_ERODE, CV_8UC1,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
    cv::Ptr<cv::cuda::Filter> dilate_gpu = cv::cuda::createMorphologyFilter(
        cv::MORPH_DILATE, CV_8UC1,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny_gpu =
        cv::cuda::createCannyEdgeDetector(50.0, 150.0);

    for (const auto& file : files) {
        cv::Mat img = cv::imread(file);
        if (img.empty()) continue;

        // --- CPU pipeline ---
        auto t0 = Clock::now();
        cv::Mat blur_cpu, gray_cpu, erode_cpu, dilate_cpu, edges_cpu, eq_cpu;
        cv::GaussianBlur(img, blur_cpu, cv::Size(5,5), 1.5);
        cv::cvtColor(blur_cpu, gray_cpu, cv::COLOR_BGR2GRAY);
        cv::erode(gray_cpu, erode_cpu,
                  cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
        cv::dilate(erode_cpu, dilate_cpu,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
        cv::Canny(dilate_cpu, edges_cpu, 50, 150);
        cv::equalizeHist(gray_cpu, eq_cpu);
        auto t1 = Clock::now();
        cpu_total += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // --- GPU pipeline ---
        t0 = Clock::now();
        cv::cuda::GpuMat gimg(img), gblur, ggray, gerode, gdilate, gedges, geq;
        gauss_gpu->apply(gimg, gblur);
        cv::cuda::cvtColor(gblur, ggray, cv::COLOR_BGR2GRAY);
        erode_gpu->apply(ggray, gerode);
        dilate_gpu->apply(gerode, gdilate);
        canny_gpu->detect(gdilate, gedges);
        cv::cuda::equalizeHist(ggray, geq);
        auto t2 = Clock::now();
        gpu_total += std::chrono::duration<double, std::milli>(t2 - t0).count();

        count++;
    }

    std::cout << "Procesadas " << count << " imágenes." << std::endl;
    std::cout << "Tiempo promedio CPU: " << (cpu_total/count)
              << " ms/frame" << std::endl;
    std::cout << "Tiempo promedio GPU: " << (gpu_total/count)
              << " ms/frame" << std::endl;
    return 0;
}