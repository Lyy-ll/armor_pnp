#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

struct Armor {
    // 固定顺序：左上、右上、右下、左下
    std::vector<cv::Point2f> pts;
    cv::Point2f center;
    std::string color;
    double score = 0.0;
};

class ArmorDetector {
public:
    explicit ArmorDetector(int binary_threshold = 90);

    std::vector<Armor> detect(const cv::Mat& bgr);

private:
    struct LightBar {
        cv::Point2f top;
        cv::Point2f bottom;
        cv::Point2f center;
        cv::Point2f axis;
        double length = 0.0;
        std::string color;
    };

    bool extractLightBar(
        const std::vector<cv::Point>& contour,
        const cv::Mat& bgr,
        LightBar& light_bar) const;

    std::vector<Armor> matchLightBars(
        const std::vector<LightBar>& light_bars) const;

private:
    int binary_threshold_;
};
