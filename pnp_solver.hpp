#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

struct CameraPose {
    // 相机在世界坐标系中的位置，单位为毫米
    cv::Vec3d position = {0.0, 0.0, 0.0};

    // 装甲板原点在相机坐标系中的位置，单位为毫米
    cv::Vec3d armor_in_camera = {0.0, 0.0, 0.0};

    double yaw = 0.0;
    double pitch = 0.0;
    double roll = 0.0;

    double reprojection_error = 0.0;
};

class PnpSolver {
public:
    PnpSolver();

    bool solve(
        const std::vector<cv::Point2f>& image_points,
        CameraPose& pose) const;

private:
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    std::vector<cv::Point3f> object_points_;
};
