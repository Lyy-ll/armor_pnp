#include "pnp_solver.hpp"

#include <cmath>

PnpSolver::PnpSolver()
{
    camera_matrix_ = (cv::Mat_<double>(3, 3) <<
        1789.2913422305087, 0, 702.93444417420312,
        0, 1789.214342198638, 557.65168675325629,
        0, 0, 1
    );

    dist_coeffs_ = (cv::Mat_<double>(1, 5) <<
        -0.075551988063684988,
         0.13117079840026666,
         0.0014612753636792362,
        -0.001318090407814957,
         0
    );

    /*
     * 世界坐标定义：
     * 原点：装甲板中心
     * X轴：沿装甲板向右
     * Y轴：沿装甲板向上
     * Z轴：垂直装甲板指向相机
     *
     * 单位：毫米
     * 点顺序：左上、右上、右下、左下
     */
    object_points_ = {
        {-67.5f,  28.0f, 0.0f},
        { 67.5f,  28.0f, 0.0f},
        { 67.5f, -28.0f, 0.0f},
        {-67.5f, -28.0f, 0.0f}
    };
}

bool PnpSolver::solve(
    const std::vector<cv::Point2f>& image_points,
    CameraPose& pose) const
{
    if (image_points.size() != 4) return false;

    cv::Mat rvec;
    cv::Mat tvec;

    bool success = cv::solvePnP(
        object_points_,
        image_points,
        camera_matrix_,
        dist_coeffs_,
        rvec,
        tvec,
        false,
        cv::SOLVEPNP_IPPE
    );

    if (!success) return false;
    if (tvec.at<double>(2) <= 0.0) return false;

    // 重投影检查
    std::vector<cv::Point2f> reprojected_points;

    cv::projectPoints(
        object_points_,
        rvec,
        tvec,
        camera_matrix_,
        dist_coeffs_,
        reprojected_points
    );

    double error_sum = 0.0;

    for (size_t i = 0; i < image_points.size(); ++i) {
        error_sum += cv::norm(
            reprojected_points[i] - image_points[i]
        );
    }

    double mean_error =
        error_sum / static_cast<double>(image_points.size());

    if (mean_error > 8.0) return false;

    /*
     * solvePnP 得到的是：
     * 世界坐标 -> 相机坐标
     *
     * X_camera = R * X_world + t
     */
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // 相机坐标 -> 世界坐标
    cv::Mat R_world_camera = R.t();

    // 相机光心在世界坐标系中的位置
    cv::Mat camera_center =
        -R_world_camera * tvec;

    pose.position = cv::Vec3d(
        camera_center.at<double>(0),
        camera_center.at<double>(1),
        camera_center.at<double>(2)
    );

    pose.armor_in_camera = cv::Vec3d(
        tvec.at<double>(0),
        tvec.at<double>(1),
        tvec.at<double>(2)
    );

    /*
     * 按下面的旋转顺序提取欧拉角：
     * R = Rz(yaw) * Ry(pitch) * Rx(roll)
     */
    double sy = std::hypot(
        R_world_camera.at<double>(0, 0),
        R_world_camera.at<double>(1, 0)
    );

    double roll;
    double pitch;
    double yaw;

    if (sy > 1e-6) {
        roll = std::atan2(
            R_world_camera.at<double>(2, 1),
            R_world_camera.at<double>(2, 2)
        );

        pitch = std::atan2(
            -R_world_camera.at<double>(2, 0),
            sy
        );

        yaw = std::atan2(
            R_world_camera.at<double>(1, 0),
            R_world_camera.at<double>(0, 0)
        );
    } else {
        roll = std::atan2(
            -R_world_camera.at<double>(1, 2),
             R_world_camera.at<double>(1, 1)
        );

        pitch = std::atan2(
            -R_world_camera.at<double>(2, 0),
            sy
        );

        yaw = 0.0;
    }

    constexpr double RAD_TO_DEG = 180.0 / CV_PI;

    pose.roll = roll * RAD_TO_DEG;
    pose.pitch = pitch * RAD_TO_DEG;
    pose.yaw = yaw * RAD_TO_DEG;
    pose.reprojection_error = mean_error;

    return true;
}
