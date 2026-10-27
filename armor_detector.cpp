#include "armor_detector.hpp"

#include <algorithm>
#include <cmath>

ArmorDetector::ArmorDetector(int binary_threshold)
    : binary_threshold_(binary_threshold) {}

bool ArmorDetector::extractLightBar(
    const std::vector<cv::Point>& contour,
    const cv::Mat& bgr,
    LightBar& light_bar) const
{
    if (contour.size() < 5) return false;
    if (cv::contourArea(contour) < 10.0) return false;

    cv::RotatedRect rect = cv::minAreaRect(contour);

    cv::Point2f vertices[4];
    rect.points(vertices);

    cv::Point2f edge1 = vertices[1] - vertices[0];
    cv::Point2f edge2 = vertices[2] - vertices[1];

    double length1 = cv::norm(edge1);
    double length2 = cv::norm(edge2);

    cv::Point2f axis;
    double long_length;
    double short_length;

    if (length1 >= length2) {
        long_length = length1;
        short_length = length2;
        axis = cv::Point2f(
            edge1.x / static_cast<float>(length1),
            edge1.y / static_cast<float>(length1)
        );
    } else {
        long_length = length2;
        short_length = length1;
        axis = cv::Point2f(
            edge2.x / static_cast<float>(length2),
            edge2.y / static_cast<float>(length2)
        );
    }

    if (long_length < 8.0 || short_length < 1.0) return false;

    double ratio = long_length / short_length;
    if (ratio < 1.5 || ratio > 20.0) return false;

    // 统一方向，让 axis 指向图像下方
    if (axis.y < 0) {
        axis = -axis;
    }

    light_bar.top =
        rect.center - axis * static_cast<float>(long_length * 0.5);
    light_bar.bottom =
        rect.center + axis * static_cast<float>(long_length * 0.5);
    light_bar.center = rect.center;
    light_bar.axis = axis;
    light_bar.length = long_length;

    long blue_sum = 0;
    long red_sum = 0;

    for (const auto& p : contour) {
        if (p.x < 0 || p.x >= bgr.cols || p.y < 0 || p.y >= bgr.rows) {
            continue;
        }

        cv::Vec3b pixel = bgr.at<cv::Vec3b>(p.y, p.x);
        blue_sum += pixel[0];
        red_sum += pixel[2];
    }

    light_bar.color = (blue_sum > red_sum) ? "blue" : "red";

    return true;
}

std::vector<Armor> ArmorDetector::matchLightBars(
    const std::vector<LightBar>& light_bars) const
{
    std::vector<Armor> armors;

    for (size_t i = 0; i < light_bars.size(); ++i) {
        for (size_t j = i + 1; j < light_bars.size(); ++j) {
            const LightBar* left = &light_bars[i];
            const LightBar* right = &light_bars[j];

            if (left->center.x > right->center.x) {
                std::swap(left, right);
            }

            if (left->color != right->color) continue;

            double max_length =
                std::max(left->length, right->length);
            double min_length =
                std::min(left->length, right->length);

            double length_ratio = max_length / min_length;
            if (length_ratio > 1.35) continue;

            cv::Point2f connector =
                right->center - left->center;
            double distance = cv::norm(connector);

            if (distance < 1.0) continue;

            connector = cv::Point2f(
                connector.x / static_cast<float>(distance),
                connector.y / static_cast<float>(distance)
            );

            double distance_ratio = distance / max_length;
            if (distance_ratio < 1.2 || distance_ratio > 5.0) continue;

            // 检查两条灯条是否接近平行
            double axis_dot = std::abs(
                static_cast<double>(left->axis.dot(right->axis))
            );
            axis_dot = std::clamp(axis_dot, 0.0, 1.0);
            double angle_diff =
                std::acos(axis_dot) * 180.0 / CV_PI;

            if (angle_diff > 15.0) continue;

            // 检查中心连线是否与灯条方向接近垂直
            double perpendicular_error = std::max(
                std::abs(static_cast<double>(left->axis.dot(connector))),
                std::abs(static_cast<double>(right->axis.dot(connector)))
            );

            if (perpendicular_error > 0.35) continue;

            Armor armor;
            armor.pts = {
                left->top,
                right->top,
                right->bottom,
                left->bottom
            };
            armor.center = (left->center + right->center) * 0.5f;
            armor.color = left->color;

            // 分数越小，装甲板越可靠
            armor.score =
                2.0 * std::abs(length_ratio - 1.0) +
                0.05 * angle_diff +
                0.50 * std::abs(distance_ratio - 135.0 / 56.0) +
                0.50 * perpendicular_error;

            armors.push_back(armor);
        }
    }

    std::sort(
        armors.begin(),
        armors.end(),
        [](const Armor& a, const Armor& b) {
            return a.score < b.score;
        }
    );

    return armors;
}

std::vector<Armor> ArmorDetector::detect(const cv::Mat& bgr)
{
    std::vector<Armor> result;

    if (bgr.empty()) return result;

    cv::Mat gray;
    cv::Mat binary;

    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(
        gray,
        binary,
        binary_threshold_,
        255,
        cv::THRESH_BINARY
    );

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(
        binary,
        contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_NONE
    );

    std::vector<LightBar> light_bars;

    for (const auto& contour : contours) {
        LightBar light_bar;

        if (extractLightBar(contour, bgr, light_bar)) {
            light_bars.push_back(light_bar);
        }
    }

    result = matchLightBars(light_bars);
    return result;
}
