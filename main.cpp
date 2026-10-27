#include "armor_detector.hpp"
#include "pnp_solver.hpp"

#include <opencv2/opencv.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

static cv::Point toPoint(const cv::Point2f& point)
{
    return cv::Point(
        cvRound(point.x),
        cvRound(point.y)
    );
}

static void drawArmor(cv::Mat& image, const Armor& armor)
{
    cv::Scalar color;

    if (armor.color == "blue") {
        color = cv::Scalar(255, 0, 0);
    } else {
        color = cv::Scalar(0, 0, 255);
    }

    for (size_t i = 0; i < armor.pts.size(); ++i) {
        cv::Point p1 = toPoint(armor.pts[i]);
        cv::Point p2 = toPoint(
            armor.pts[(i + 1) % armor.pts.size()]
        );

        cv::line(image, p1, p2, cv::Scalar(0, 0, 255), 2);
        cv::circle(image, p1, 5, color, cv::FILLED);

        cv::putText(
            image,
            std::to_string(i),
            p1 + cv::Point(8, -8),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 255, 255),
            2
        );
    }

    cv::circle(
        image,
        toPoint(armor.center),
        5,
        cv::Scalar(0, 255, 0),
        cv::FILLED
    );

    cv::putText(
        image,
        armor.color,
        toPoint(armor.center) + cv::Point(10, -10),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(0, 255, 0),
        2
    );
}

static std::string formatValue(
    const std::string& name,
    double value,
    const std::string& unit)
{
    std::ostringstream stream;
    stream << name << ": "
           << std::fixed << std::setprecision(1)
           << value << " " << unit;
    return stream.str();
}

static void drawPose(cv::Mat& image, const CameraPose& pose)
{
    int panel_width = 420;
    int panel_height = 240;

    int x = std::max(10, image.cols - panel_width - 25);
    int y = 25;

    cv::rectangle(
        image,
        cv::Rect(x, y, panel_width, panel_height),
        cv::Scalar(40, 40, 40),
        cv::FILLED
    );

    cv::rectangle(
        image,
        cv::Rect(x, y, panel_width, panel_height),
        cv::Scalar(200, 200, 200),
        1
    );

    cv::putText(
        image,
        "Camera Pose",
        cv::Point(x + 20, y + 35),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(0, 255, 255),
        2
    );

    std::vector<std::string> lines = {
        formatValue("X", pose.position[0], "mm"),
        formatValue("Y", pose.position[1], "mm"),
        formatValue("Z", pose.position[2], "mm"),
        formatValue("Yaw", pose.yaw, "deg"),
        formatValue("Pitch", pose.pitch, "deg"),
        formatValue("Roll", pose.roll, "deg"),
        formatValue("Reproj", pose.reprojection_error, "px")
    };

    for (size_t i = 0; i < lines.size(); ++i) {
        cv::putText(
            image,
            lines[i],
            cv::Point(x + 20, y + 65 + static_cast<int>(i) * 25),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(0, 255, 0),
            1
        );
    }
}

int main(int argc, char** argv)
{
    std::string video_path = "raw.mp4";

    if (argc >= 2) {
        video_path = argv[1];
    }

    cv::VideoCapture capture(video_path);

    if (!capture.isOpened()) {
        std::cerr << "Cannot open video: "
                  << video_path << std::endl;
        return 1;
    }

    double fps = capture.get(cv::CAP_PROP_FPS);

    if (fps <= 1.0) {
        fps = 30.0;
    }

    ArmorDetector detector(90);
    PnpSolver pnp_solver;

    cv::VideoWriter writer;
    bool writer_opened = false;

    cv::Mat frame;

    int delay = std::max(
        1,
        cvRound(1000.0 / fps)
    );

    while (capture.read(frame)) {
        if (frame.empty()) continue;

        if (!writer_opened) {
            writer.open(
                "result.mp4",
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                fps,
                frame.size()
            );

            if (!writer.isOpened()) {
                std::cerr << "Cannot create result.mp4"
                          << std::endl;
            }

            writer_opened = true;
        }

        cv::Mat display = frame.clone();

        std::vector<Armor> armors =
            detector.detect(frame);

        if (!armors.empty()) {
            const Armor& target = armors.front();

            drawArmor(display, target);

            CameraPose pose;

            if (pnp_solver.solve(target.pts, pose)) {
                drawPose(display, pose);
            } else {
                cv::putText(
                    display,
                    "PnP failed",
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(0, 0, 255),
                    2
                );
            }
        } else {
            cv::putText(
                display,
                "No armor detected",
                cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 0, 255),
                2
            );
        }

        if (writer.isOpened()) {
            writer.write(display);
        }

        cv::imshow("Armor PnP Pose", display);

        int key = cv::waitKey(delay);

        if (key == 27) {
            break;
        }
    }

    capture.release();
    writer.release();
    cv::destroyAllWindows();

    if (writer.isOpened() || writer_opened) {
        std::cout << "Result saved to result.mp4"
                  << std::endl;
    }

    return 0;
}
