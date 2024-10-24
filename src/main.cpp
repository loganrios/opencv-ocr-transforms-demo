#include <opencv2/opencv.hpp>

#include <concepts>
#include <iostream>
#include <filesystem>
#include <functional>
#include <string>

#include "config.hpp"

using std::string;

namespace transforms {

template <typename F>
concept ImageTransform = requires(F f, cv::Mat m) {
    { f(m) } -> std::convertible_to<cv::Mat>;
};

cv::Mat operator|(cv::Mat m, ImageTransform auto&& transform) {
    return transform(std::move(m));
}

template<typename F>
auto make_cv_op(F&& operation) {
    return [op = std::forward<F>(operation)](cv::Mat in) {
        cv::Mat out;
        op(in, out);
        return out;
    };
}

auto grayscale(int conv_code = cv::COLOR_BGRA2GRAY) {
    return make_cv_op([=](cv::Mat in, cv::Mat& out) {
        cv::cvtColor(in, out, conv_code);
    });
}

auto invert() {
    return make_cv_op([](cv::Mat in, cv::Mat& out) {
        cv::bitwise_not(in, out);
    });
}

auto normalize(int norm_type = cv::NORM_MINMAX) {
    return make_cv_op([=](cv::Mat in, cv::Mat& out) {
        cv::normalize(in, out, 0, 255, norm_type);
    });
}

auto upscale_to(int target_height = 100) {
    return make_cv_op([=](cv::Mat in, cv::Mat& out) {
        if (in.rows > target_height) {
            out = std::move(in);
            return;
        }

        double scale{static_cast<double>(target_height) / in.rows};
        cv::resize(in, out, cv::Size(), scale, scale, cv::INTER_CUBIC);
    });
}

auto binarize() {
    return make_cv_op([=](cv::Mat in, cv::Mat& out) {
        cv::threshold(in, out, 0, 255, cv::THRESH_BINARY
                                       | cv::THRESH_OTSU);
    });
}

inline cv::Mat standard(cv::Mat img) {
    return img
        | grayscale()
        | invert()
        | normalize()
        | upscale_to(1200)
        | binarize();
}

} // transforms

int main() {
    using std::cout;
    using std::endl;

    cv::Mat img = cv::imread(config::test_image.string());

    if(img.empty()) {
            std::cerr << "Error: Could not read image at: "
                      << config::test_image << std::endl;
            return -1;
    }

    namespace t = transforms;

    // using the pipelines itself
    cv::Mat new_img =
        img
        | t::grayscale()
        | t::invert()
        | t::normalize()
        | t::upscale_to(1200)
        | t::binarize();

    cv::imshow("Image", new_img);
    cv::waitKey(0);

    cv::Mat std_img = t::standard(img);

    cout << "Done." << endl;
    return 0;
}
