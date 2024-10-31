#pragma once

#include <opencv2/opencv.hpp>

#include <functional>
#include <optional>
#include <string>

namespace transforms {

using ImageTransform = std::function<cv::Mat(cv::Mat)>;

ImageTransform grayscale(int conv_code = cv::COLOR_BGRA2GRAY);
ImageTransform invert();
ImageTransform normalize(int norm_type = cv::NORM_MINMAX);
ImageTransform upscale_to(int target_height = 100);
ImageTransform binarize();

enum class TextColorsType { DarkOnLight, LightOnDark };

using enum TextColorsType;

using DebugFn = std::function<void(cv::Mat &, std::string)>;
void show_pause(cv::Mat &in, std::string label);

using TransformStep = std::pair<ImageTransform, std::string>;

struct ImagePipeline {
  std::vector<TransformStep> steps;

  ImagePipeline &add(ImageTransform t, std::string name) {
    steps.emplace_back(std::move(t), std::move(name));
    return *this;
  }

  cv::Mat operator()(cv::Mat input, std::optional<DebugFn> dbg = std::nullopt) {
    cv::Mat current = std::move(input);
    for (const auto &[transform, step_name] : steps) {
      current = transform(std::move(current));
      if (dbg)
        (*dbg)(current, step_name);
    }
    return current;
  }

  ImagePipeline &append(const ImagePipeline &other) {
    steps.insert(steps.end(), other.steps.begin(), other.steps.end());
    return *this;
  }
};

ImagePipeline standard_pipeline(TextColorsType colors = LightOnDark);

} // namespace transforms

inline cv::Mat operator|(cv::Mat m, transforms::ImageTransform f) {
  return f(std::move(m));
}
