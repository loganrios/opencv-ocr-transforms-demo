#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iostream>
#include <string>

#include "config.hpp"
#include "transforms.hpp"

using std::string;

int main() {
  using std::cout;
  using std::endl;

  cv::Mat img = cv::imread(config::test_image.string());

  if (img.empty()) {
    std::cerr << "Error: Could not read image at: " << config::test_image
              << std::endl;
    return -1;
  }

  namespace tf = transforms;

  // using the piping itself
  cv::Mat new_img = img | tf::grayscale() | tf::invert() | tf::normalize() |
                    tf::upscale_to() | tf::binarize();

  cv::Mat myImg2 = tf::standard_pipeline()(img, tf::show_pause);

  cv::imshow("Image", myImg2);
  cv::waitKey(0);

  cv::Mat sameImg = tf::ImagePipeline()(img);

  cv::imshow("Same Image", sameImg);
  cv::waitKey(0);

  ocr::ocr(img); // is an alias for...
  ocr::ocr(img, ocr::transforms::default());

  ocr::ocr(img, ocr::transforms::preset(ocr::LightOnDark));
  ocr::ocr(img, ocr::transforms::preset(ocr::DarkOnLight));

  ocr::ocr(img, ocr::transforms::none());

  using ocr::ocr;
  namespace ot = ocr::transforms;

  ocr(img); // is an alias for...
  ocr(img, ot::defaults());

  ocr(img, ot::preset(ot::LightOnDark));
  ocr(img, ot::preset(ot::DarkOnLight));

  ocr(img, ot::none());
  ocr(img, ot::none().add("grayscale", ot::grayscale()));

  ocr(img, ot::preset(ot::LightOnDark).emplace("upscale", ot::upscale_to(1200)));

  // add, remove, replace
  // construct with TransformSteps or nothing

  cout << "Done." << endl;
  return 0;
}
