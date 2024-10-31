#include "transforms.hpp"

#include <utility>
#include <vector>

namespace transforms {

template <typename F> ImageTransform cv_op(F &&operation) {
  return [op = std::forward<F>(operation)](cv::Mat in) {
    cv::Mat out;
    op(in, out);
    return out;
  };
}

ImageTransform grayscale(int conv_code) {
  return cv_op(
      [=](cv::Mat in, cv::Mat &out) { cv::cvtColor(in, out, conv_code); });
}

ImageTransform invert() {
  return cv_op([](cv::Mat in, cv::Mat &out) { cv::bitwise_not(in, out); });
}

ImageTransform normalize(int norm_type) {
  return cv_op([=](cv::Mat in, cv::Mat &out) {
    cv::normalize(in, out, 0, 255, norm_type);
  });
}

ImageTransform upscale_to(int target_height) {
  return cv_op([=](cv::Mat in, cv::Mat &out) {
    if (in.rows > target_height) {
      out = std::move(in);
      return;
    }

    double scale{static_cast<double>(target_height) / in.rows};
    cv::resize(in, out, cv::Size(), scale, scale, cv::INTER_CUBIC);
  });
}

ImageTransform binarize() {
  return cv_op([=](cv::Mat in, cv::Mat &out) {
    cv::threshold(in, out, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  });
}

void show_pause(cv::Mat &in, std::string label) {
  cv::imshow(label, in);
  cv::waitKey(0);
}

cv::Mat standard(cv::Mat img, TextColorsType colors,
                 std::optional<DebugFn> df) {
  using Step = std::pair<std::string, ImageTransform>;
  std::vector<Step> ts;

  if (colors == LightOnDark)
    ts.push_back({"invert", invert()});

  for (Step s : std::vector<Step>{{"grayscale", grayscale()},
                                  {"normalize", normalize()},
                                  {"upscale", upscale_to()},
                                  {"binarize", binarize()}})
    ts.push_back(s);

  cv::Mat out = img;
  for (Step t : ts) {
    out = t.second(out);
    if (df)
      (*df)(out, t.first);
  }

  return out;
}

ImagePipeline standard_pipeline(TextColorsType colors) {
  ImagePipeline p;

  if (colors == LightOnDark)
    p.add(invert(), "invert");

  return p.add(grayscale(), "grayscale")
      .add(normalize(), "normalize")
      .add(upscale_to(), "upscale")
      .add(binarize(), "binarize");
}

struct TransformStep2 {
  std::string label;
  ImageTransform transform;
};

struct ImagePipeline2 {
  std::vector<TransformStep2> steps;

  ImagePipeline2() {}

  ImagePipeline2(std::vector<TransformStep2> steps) : steps(steps) {}

  ImagePipeline2 &add(std::string label, ImageTransform transform) {
    steps.emplace_back(label, transform);
    return *this;
  }

  ImagePipeline2 &add(TransformStep2 step) {
    steps.push_back(step);
    return *this;
  }

  cv::Mat operator()(cv::Mat input, std::optional<DebugFn> d = std::nullopt) {
    cv::Mat curr = std::move(input);

    for (const TransformStep2 &ts : steps) {
      curr = ts.transform(std::move(curr));

      if (d)
        (*d)(curr, ts.label);
    }

    return curr;
  }
};

ImagePipeline2 preset(TextColorsType c) {
  ImagePipeline2 p;

  if (c == LightOnDark)
    p.add("invert", invert());

  return p.add("grayscale", grayscale())
      .add("normalize", normalize())
      .add("upscale", upscale_to())
      .add("binarize", binarize());
}

ImagePipeline2 defaults() { return preset(LightOnDark); }



} // namespace transforms

namespace o {

using std::vector;
using std::string;
using std::optional;
using std::function;
using std::move;
using std::nullopt;
using std::find_if;

template<typename T>
struct Pipeline {
    using Transform = function<T(T)>;
    using Debug = function<void(const T&, const string&)>;

    struct Step {
        string label;
        T transform;
    };

    vector<T> steps;

    Pipeline() {}

    Pipeline(vector<Step> steps) : steps(move(steps)) {}

    Pipeline &add(Step s) {
        steps.push_back(move(s));
        return *this;
    }

    Pipeline &remove(const string& label) {
        auto it = find_if(steps.begin(), steps.end(),
            [&label](const Step& step) {
                return step.label == label;
            });

        if (it != steps.end()) {
            steps.erase(it);
        }

        return *this;
    }

    Pipeline &replace(const string& label, Transform new_transform) {
        auto it = find_if(steps.begin(), steps.end(),
            [&label](const Step& step) {
                return step.label == label;
            });

        if (it != steps.end()) {
            it->transform = move(new_transform);
        }

        return *this;
    }

    T operator()(T input, optional<Debug> d = nullopt) {
        T curr = move(input);

        for (const auto &s : steps) {
            curr = s.transform(move(curr));
            if (d)
                (*d)(curr, s.label);
        }

        return curr;
    }
};

}
