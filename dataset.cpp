#include <iostream>
#include <exception>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "dataset.hpp"

static const std::vector<std::string> fsiv_pollen_label_names_{"alnus", "betula",
                                                               "carpinus", "corylus", "cupressaceae", "fagus",
                                                               "fraxinus", "picea", "pinus", "poaceae",
                                                               "populus", "quercus", "salix", "tilia",
                                                               "urticaceae", "unknown"};
std::unordered_map<std::string, int> Dataset::class_name_to_id_;
Dataset::Dataset()
{
    // Initialize class name to id map the first time.
    if (class_name_to_id_.empty())
    {
        for (size_t i = 0; i < fsiv_pollen_label_names_.size(); ++i)
        {
            class_name_to_id_[fsiv_pollen_label_names_[i]] = static_cast<int>(i);
        }
    }
}

bool Dataset::load(const std::string &folder,
                   const std::string &set_name)
{
    // Label file
    std::string set_filename = folder + "/" + set_name + ".csv";
    // Load label file
    std::ifstream label_file(set_filename);
    if (!label_file)
        return false;

    std::string line;

    std::getline(label_file, line); // skip header line.

    // Load images/labels line by line
    while (std::getline(label_file, line))
    {
        std::stringstream line_stream(line);
        std::string image_filename;
        std::string label;

        if (std::getline(line_stream, image_filename, ',') && line_stream >> label)
        {
            std::string image_path = folder + "/" + image_filename;
            sample_images_.push_back(image_path);
            sample_labels_.push_back(class_name_to_id_[label]);
        }
    }
    return true;
}
cv::Mat Dataset::get_sample(size_t index) const
{
    CV_Assert(index < size());
    cv::Mat img = cv::imread(sample_images_[index], cv::IMREAD_GRAYSCALE);
    // resize to 128x128 if image is not empty (standard size for pollen dataset)
    if (!img.empty() && (img.rows != 128 || img.cols != 128))
    {
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(128, 128));
        return resized;
    }
    return img;
}

std::string Dataset::get_sample_filename(size_t index) const
{
    CV_Assert(index < size());
    return sample_images_[index];
}

int Dataset::get_label(size_t index) const
{
    CV_Assert(index < size());
    return sample_labels_[index];
}
std::tuple<cv::Mat, int> Dataset::operator[](size_t index) const
{
    CV_Assert(index < size());
    cv::Mat img = get_sample(index);
    int label = get_label(index);
    return std::make_tuple(img, label);
}
size_t Dataset::size() const
{
    return sample_images_.size();
}
const std::vector<std::string> &Dataset::get_class_names()
{
    return fsiv_pollen_label_names_;
}

int Dataset::get_class_label(const std::string &class_name) const
{
    auto it = class_name_to_id_.find(class_name);
    if (it != class_name_to_id_.end())
    {
        return it->second;
    }
    else
    {
        throw std::runtime_error("Class name \"" + class_name + "\" not found in dataset.");
    }
}

bool fsiv_compute_file_size(std::string const &path, size_t &size)
{
    bool success = true;
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if (file)
        size = file.tellg();
    else
        success = false;

    return success;
}

static std::vector<std::string> split_string(const std::string &s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream token_stream(s);
    while (std::getline(token_stream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

bool fsiv_save_predictions(const Dataset &dataset, const cv::Mat &predicted_labels,
                           const std::string &predictions_fname,
                           const std::string &header_line)
{
    CV_Assert(dataset.size() > 0);
    CV_Assert(static_cast<size_t>(predicted_labels.rows) == dataset.size());
    CV_Assert(predicted_labels.cols == 1 && predicted_labels.depth() == CV_32S);

    std::ofstream predicted_file(predictions_fname);
    if (!predicted_file.is_open())
        return false;
    if (!header_line.empty())
        predicted_file << header_line << "\n";

    for (size_t i = 0; i < dataset.size(); ++i)
    {
        predicted_file << split_string(dataset.get_sample_filename(i), '/').back()
                       << ","
                       << dataset.get_class_names()[predicted_labels.at<int>(i)] << std::endl;
    }
    return true;
}