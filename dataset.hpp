#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

/** @brief Class to manage a dataset of images and their labels */
class Dataset
{
public:
    /** @brief Constructor */
    Dataset();
    /** @brief Destructor */
    ~Dataset() {}
    /** @brief Load dataset from folder
     * @param folder is the dataset folder path.
     * @param set_name is the set name to load. A file with fname "set_name.csv" is expected.
     * @return true if success.
     */
    bool load(const std::string &folder, const std::string &set_name);

    /** @brief Get the filename of a sample image
     * @param index is the sample index.
     * @return the sample filename.
     * @pre index < size()
     */
    std::string get_sample_filename(size_t index) const;

    /** @brief Get the sample image
     * @param index is the sample index.
     * @return the sample image as a cv::Mat.
     * @pre index < size()
     */
    cv::Mat get_sample(size_t index) const;

    /** @brief Get the label of a sample
     * @param index is the sample index.
     * @return the sample label.
     * @pre index < size()
     */
    int get_label(size_t index) const;

    /** @brief Get a sample as a tuple (image, label)
     * @param index is the sample index.
     * @return a tuple (image, label).
     * @pre index < size()
     */
    std::tuple<cv::Mat, int> operator[](size_t index) const;

    /** @brief Get the number of samples in the dataset
     * @return the number of samples.
     */
    size_t size() const;

    /** @brief Get the class names
     * @return the class names.
     */
    static const std::vector<std::string> &get_class_names();

    /** @brief Get the class label from its name
     * @param class_name is the class name.
     * @return the class label.
     * @throw runtime_error if the class name is not found.
     */
    int get_class_label(const std::string &class_name) const;

private:
    std::vector<std::string> sample_images_;
    std::vector<int> sample_labels_;
    static std::unordered_map<std::string, int> class_name_to_id_;
};

/**
 * @brief Save the predicted labels.
 *
 * @param dataset is the dataset containing the predicted samples.
 * @param predicted_labels are the predicted labels.
 * @param predictions_fname is the pathname of the file were save the predictions.
 * @param header_line is an  header line to write at the beginning of the file. Value "" means no header.
 * @return true if success.
 * @pre dataset.size()>0
 * @pre static_cast<size_t>(predicted_labels.rows)==dataset.size()
 * @pre predicted_labels.cols==1 && predicted_labels.depth()==CV_32S
 * */
bool fsiv_save_predictions(const Dataset &dataset, const cv::Mat &predicted_labels, const std::string &predictions_fname, const std::string &header_line);

/**
 * @brief Compute the size in bytes of a file.
 *
 * @param[in] path is the pathname of the file.
 * @param[out] size is the size of the file in bytes.
 * @return true if success.
 */
bool fsiv_compute_file_size(std::string const &path, size_t &size);
