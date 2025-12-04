#include "metrics.hpp"

cv::Mat
fsiv_compute_confusion_matrix(const cv::Mat &true_labels,
                              const cv::Mat &predicted_labels,
                              int n_categories)
{
    CV_Assert(true_labels.rows == predicted_labels.rows);
    CV_Assert(true_labels.type() == CV_32SC1);
    CV_Assert(predicted_labels.type() == CV_32SC1);
    cv::Mat cmat = cv::Mat::zeros(n_categories, n_categories, CV_32F);

    // TODO: Compute the confusion matrix.
    // Remember: Rows are the Ground Truth. Cols are the predictions.

    //
    CV_Assert(cmat.type() == CV_32FC1);
    CV_Assert(std::abs(cv::sum(cmat)[0] - static_cast<double>(true_labels.rows)) <= 1.0e-6);
    return cmat;
}

cv::Mat
fsiv_compute_recognition_rates(const cv::Mat &cmat)
{
    CV_Assert(!cmat.empty() && cmat.type() == CV_32FC1);
    CV_Assert(cmat.rows == cmat.cols);
    cv::Mat RR = cv::Mat::zeros(cmat.rows, 1, CV_32FC1);

    // TODO
    // Hint: Compute the recognition rate (RR) for the each category (row).

    //
    CV_Assert(RR.rows == cmat.rows && RR.cols == 1);
    CV_Assert(RR.type() == CV_32FC1);
    return RR;
}

float fsiv_compute_accuracy(const cv::Mat &cmat)
{
    CV_Assert(!cmat.empty() && cmat.type() == CV_32FC1);
    CV_Assert(cmat.rows == cmat.cols && cmat.rows > 1);

    float acc = 0.0;

    // TODO: compute the accuracy.
    // Hint: the accuracy is the rate of correct classifications
    //   to the total.
    // Remember: avoid zero divisions!!.

    //
    CV_Assert(acc >= 0.0f && acc <= 1.0f);
    return acc;
}

float fsiv_compute_mean_recognition_rate(const cv::Mat &RRs)
{
    float m_rr = 0.0;
    // TODO

    //
    return m_rr;
}
