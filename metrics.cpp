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

    // count predictions: row = true label, col = predicted label
    for (int i = 0; i < true_labels.rows; i++)
    {
        int true_label = true_labels.at<int>(i, 0);
        int pred_label = predicted_labels.at<int>(i, 0);
        cmat.at<float>(true_label, pred_label) += 1.0f;
    }

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

    // for each category: diagonal / row sum
    for (int i = 0; i < cmat.rows; i++)
    {
        float row_sum = 0.0f;
        for (int j = 0; j < cmat.cols; j++)
        {
            row_sum += cmat.at<float>(i, j);
        }
        
        if (row_sum > 0.0f)
        {
            RR.at<float>(i, 0) = cmat.at<float>(i, i) / row_sum;
        }
    }

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

    // sum of diagonal (correct predictions) / total sum
    float diagonal_sum = 0.0f;
    float total_sum = static_cast<float>(cv::sum(cmat)[0]);
    
    for (int i = 0; i < cmat.rows; i++)
    {
        diagonal_sum += cmat.at<float>(i, i);
    }
    
    if (total_sum > 0.0f)
    {
        acc = diagonal_sum / total_sum;
    }

    CV_Assert(acc >= 0.0f && acc <= 1.0f);
    return acc;
}

float fsiv_compute_mean_recognition_rate(const cv::Mat &RRs)
{
    float m_rr = 0.0;
    // TODO

    // compute mean of all recognition rates
    if (RRs.rows > 0)
    {
        float sum = 0.0f;
        for (int i = 0; i < RRs.rows; i++)
        {
            sum += RRs.at<float>(i, 0);
        }
        m_rr = sum / static_cast<float>(RRs.rows);
    }

    return m_rr;
}
