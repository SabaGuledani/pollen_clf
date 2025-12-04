#include "classifiers.hpp"

cv::Ptr<cv::ml::StatModel>
fsiv_create_knn_classifier(int K)
{
    cv::Ptr<cv::ml::KNearest> knn;

    // TODO: Create an KNN classifier.
    // Set algorithm type to BRUTE_FORCE.
    // Set it as a classifier (setIsClassifier)
    // Set hyperparameter K.

    //

    CV_Assert(knn != nullptr);
    return knn;
}

cv::Ptr<cv::ml::StatModel>
fsiv_create_svm_classifier(int Kernel,
                           float C,
                           float degree,
                           float gamma)
{
    cv::Ptr<cv::ml::SVM> svm;
    // TODO: Create an SVM classifier.
    // Set algorithm type to C_SVC.
    // Set it as a classifier (setIsClassifier)
    // Set hyperparameters: C, kernel, Gamma, Degree.

    //
    CV_Assert(svm != nullptr);
    return svm;
}

cv::Ptr<cv::ml::StatModel>
fsiv_create_rtrees_classifier(int V,
                              int T,
                              float E)
{
    cv::Ptr<cv::ml::RTrees> rtrees;
    // TODO: Create an RTrees classifier.
    // REMEMBER: the parameters T and E are set using a cv::TermCriteria.
    // @see opencv docs.

    //
    CV_Assert(rtrees != nullptr);
    return rtrees;
}

void fsiv_train_classifier(cv::Ptr<cv::ml::StatModel> &clf,
                           cv::Mat const &X, cv::Mat const &y)
{
    CV_Assert(clf != nullptr);
    // TODO: train the classifier.

    //
    CV_Assert(clf->isTrained());
}

cv::Mat
fsiv_predict_labels(cv::Ptr<cv::ml::StatModel> &clf, cv::Mat const &X)
{
    CV_Assert(clf != nullptr);
    CV_Assert(clf->isTrained());
    cv::Mat predictions;

    // TODO: compute the predictions.
    // Remember: convert the type of predicted labels to int32.

    //
    CV_Assert(predictions.rows == X.rows);
    CV_Assert(predictions.type() == CV_32SC1);
    return predictions;
}

void fsiv_save_classifier_model(cv::Ptr<cv::ml::StatModel> &clf,
                                const std::string &model_fname)
{
    clf->save(model_fname);
    int id = -1;
    if (dynamic_cast<cv::ml::KNearest *>(clf.get()))
        id = 0;
    else if (dynamic_cast<cv::ml::SVM *>(clf.get()))
        id = 1;
    else if (dynamic_cast<cv::ml::RTrees *>(clf.get()))
        id = 2;
    else
        throw std::runtime_error("Error: unknown classifier type.");
    cv::FileStorage f(model_fname, cv::FileStorage::APPEND);
    if (!f.isOpened())
        throw std::runtime_error("Error: could append classifier type to " +
                                 model_fname);
    f << "fsiv_classifier_type" << id;
}

cv::Ptr<cv::ml::StatModel>
fsiv_load_knn_classifier_model(const std::string &model_fname)
{
    cv::Ptr<cv::ml::StatModel> clsf;

    // TODO: load a KNN classifier.
    // Hint: use the generic interface cv::Algorithm::load< classifier_type >

    //

    CV_Assert(clsf != nullptr);
    return clsf;
}

cv::Ptr<cv::ml::StatModel>
fsiv_load_svm_classifier_model(const std::string &model_fname)
{
    cv::Ptr<cv::ml::StatModel> clsf;

    // TODO: load a SVM classifier.
    // Hint: use the generic interface cv::Algorithm::load< classifier_type >

    //

    CV_Assert(clsf != nullptr);
    return clsf;
}

cv::Ptr<cv::ml::StatModel>
fsiv_load_rtrees_classifier_model(const std::string &model_fname)
{
    cv::Ptr<cv::ml::StatModel> clsf;

    // TODO: load a RTrees classifier.
    // Hint: use the generic interface cv::Algorithm::load< classifier_type >

    //

    CV_Assert(clsf != nullptr);
    return clsf;
}

cv::Ptr<cv::ml::StatModel>
fsiv_load_classifier_model(const std::string &model_fname)
{
    cv::FileStorage f(model_fname, cv::FileStorage::READ);
    if (!f.isOpened())
        std::runtime_error("Error could not read from " + model_fname);
    int id = -1;
    f["fsiv_classifier_type"] >> id;
    f.release();
    cv::Ptr<cv::ml::StatModel> clsf;
    switch (id)
    {
    case 0:
    {
        clsf = fsiv_load_knn_classifier_model(model_fname);
        cv::ml::KNearest *clfs_ = dynamic_cast<cv::ml::KNearest *>(clsf.get());
        std::cout << "Loaded a KNN classifier: K=" << clfs_->getDefaultK() << std::endl;
        break;
    }
    case 1:
    {
        clsf = fsiv_load_svm_classifier_model(model_fname);
        cv::ml::SVM *clfs_ = dynamic_cast<cv::ml::SVM *>(clsf.get());
        std::cout << "Loaded a SVM classifier:" << " K=" << clfs_->getKernelType() << " C=" << clfs_->getC() << " D=" << clfs_->getDegree() << " G=" << clfs_->getGamma() << std::endl;
        break;
    }
    case 2:
    {
        clsf = fsiv_load_rtrees_classifier_model(model_fname);
        cv::ml::RTrees *clfs_ = dynamic_cast<cv::ml::RTrees *>(clsf.get());
        std::cout << "Loaded a RTrees classifier with " << clfs_->getRoots().size() << " trees." << std::endl;
        break;
    }
    default:
    {
        throw std::runtime_error("Unknown classifier id: " + std::to_string(id));
    }
    }
    return clsf;
}
