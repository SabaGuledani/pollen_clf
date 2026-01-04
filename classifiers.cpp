#include "classifiers.hpp"

cv::Ptr<cv::ml::StatModel>
fsiv_create_knn_classifier(int K)
{
    cv::Ptr<cv::ml::KNearest> knn;
    knn = cv::ml::KNearest::create();
    knn->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
    knn->setIsClassifier(true);
    knn->setDefaultK(K);

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
    svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC); // set type to C_SVC for multiclass
    svm->setC(C);
    
    // set kernel and params
    switch (Kernel)
    {
    case 0: // Linear 
        svm->setKernel(cv::ml::SVM::LINEAR);
        break;
    case 1: // Polynomial 
        svm->setKernel(cv::ml::SVM::POLY);
        svm->setDegree(degree);
        svm->setGamma(gamma);
        break;
    case 2: // RBF 
        svm->setKernel(cv::ml::SVM::RBF);
        svm->setGamma(gamma);
        break;
    case 3: // Sigmoid 
        svm->setKernel(cv::ml::SVM::SIGMOID);
        svm->setGamma(gamma);
        break;
    case 4: // CHI2 
        svm->setKernel(cv::ml::SVM::CHI2);
        svm->setGamma(gamma);
        break;
    case 5: // INTER
        svm->setKernel(cv::ml::SVM::INTER);
        break;
    default:
        // default to RBF
        svm->setKernel(cv::ml::SVM::RBF);
        svm->setGamma(gamma);
        break;
    }

    CV_Assert(svm != nullptr);
    return svm;
}

cv::Ptr<cv::ml::StatModel>
fsiv_create_rtrees_classifier(int V,
                              int T,
                              float E)
{
    
    cv::Ptr<cv::ml::RTrees> rtrees = cv::ml::RTrees::create();
    
    if (rtrees.empty() || rtrees.get() == nullptr)
    {
        throw std::runtime_error("Error: Failed to create RTrees classifier.");
    }

    // set number of features used per node
    if (V > 0)
    {
        rtrees->setActiveVarCount(V);
    }
    
    // set termination criteria using T max iterations and E epsilon
    cv::TermCriteria term_crit(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, T, static_cast<double>(E));
    rtrees->setTermCriteria(term_crit);
    
    // set other useful parameters
    rtrees->setMaxDepth(20);  
    rtrees->setMinSampleCount(5);  
    rtrees->setMaxCategories(15); 
    rtrees->setCalculateVarImportance(true);
    
    return rtrees;
}

void fsiv_train_classifier(cv::Ptr<cv::ml::StatModel> &clf,
                           cv::Mat const &X, cv::Mat const &y)
{
    CV_Assert(clf != nullptr);
    clf->train(X, cv::ml::ROW_SAMPLE, y);

    CV_Assert(clf->isTrained());
}

cv::Mat
fsiv_predict_labels(cv::Ptr<cv::ml::StatModel> &clf, cv::Mat const &X)
{
    CV_Assert(clf != nullptr);
    CV_Assert(clf->isTrained());
    cv::Mat predictions;

    // predict labels
    clf->predict(X, predictions);
    // convert to int32
    predictions.convertTo(predictions, CV_32SC1);

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

    // load KNN from file
    clsf = cv::Algorithm::load<cv::ml::KNearest>(model_fname);

    CV_Assert(clsf != nullptr);
    return clsf;
}

cv::Ptr<cv::ml::StatModel>
fsiv_load_svm_classifier_model(const std::string &model_fname)
{
    cv::Ptr<cv::ml::StatModel> clsf;

    // load SVM from file
    clsf = cv::Algorithm::load<cv::ml::SVM>(model_fname);

    CV_Assert(clsf != nullptr);
    return clsf;
}

cv::Ptr<cv::ml::StatModel>
fsiv_load_rtrees_classifier_model(const std::string &model_fname)
{
    cv::Ptr<cv::ml::StatModel> clsf;

    // load RTrees from file
    clsf = cv::Algorithm::load<cv::ml::RTrees>(model_fname);

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
