#include <iostream>
#include <sstream>
#include <exception>
#include <time.h>
#include <stdlib.h>

// Includes para OpenCV, Descomentar según los módulo utilizados.
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include "common_code.hpp"

#ifndef NDEBUG
int __Debug_Level = 0;
#endif

const char *keys =
    "{help h usage ? |      | print this message   }"
    "{rseed        |0     | Use this value as random seed. Default 0 means use time(0)}"
    "{f_list       |      | List available feature extractors and exits.}"
    "{f_id         |0     | Feature to extract. Default is normalized [0,1] gray levels.}"
    "{f_params     |      | Feature extractor parameters (if any). Format <value>[:<value>:<value>...].}"
    "{f_save_model |      | Filename to save the trained feature extractor model. If empty no model is saved.}"
    "{f_load_model |      | Filename to load a pre-trained feature extractor model. If empty a new model is trained.}"
    "{clf          |0     | Classifier to train/test. 0: K-NN, 1:SVM, 2:RTREES.}"
    "{knn_K        |1     | Parameter K for K-NN class.}"
    "{svm_C        |1.0   | Parameter C for SVM class.}"
    "{svm_K        |0     | Kernel to use with SVM class. 0:Linear, 1:Polynomial. "
    "2:RBF, 3:SIGMOID, 4:CHI2, 5:INTER}"
    "{svm_D        |3.0   | Degree of svm polynomial kernel.}"
    "{svm_G        |1.0   | Gamma for svm RBF kernel.}"
    "{rtrees_V     |0     | Num of random features sampled per node. "
    "Default 0 meas sqrt(num. of total features).}"
    "{rtrees_T     |50    | Max num. of rtrees in the forest.}"
    "{rtrees_E     |0.1   | OOB error to stop adding more rtrees.}"
    "{train_set    |train| Set from the dataset used to train.}"
    "{valid_set    |valid| Set from the dataset used to validation.}"
    "{@dataset     |<none>| Path to the dataset.}"
    "{@model       |<none>| Filename to save the model.}"
#ifndef NDEBUG
    "{verbose        |0     | Set the verbose level.}"
#endif
    ;

std::vector<float>
parse_feature_params(const std::string &f_params)
{
  std::vector<float> feature_params;
  std::istringstream in(f_params);
  float v;
  while (in)
  {
    in >> v;
    if (in)
      feature_params.push_back(v);
  }
  return feature_params;
}

int main(int argc, char *const *argv)
{
  int retCode = EXIT_SUCCESS;

  try
  {

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Train a classifier using the BAA500 pollen dataset.");
    if (parser.has("help"))
    {
      parser.printMessage();
      return 0;
    }
    if (parser.has("f_list"))
    {
      std::cout << "Available feature extractors:" << std::endl;
      for (int fid = 0; fid < int(FeaturesExtractor::FSIV_NEXT_FEATURE_ID); ++fid)
      {
        auto extractor = FeaturesExtractor::create(
            FeaturesExtractor::FEATURE_IDS(fid));
        std::cout << extractor->get_extractor_name() << " (id: " << fid << ")" << std::endl
                  << extractor->get_extractor_help()
                  << std::endl;
      }
      return EXIT_SUCCESS;
    }

#ifndef NDEBUG
    __Debug_Level = parser.get<int>("verbose");
#endif
    auto feature_id = FeaturesExtractor::FEATURE_IDS(parser.get<int>("f_id"));
    std::vector<float> feature_params =
        parse_feature_params(parser.get<std::string>("f_params"));
    std::string f_save_model = parser.get<std::string>("f_save_model");
    std::string f_load_model = parser.get<std::string>("f_load_model");

    std::string train_set = parser.get<std::string>("train_set");
    std::string valid_set = parser.get<std::string>("valid_set");

    std::string dataset_path = parser.get<std::string>("@dataset");
    std::string model_fname = parser.get<std::string>("@model");
    int classifier = parser.get<int>("clf");
    int knn_K = parser.get<int>("knn_K");
    float svm_C = parser.get<float>("svm_C");
    int svm_K = parser.get<int>("svm_K");
    float svm_D = parser.get<float>("svm_D");
    float svm_G = parser.get<float>("svm_G");
    int rtrees_V = parser.get<int>("rtrees_V");
    int rtrees_T = parser.get<int>("rtrees_T");
    double rtrees_E = parser.get<double>("rtrees_E");
    size_t seed = parser.get<size_t>("rseed");
    if (!parser.check())
    {
      parser.printErrors();
      return 0;
    }

    std::cout.setf(std::ios::unitbuf);

    if (seed == 0)
      seed = time(0);
    std::cout << "Set the random seed to: " << seed << std::endl;
    cv::theRNG().state = seed;

    Dataset train_dataset;
    std::cout << "Loading train dataset ...";
    if (!train_dataset.load(dataset_path, train_set))
      throw std::runtime_error("Error: could not open dataset_path path [" + dataset_path + "] or load train set [" + train_set + "]");
    std::cout << " done." << std::endl;
    std::cout << "Train partition with " << train_dataset.size() << " samples."
              << std::endl;

    Dataset valid_dataset;
    std::cout << "Loading validation dataset ... ";
    if (!valid_dataset.load(dataset_path, valid_set))
      throw std::runtime_error("Error: could not open dataset_path path [" + dataset_path + "] or load validation set [" + valid_set + "]");
    std::cout << " done." << std::endl;
    std::cout << "Validation partition with "
              << valid_dataset.size() << " samples." << std::endl;
    std::cout << std::endl;

    auto extractor = FeaturesExtractor::create(feature_id);
    std::cout << "Feature extractor: " << extractor->get_extractor_name()
              << std::endl;
    if (!f_load_model.empty())
    {
      std::cout << "Loading pre-trained feature extractor model from '"
                << f_load_model << "' ... ";
      if (!extractor->load_model(f_load_model))
        throw std::runtime_error("Error: could not load the feature extractor model from file " + f_load_model);
      std::cout << "done." << std::endl;
    }
    else
    {
      extractor->set_params(feature_params);
      std::cout << "Feature extractor params: " << extractor->get_params()
                << std::endl;

      std::cout << "Training feature extractor ... ";
      extractor->train(train_dataset);
      std::cout << "Done." << std::endl;
    }
    if (!f_save_model.empty())
    {
      std::cout << "Saving trained feature extractor model to '"
                << f_save_model << "' ... ";
      if (!extractor->save_model(f_save_model))
        throw std::runtime_error("Error: could not save the feature extractor model to file " + f_save_model);
      std::cout << "done." << std::endl;
    }

    std::cout << "Extracting features in train partition ... ";
    cv::Mat X_t, y_t;
    std::tie(X_t, y_t) = fsiv_extract_features(train_dataset, extractor);
    std::cout << "done." << std::endl;
    std::cout << "Extracted features vector dimension: 1x" << X_t.cols << std::endl;

    cv::Mat X_v, y_v;
    if (valid_dataset.size() > 0)
    {
      std::cout << "Extracting features in validation partition ... ";
      std::tie(X_v, y_v) = fsiv_extract_features(valid_dataset, extractor);
      std::cout << "done." << std::endl;
    }

    std::cout << "Extracted features use " << ((X_t.rows * X_t.cols * X_t.elemSize()) + (X_v.empty() ? 0 : ((X_v.rows * X_v.cols * X_v.elemSize())))) / (1024 * 1024)
              << " Mb of memory." << std::endl;
    std::cout << std::endl;

    cv::Ptr<cv::ml::StatModel> clsf;
    if (classifier == 0)
    {
      std::cout << "Using a K-NN classifier with k=" << knn_K << std::endl;
      clsf = fsiv_create_knn_classifier(knn_K);
    }
    else if (classifier == 1)
    {
      std::cout << "Using a SVM classifier with K=" << svm_K
                << " C=" << svm_C << " D=" << svm_D << " G=" << svm_G
                << std::endl;
      clsf = fsiv_create_svm_classifier(svm_K, svm_C, svm_D, svm_G);
    }
    else if (classifier == 2)
    {
      std::cout << "Using a RTrees classifier with V=" << rtrees_V
                << " T=" << rtrees_T << " E=" << rtrees_E
                << std::endl;
      clsf = fsiv_create_rtrees_classifier(rtrees_V, rtrees_T, rtrees_E);
    }
    else
    {
      std::cerr << "Error: unknown classifier." << std::endl;
      return EXIT_FAILURE;
    }

    std::cout << std::endl;
    std::cout << "Training ... ";
    fsiv_train_classifier(clsf, X_t, y_t);
    std::cout << "done." << std::endl;

    std::cout << "Computing training accuracy ... ";
    cv::Mat predict_labels = fsiv_predict_labels(clsf, X_t);
    cv::Mat cmat = fsiv_compute_confusion_matrix(y_t, predict_labels, 15);
    float acc = fsiv_compute_accuracy(cmat);
    std::cout << "done." << std::endl;
    std::cout << "Training accuracy: " << acc << std::endl;
    std::cout << std::endl;

    if (!X_v.empty())
    {
      std::cout << "Validating ... ";
      predict_labels = fsiv_predict_labels(clsf, X_v);
      std::cout << "done." << std::endl;
      cmat = fsiv_compute_confusion_matrix(y_v, predict_labels, 15);
      acc = fsiv_compute_accuracy(cmat);
      std::cout << "Validation accuracy: " << acc << std::endl;
      std::cout << std::endl;
    }

    std::cout << "Saving the model to '" << model_fname << "'." << std::endl;

    // First, save the classifier's model.
    fsiv_save_classifier_model(clsf, model_fname);

    // Second, save the feature extractor model.
    extractor->save_model(model_fname);
    cv::FileStorage fs(model_fname, cv::FileStorage::APPEND);
    fs << "fsiv_random_seed" << static_cast<double>(seed);
    fs.release();

    // Third, compute model size.
    size_t model_size = 0;
    if (fsiv_compute_file_size(model_fname, model_size))
    {
      float model_size_mb = model_size / (1024.0 * 1024.0);
      std::cout << "Model size: " << model_size_mb << " Mb." << std::endl;
      float size_score = std::max(0.0, 1.0 - (model_size_mb / (4.0 * 45.06)));
      std::cout << "Size score max(0.0, 1.0-(model_size_mb/dataset_size_mb)) = "
                << size_score << std::endl;
      std::cout << "Predicted final score 2*(acc*size_score)/(acc+size_score) = "
                << (2.0 * acc * size_score) / (acc + size_score) << std::endl;
    }
    else
      throw std::runtime_error("Error: could not open the file " + model_fname);
  }
  catch (std::exception &e)
  {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}
