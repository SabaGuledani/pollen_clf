/*!
  Esto es un esqueleto de programa para usar en las prácticas
  de Visión Artificial.

  Se supone que se utilizará OpenCV.

  Para compilar, puedes ejecutar:
    g++ -Wall -o esqueleto esqueleto.cc `pkg-config opencv --cflags --libs`

*/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <exception>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include "common_code.hpp"

#ifndef NDEBUG
int __Debug_Level = 0;
#endif

const char *keys =
    "{help h usage ? |      | print this message   }"
    "{t              |      | Only get test labels (no metrics), used for final upload.}"
#ifndef NDEBUG
    "{verbose        |0     | Set the verbose level.}"
#endif
    "{@dataset       |<none>| Dataset pathname.}"
    "{@set          |<none>| Set name to use (train, valid, train_total, test).}"
    "{@model         |<none>| Model filename to test.}"
    "{@predictions   |<none>| Fname where save the predictions}";

int main(int argc, char *const *argv)
{
  int retCode = EXIT_SUCCESS;

  try
  {

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Test a classifier using the Fashion MNIST dataset.");
    if (parser.has("help"))
    {
      parser.printMessage();
      return 0;
    }

#ifndef NDEBUG
    __Debug_Level = parser.get<int>("verbose");
#endif
    std::string dataset_path = parser.get<std::string>("@dataset");
    std::string set_name = parser.get<std::string>("@set");
    std::string model_fname = parser.get<std::string>("@model");
    std::string predictions_fname = parser.get<std::string>("@predictions");
    bool only_test = parser.has("t");
    if (!parser.check())
    {
      parser.printErrors();
      return 0;
    }

    std::cout.setf(std::ios::unitbuf);

    Dataset test_dataset;
    std::cout << "Loading set '" << set_name << "' from dataset ... ";
    if (!test_dataset.load(dataset_path, set_name))
    {
      std::cerr << "Error: could not open dataset path ["
                << dataset_path << "] or load set ["
                << set_name << "]" << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "done." << std::endl;

    std::cout << "Test data with " << test_dataset.size() << " samples."
              << std::endl;
    std::cout << std::endl;
    std::cout << "Loading classifier model from file "
              << model_fname << " ... " << std::endl;
    cv::Ptr<cv::ml::StatModel> clsf = fsiv_load_classifier_model(model_fname);
    if (clsf == nullptr || !clsf->isTrained())
    {
      std::cerr << "Error: I need a trained model!" << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << std::endl;
    auto extractor = FeaturesExtractor::create(model_fname);
    std::cout << "Feature extractor: " << extractor->get_extractor_name()
              << std::endl;
    std::cout << "Feature extractor params: " << extractor->get_params()
              << std::endl;
    std::cout << "Extracting features ... ";
    cv::Mat X, y;
    std::tie(X, y) = fsiv_extract_features(test_dataset, extractor);
    std::cout << "done." << std::endl;

    std::cout << std::endl;
    std::cout << "Computing predictions ... ";
    cv::Mat predicted_labels = fsiv_predict_labels(clsf, X);
    std::cout << "done." << std::endl;

    std::cout << "Saving predictions to file " << predictions_fname
              << " ... ";
    if (!fsiv_save_predictions(test_dataset, predicted_labels, predictions_fname, "sample,species"))
    {
      std::cerr << "Error: could not save predictions to file "
                << predictions_fname << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "done." << std::endl;

    if (only_test == false)
    {
      std::cout << std::endl;
      std::cout << "Computing metrics ... ";
      cv::Mat cmat = fsiv_compute_confusion_matrix(y, predicted_labels,
                                                   static_cast<int>(test_dataset.get_class_names().size()));
      float acc = fsiv_compute_accuracy(cmat);
      cv::Mat RRs = fsiv_compute_recognition_rates(cmat);
      float m_rr = fsiv_compute_mean_recognition_rate(RRs);
      std::cout << "done.\n"
                << std::endl;

      std::cout << std::endl;
      std::cout << "Model metrics #########################\n"
                << std::endl;
      std::cout << "Recognition Rate per class:";
      for (int i = 0; i < RRs.rows; ++i)
        std::cout << std::setw(15) << test_dataset.get_class_names()[i]
                  << ": " << std::setw(4) << (RRs.at<float>(i) * 100.0) << "%" << std::endl;
      std::cout << std::endl;
      std::cout << "mRR:\t" << m_rr << std::endl;
      std::cout << "Acc:\t" << acc << std::endl;
      size_t model_size = 0;
      if (fsiv_compute_file_size(model_fname, model_size))
      {
        float model_size_mb = model_size / (1024.0 * 1024.0);
        std::cout << "Model size: " << model_size_mb << " Mb." << std::endl;
        float size_score = std::max(0.0, 1.0 - (model_size_mb / (4.0 * 45.06)));
        std::cout << "Size score max(0.0, 1.0-(model_size_mb/dataset_size_mb)) = "
                  << size_score << std::endl;
        std::cout << "Test final score 2*(acc*size_score)/(acc+size_score) = "
                  << (2.0 * acc * size_score) / (acc + size_score) << std::endl;
      }
      else
        throw std::runtime_error("Error: could not open the file " + model_fname);
    }
  }
  catch (std::exception &e)
  {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}
