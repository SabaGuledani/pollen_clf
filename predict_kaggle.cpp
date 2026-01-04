#include <iostream>
#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include "common_code.hpp"

const char *keys =
    "{help h usage ? |      | print this message   }"
    "{@dataset       |<none>| Dataset pathname.}"
    "{@model         |<none>| Model filename.}"
    "{@output        |<none>| Output CSV filename for Kaggle submission.}";

int main(int argc, char *const *argv)
{
    int retCode = EXIT_SUCCESS;

    try
    {
        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Generate Kaggle submission predictions.");
        
        if (parser.has("help"))
        {
            parser.printMessage();
            return 0;
        }

        std::string dataset_path = parser.get<std::string>("@dataset");
        std::string model_fname = parser.get<std::string>("@model");
        std::string output_fname = parser.get<std::string>("@output");
        
        if (!parser.check())
        {
            parser.printErrors();
            return EXIT_FAILURE;
        }

        std::cout.setf(std::ios::unitbuf);

        // Load test dataset
        Dataset test_dataset;
        std::cout << "Loading test dataset from '" << dataset_path << "' ... ";
        if (!test_dataset.load(dataset_path, "test"))
        {
            std::cerr << "Error: could not load test set from [" << dataset_path << "]" << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "done." << std::endl;
        std::cout << "Test data with " << test_dataset.size() << " samples." << std::endl;
        std::cout << std::endl;

        // Load model
        std::cout << "Loading model from '" << model_fname << "' ... ";
        cv::Ptr<cv::ml::StatModel> clsf = fsiv_load_classifier_model(model_fname);
        if (clsf == nullptr || !clsf->isTrained())
        {
            std::cerr << "Error: Could not load trained model!" << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "done." << std::endl;

        // Load feature extractor
        auto extractor = FeaturesExtractor::create(model_fname);
        std::cout << "Feature extractor: " << extractor->get_extractor_name() << std::endl;
        std::cout << "Feature extractor params: " << extractor->get_params() << std::endl;
        std::cout << std::endl;

        // Extract features
        std::cout << "Extracting features ... ";
        cv::Mat X, y;
        std::tie(X, y) = fsiv_extract_features(test_dataset, extractor);
        std::cout << "done." << std::endl;
        std::cout << "Original feature dimension: " << X.cols << std::endl;
        std::cout << std::endl;

        // Load and apply PCA if it was used during training
        cv::PCA pca = fsiv_load_pca_model(model_fname);
        if (!pca.eigenvectors.empty())
        {
            std::cout << "Applying PCA transformation ... ";
            fsiv_transform_pca(pca, X);
            std::cout << "done." << std::endl;
            std::cout << "Features reduced to " << X.cols << " dimensions after PCA." << std::endl;
            std::cout << std::endl;
        }

        // Make predictions
        std::cout << "Computing predictions ... ";
        cv::Mat predicted_labels = fsiv_predict_labels(clsf, X);
        std::cout << "done." << std::endl;
        std::cout << std::endl;

        // Save predictions in Kaggle format (sample,species)
        std::cout << "Saving predictions to '" << output_fname << "' ... ";
        if (!fsiv_save_predictions(test_dataset, predicted_labels, output_fname, "sample,species"))
        {
            std::cerr << "Error: could not save predictions to file " << output_fname << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "done." << std::endl;
        std::cout << std::endl;
        std::cout << "Kaggle submission file created: " << output_fname << std::endl;
        std::cout << "Ready for submission!" << std::endl;
    }
    catch (std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        retCode = EXIT_FAILURE;
    }
    return retCode;
}

