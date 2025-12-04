/*!
  Esto es un esqueleto de programa para usar en las prácticas
  de Visión Artificial.

  Se supone que se utilizará OpenCV.

  Para compilar, puedes ejecutar:
    g++ -Wall -o esqueleto esqueleto.cc `pkg-config opencv --cflags --libs`

*/

#include <iostream>
#include <string>
#include <exception>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "dataset.hpp"

const char *keys =
    "{help h usage ? |      | print this message   }"
    "{show_labels    |      | print the list of class labels and exit.}"
    "{label          |-1    | Label of the samples to show. -1 means show all samples.}"
    "{@dataset       |<none>| folder with the dataset.}"
    "{@set           |<none>| set to load: [train, valid, train_total, test]}";

// Check if this is a Windows system
#ifdef _WIN32
// Arrow keys are different in Windows: ASCII codes of 'n' and 'm'
static const int LEFT_ARROW = 110;
static const int RIGHT_ARROW = 109;
#else
static const int LEFT_ARROW = 81;
static const int RIGHT_ARROW = 83;
#endif

int main(int argc, char *const *argv)
{
    int retCode = EXIT_SUCCESS;

    try
    {
        Dataset dataset;
        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Load the BAA500 dataset (binary format). ");
        if (parser.has("help"))
        {
            parser.printMessage();
            return 0;
        }
        if (parser.has("show_labels"))
        {
            std::cout << "Class labels:" << std::endl;
            for (size_t i = 0; i < dataset.get_class_names().size() - 1; ++i) // -1 to skip 'unknown' label
                std::cout << "  " << i << ": " << dataset.get_class_names()[i] << std::endl;
            return 0;
        }
        std::string dataset_path = parser.get<std::string>("@dataset");
        std::string set = parser.get<std::string>("@set");
        int label_to_show = parser.get<int>("label");
        if (!parser.check())
        {
            parser.printErrors();
            return 0;
        }

        if (label_to_show < -1 || label_to_show >= static_cast<int>(dataset.get_class_names().size() - 1)) // -1 to skip 'unknown' label
        {
            std::cerr << "Error: invalid label to show: " + std::to_string(label_to_show) << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Loading data from folder: " << dataset_path << std::endl;

        if (!dataset.load(dataset_path, set))
            throw std::runtime_error("Error: could not open dataset path [" + dataset_path + "] or load set [" + set + "]");
        std::cout << "Loaded " << dataset.size() << " images with size " << dataset.get_sample(0).size() << std::endl;

        int key = 0;
        int idx = 0;
        std::string wname = "IMAGE";
        cv::namedWindow(wname, cv::WINDOW_GUI_EXPANDED + cv::WINDOW_NORMAL);
        cv::resizeWindow(wname, cv::Size(256, 256));
        cv::Mat X, y;
        if (label_to_show != -1)
        {
            // Find the first sample with the requested label
            while (dataset.get_label(idx) != label_to_show && idx < static_cast<int>(dataset.size()))
                idx++;
            if (idx == static_cast<int>(dataset.size()))
            {
                std::cerr << "Error: no samples with label " << label_to_show << " found." << std::endl;
                return EXIT_FAILURE;
            }
        }
        do
        {
            cv::Mat img = dataset.get_sample(idx);
            cv::imshow(wname, img);
            std::cout << "Idx " << idx << ": "
                      << dataset.get_class_names()[dataset.get_label(idx)]
                      << std::endl;
            key = cv::waitKey(0) & 0xff;
            if (key == LEFT_ARROW)
            {
                do
                {
                    idx = (idx - 1 + static_cast<int>(dataset.size())) % static_cast<int>(dataset.size());
                } while (label_to_show != -1 && dataset.get_label(idx) != label_to_show);
            }
            else if (key == RIGHT_ARROW)
            {
                do
                {
                    idx = (idx + 1) % dataset.size();
                } while (label_to_show != -1 && dataset.get_label(idx) != label_to_show);
            }
            else if (key != 27)
                std::cout << "Unknown keypress code '" << key
                          << "' [Press <-, ->, or ESC]." << std::endl;
        } while (key != 27);

        cv::destroyWindow(wname);
    }
    catch (std::exception &e)
    {
        std::cerr << "Capturada excepcion: " << e.what() << std::endl;
        retCode = EXIT_FAILURE;
    }
    catch (...)
    {
        std::cerr << "Capturada excepcion desconocida!" << std::endl;
        retCode = EXIT_FAILURE;
    }
    return retCode;
}
