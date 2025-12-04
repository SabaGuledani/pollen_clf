CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(pollen_clf VERSION 1.1 LANGUAGES CXX)
ENABLE_LANGUAGE(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -Wall")
enable_testing()

FIND_PACKAGE(OpenCV REQUIRED)
LINK_LIBRARIES(${OpenCV_LIBS})
include_directories("${OpenCV_INCLUDE_DIRS}")

set(WITH_OPENMP ON CACHE BOOL "Use openmp for pararell processing.")

if(WITH_OPENMP)
  find_package(OpenMP REQUIRED)

  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DUSE_OPENMP)
  endif(OPENMP_FOUND)
endif(WITH_OPENMP)

add_library(common_code STATIC common_code.hpp
  dataset.cpp dataset.hpp
  classifiers.cpp classifiers.hpp
  metrics.cpp metrics.hpp
  features.cpp features.hpp
  gray_levels_features.hpp gray_levels_features.cpp

  # Add your feature extractors modules here

  )

add_executable(pollen_clf_test_common_code test_common_code.cpp)
target_link_libraries(pollen_clf_test_common_code common_code)
set_target_properties(pollen_clf_test_common_code PROPERTIES OUTPUT_NAME test_common_code)

add_executable(show_BAA500 show_BAA500.cpp)
target_link_libraries(show_BAA500 common_code)

add_executable(train_clf train_clf.cpp)
target_link_libraries(train_clf common_code)

add_executable(test_clf test_clf.cpp)
target_link_libraries(test_clf common_code)

add_test(NAME TestFSIVExtract01NormalizedGrayLevels COMMAND test_common_code fsiv_extract_01_normalized_graylevels)
add_test(NAME TestFSIVCreateKNNClassifier COMMAND test_common_code fsiv_create_knn_classifier)
add_test(NAME TestFSIVCreateSVMClassifier COMMAND test_common_code fsiv_create_svm_classifier)
add_test(NAME TestFSIVCreateRTreesClassifier COMMAND test_common_code fsiv_create_rtrees_classifier)
add_test(NAME TestFSIVTrainClassifier COMMAND test_common_code fsiv_train_classifier)
add_test(NAME TestFSIVPredictLabels COMMAND test_common_code fsiv_predict_labels)
add_test(NAME TestFSIVLoadKNNClassifierModel COMMAND test_common_code fsiv_load_knn_classifier_model)
add_test(NAME TestFSIVLoadSVMClassifierModel COMMAND test_common_code fsiv_load_svm_classifier_model)
add_test(NAME TestFSIVLoadRTreesClassifierModel COMMAND test_common_code fsiv_load_rtrees_classifier_model)
add_test(NAME TestFSIVComputeConfusionMatrix COMMAND test_common_code fsiv_compute_confusion_matrix)
add_test(NAME TestFSIVComputeRecognitionRates COMMAND test_common_code fsiv_compute_recognition_rates)
add_test(NAME TestFSIVComputeAccuracy COMMAND test_common_code fsiv_compute_accuracy)
add_test(NAME TestFSIVComputeMeanRecognitionRate COMMAND test_common_code fsiv_compute_mean_recognition_rate)
