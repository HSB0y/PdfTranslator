cmake_minimum_required(VERSION 3.20)
project(PdfTranslator LANGUAGES CXX VERSION 0.0.1)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/deploy)

include(CheckIncludeFileCXX)

check_include_file_cxx(any HAS_ANY)
check_include_file_cxx(string_view HAS_STRING_VIEW)
check_include_file_cxx(coroutine HAS_COROUTINE)
if (NOT "${CMAKE_CXX_STANDARD}" STREQUAL "")
    # Do nothing
elseif (HAS_ANY AND HAS_STRING_VIEW AND HAS_COROUTINE)
    set(CMAKE_CXX_STANDARD 20)
elseif (HAS_ANY AND HAS_STRING_VIEW)
    set(CMAKE_CXX_STANDARD 17)
else ()
    set(CMAKE_CXX_STANDARD 14)
endif ()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB LLM_SRC "./llm/*.cpp")

if(WIN32)
	set(CMAKE_BUILD_TYPE Release)
    add_executable(${PROJECT_NAME} WIN32 main.cc ${LLM_SRC})
else()
    add_executable(${PROJECT_NAME} main.cc ${LLM_SRC})
endif()

# ##############################################################################
# If you include the drogon source code locally in your project, use this method
# to add drogon 
# add_subdirectory(drogon) 
# target_link_libraries(${PROJECT_NAME} PRIVATE drogon)
#
# and comment out the following lines
find_package(Drogon CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Drogon::Drogon)

find_package(webview REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE webview::core_static)

if (NOT DEFINED ONNXRUNTIME_DIR)
	message(FATAL_ERROR "Variable $\{ONNXRUNTIME_DIR\} is not set. You should download onnxruntime from https://github.com/microsoft/onnxruntime/releases/tag/v1.17.1 first then set the var to point to the unzipped path.")
endif ()

if (NOT DEFINED SENTENCEPIECE_DIR)
	message(FATAL_ERROR "Variable $\{SENTENCEPIECE_DIR\} is not set. You should download sentencepiece from https://github.com/google/sentencepiece/releases/tag/v0.2.1 first then set the var to point to the unzipped path.")
endif ()

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR} "${ONNXRUNTIME_DIR}/include" "${SENTENCEPIECE_DIR}/include")
target_link_directories(${PROJECT_NAME} PRIVATE "${ONNXRUNTIME_DIR}/lib" "${SENTENCEPIECE_DIR}/lib")
target_link_libraries(${PROJECT_NAME} PRIVATE sentencepiece sentencepiece_train)
target_link_libraries(${PROJECT_NAME} PRIVATE onnxruntime)

target_include_directories(${PROJECT_NAME} PRIVATE
    "${PROJECT_SOURCE_DIR}/3rd_party/json/single_include"
    "${PROJECT_SOURCE_DIR}/3rd_party/bimap"
)

# ##############################################################################

if (CMAKE_CXX_STANDARD LESS 17)
    message(FATAL_ERROR "c++17 or higher is required")
elseif (CMAKE_CXX_STANDARD LESS 20)
    message(STATUS "use c++17")
else ()
    message(STATUS "use c++20")
endif ()

aux_source_directory(controllers CTL_SRC)
aux_source_directory(filters FILTER_SRC)
aux_source_directory(plugins PLUGIN_SRC)
aux_source_directory(models MODEL_SRC)

drogon_create_views(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/views
                    ${CMAKE_CURRENT_BINARY_DIR})
# use the following line to create views with namespaces.
# drogon_create_views(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/views
#                     ${CMAKE_CURRENT_BINARY_DIR} TRUE)
# use the following line to create views with namespace CHANGE_ME prefixed
# and path namespaces.
# drogon_create_views(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/views
#                     ${CMAKE_CURRENT_BINARY_DIR} TRUE CHANGE_ME)

target_include_directories(${PROJECT_NAME}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                   ${CMAKE_CURRENT_SOURCE_DIR}/models
                                   ${CMAKE_CURRENT_SOURCE_DIR}/llm)
target_sources(${PROJECT_NAME}
               PRIVATE
               ${SRC_DIR}
               ${CTL_SRC}
               ${FILTER_SRC}
               ${PLUGIN_SRC}
               ${MODEL_SRC})
# ##############################################################################
# uncomment the following line for dynamically loading views 
# set_property(TARGET ${PROJECT_NAME} PROPERTY ENABLE_EXPORTS ON)

# ##############################################################################

# add_subdirectory(test)
