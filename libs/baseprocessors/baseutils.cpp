#include "baseprocessor.h"

using namespace spdlog;

void concatConfigsGenerator(
    const RequestShapeType &inferenceShape,
    ConcatConfigs &concat,
    const uint8_t padding
) {
    concat.list.clear();
    // Empty config for 0 image case
    concat.list.push_back({});

    const uint16_t concatHeight = inferenceShape[0][1];
    const uint16_t concatWidth = inferenceShape[0][2];

    // Handle different numbers of images (up to 4 for this example)
    // Single image, it takes up the entire concatenated image (no padding needed)
    ConcatConfig config = {{0, 0, concatWidth, concatHeight}};
    concat.list.push_back(config);

    uint16_t imgHeight, imgWidth;
    ConcatDims dims1, dims2, dims3, dims4;
    
    // Two images stacked vertically with 2 pixels of padding between them
    imgHeight = (concatHeight - padding) / 2; // Subtract padding, then divide
    dims1 = {0, 0, concatWidth, imgHeight}; // Top image
    dims2 = {0, imgHeight + padding, concatWidth, imgHeight};    // Bottom image
    concat.list.push_back({dims1, dims2});


    // Three images stacked vertically with 2 pixels between each
    imgHeight = (concatHeight - 2 * padding) / 3; // Subtract 2 gaps, then divide
    dims1 = {0, 0, concatWidth, imgHeight}; // Top image
    dims2 = {0, imgHeight + padding, concatWidth, imgHeight}; // Middle image
    dims3 = {0, 2 * imgHeight + 2 * padding, concatWidth, imgHeight}; // Bottom image
    concat.list.push_back({dims1, dims2, dims3});

    // Four images in a 2x2 grid with 2 pixels of padding between them
    imgWidth = (concatWidth - padding) / 2;   // Subtract padding, then divide width
    imgHeight = (concatHeight - padding) / 2; // Subtract padding, then divide height
    dims1 = {0, 0, imgWidth, imgHeight}; // Top-left image
    dims2 = {imgWidth + padding, 0, imgWidth, imgHeight}; // Top-right image
    dims3 = {0, imgHeight + padding, imgWidth, imgHeight}; // Bottom-left image
    dims4 = {imgWidth + padding, imgHeight + padding, imgWidth, imgHeight}; // Bottom-right image
    concat.list.push_back({dims1, dims2, dims3, dims4});
}