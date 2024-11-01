#include "baseprocessor.h"

using namespace spdlog;

void concatConfigsGenerator(
    const RequestShapeType &inferenceShape,
    ConcatConfigs &concat,
    const uint8_t padding
) {
    concat.concatDims.clear();
    concat.currIndex = 0;

    const uint16_t concatHeight = inferenceShape[0][1];
    const uint16_t concatWidth = inferenceShape[0][2];

    // Handle different numbers of images (up to 4 for this example)
    if (concat.numImgs == 1) {
        // Single image, it takes up the entire concatenated image (no padding needed)
        ConcatDims dims = {0, 0, concatWidth, concatHeight};
        concat.concatDims.push_back(dims);
    } 
    else if (concat.numImgs == 2) {
        // Two images stacked vertically with 2 pixels of padding between them
        uint16_t imgHeight = (concatHeight - padding) / 2; // Subtract padding, then divide
        ConcatDims dims1 = {0, 0, concatWidth, imgHeight};
        ConcatDims dims2 = {0, imgHeight + padding, concatWidth, imgHeight};
        concat.concatDims.push_back(dims1);  // First image
        concat.concatDims.push_back(dims2);  // Second image
    } 
    else if (concat.numImgs == 3) {
        // Three images stacked vertically with 2 pixels between each
        uint16_t imgHeight = (concatHeight - 2 * padding) / 3; // Subtract 2 gaps, then divide
        ConcatDims dims1 = {0, 0, concatWidth, imgHeight};
        ConcatDims dims2 = {0, imgHeight + padding, concatWidth, imgHeight};
        ConcatDims dims3 = {0, 2 * (imgHeight + padding), concatWidth, imgHeight};
        concat.concatDims.push_back(dims1);  // First image
        concat.concatDims.push_back(dims2);  // Second image
        concat.concatDims.push_back(dims3);  // Third image
    } 
    else if (concat.numImgs == 4) {
        // Four images in a 2x2 grid with 2 pixels of padding between them
        uint16_t imgWidth = (concatWidth - padding) / 2;   // Subtract padding, then divide width
        uint16_t imgHeight = (concatHeight - padding) / 2; // Subtract padding, then divide height
        ConcatDims dims1 = {0, 0, imgWidth, imgHeight};                    // Top-left image
        ConcatDims dims2 = {imgWidth + padding, 0, imgWidth, imgHeight};   // Top-right image
        ConcatDims dims3 = {0, imgHeight + padding, imgWidth, imgHeight};  // Bottom-left image
        ConcatDims dims4 = {imgWidth + padding, imgHeight + padding, imgWidth, imgHeight}; // Bottom-right image
        concat.concatDims.push_back(dims1);  // Top-left image
        concat.concatDims.push_back(dims2);  // Top-right image
        concat.concatDims.push_back(dims3);  // Bottom-left image
        concat.concatDims.push_back(dims4);  // Bottom-right image
    }
}