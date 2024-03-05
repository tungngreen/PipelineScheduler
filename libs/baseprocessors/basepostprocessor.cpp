#include "basepostprocessor.h"

/**
 * @brief Scale the bounding box coordinates to the original aspect ratio of the image
 * 
 * @param orig_h Original height of the image
 * @param orig_w Original width of the image
 * @param infer_h Height of the image used for inference
 * @param infer_w Width of the image used for inference
 * @param bbox_coors [x1, y1, x2, y2]
 */
void scaleBBox(
    int orig_h,
    int orig_w,
    int infer_h,
    int infer_w,
    const float &infer_bboxCoors,
    int &orig_bboxCoors
) {
    // TO BE IMPLEMENTED
    for (uint8_t i = 0; i < 4; ++i) {
        *(&orig_bboxCoors + i) = (int)*(&infer_bboxCoors + i);
    }
}

void crop(
    const cv::cuda::GpuMat &image,
    int infer_h,
    int infer_w,
    int numDetections,
    const float &bbox_coorList,
    std::vector<cv::cuda::GpuMat> &croppedBBoxes
) {
    int orig_h, orig_w;
    orig_h = image.rows;
    orig_w = image.cols;
    int orig_bboxCoors[4];
    for (int i = 0; i < numDetections; ++i) {
        scaleBBox(orig_h, orig_w, infer_h, infer_w, bbox_coorList + i * 4, *orig_bboxCoors);
        cv::cuda::GpuMat croppedBBox = image(cv::Range((int)orig_bboxCoors[0], (int)orig_bboxCoors[2]), cv::Range((int)orig_bboxCoors[1], (int)orig_bboxCoors[3])).clone();
        croppedBBoxes.emplace_back(croppedBBox);
    }
}

template<typename InType>
BasePostprocessor<InType>::BasePostprocessor(const BaseMicroserviceConfigs &configs) : Microservice<InType>(configs) {
}