#pragma once

#include <obs-module.h>

#ifdef USE_CNN_OCR
#include <opencv2/opencv.hpp>

// Public API to update averaged frame data from control panel
void update_averaged_frame_viz_data(
	const cv::Mat& shot_averaged_frame,
	const cv::Mat& game_averaged_frame);
#endif

extern struct obs_source_info averaged_frame_viz_source_info;
extern void init_averaged_frame_viz_source_info();