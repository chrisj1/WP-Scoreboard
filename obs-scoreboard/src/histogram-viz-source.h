#pragma once

#include <obs-module.h>

#ifdef USE_CNN_OCR
#include "clock-ocr-engine.h"

// Public API to update histogram data from control panel
void update_histogram_viz_data(
	const ClockPrediction& shot_pred,
	const ClockPrediction& game_pred);
#endif

extern struct obs_source_info histogram_viz_source_info;
extern void init_histogram_viz_source_info();
