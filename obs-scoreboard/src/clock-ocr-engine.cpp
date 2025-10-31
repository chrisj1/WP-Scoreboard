#include "clock-ocr-engine.h"
#include "averaged-frame-viz-source.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <obs-module.h>
#include <fstream>
#include <sstream>
#include <string>

// ============================================================================
// BayesianShotClockFilter Implementation
// ============================================================================

BayesianShotClockFilter::BayesianShotClockFilter(int fps, int smoothing_frames) 
    : fps(fps), frame_count(0), consecutive_blocked(0), smoothing_frames(smoothing_frames) {
    // Initialize uniform prior for 0-30 seconds
    posterior.resize(31, 1.0f / 31.0f);
    createTransitionMatrix();
    createBlindTransitionMatrix();
}

void BayesianShotClockFilter::createTransitionMatrix() {
    transition_matrix.resize(31, std::vector<float>(31, 0.0f));
    
    // From annotation code: p_tick = min(0.05, (1.0/fps) * 1.5)
    float p_decrement_per_frame = 1.0f / fps;
    float p_tick = std::min(0.05f, p_decrement_per_frame * 1.5f);
    
    for (int current_val = 0; current_val < 31; current_val++) {
        // STATE 0: Shot clock expired (from annotation code)
        if (current_val == 0) {
            transition_matrix[0][0] = 0.88f;
            transition_matrix[0][30] = 0.10f;
            transition_matrix[0][24] = 0.01f;
            transition_matrix[0][14] = 0.005f;
            transition_matrix[0][20] = 0.005f;
        }
        // STATES 1-5: Critical time (runs faster)
        else if (current_val >= 1 && current_val <= 5) {
            if (current_val > 0)
                transition_matrix[current_val][current_val - 1] = p_tick * 1.2f;
            if (current_val >= 2)
                transition_matrix[current_val][current_val - 2] = 0.003f;
            if (current_val >= 3)
                transition_matrix[current_val][current_val - 3] = 0.001f;
            if (current_val < 30)
                transition_matrix[current_val][current_val + 1] = 0.0005f;
            
            transition_matrix[current_val][30] = 0.020f;
            transition_matrix[current_val][24] = 0.002f;
            transition_matrix[current_val][20] = 0.001f;
            transition_matrix[current_val][14] = 0.001f;
            
            // Stay probability = 1.0 - sum of all transitions
            float row_sum = 0.0f;
            for (int j = 0; j < 31; j++)
                row_sum += transition_matrix[current_val][j];
            transition_matrix[current_val][current_val] = 1.0f - row_sum;
        }
        // STATES 6-14: Mid-range
        else if (current_val >= 6 && current_val <= 14) {
            if (current_val > 0)
                transition_matrix[current_val][current_val - 1] = p_tick;
            if (current_val >= 2)
                transition_matrix[current_val][current_val - 2] = 0.004f;
            if (current_val >= 3)
                transition_matrix[current_val][current_val - 3] = 0.002f;
            if (current_val >= 4)
                transition_matrix[current_val][current_val - 4] = 0.0005f;
            if (current_val < 30)
                transition_matrix[current_val][current_val + 1] = 0.0005f;
            
            transition_matrix[current_val][30] = 0.035f;
            transition_matrix[current_val][24] = 0.003f;
            transition_matrix[current_val][20] = 0.002f;
            transition_matrix[current_val][14] = 0.002f;
            
            float row_sum = 0.0f;
            for (int j = 0; j < 31; j++)
                row_sum += transition_matrix[current_val][j];
            transition_matrix[current_val][current_val] = 1.0f - row_sum;
        }
        // STATES 15-24: Early-mid range
        else if (current_val >= 15 && current_val <= 24) {
            if (current_val > 0)
                transition_matrix[current_val][current_val - 1] = p_tick;
            if (current_val >= 2)
                transition_matrix[current_val][current_val - 2] = 0.005f;
            if (current_val >= 3)
                transition_matrix[current_val][current_val - 3] = 0.003f;
            if (current_val >= 4)
                transition_matrix[current_val][current_val - 4] = 0.001f;
            if (current_val >= 5)
                transition_matrix[current_val][current_val - 5] = 0.0005f;
            if (current_val < 30)
                transition_matrix[current_val][current_val + 1] = 0.0005f;
            if (current_val < 29)
                transition_matrix[current_val][current_val + 2] = 0.0002f;
            
            transition_matrix[current_val][30] = 0.050f;
            transition_matrix[current_val][24] = 0.004f;
            transition_matrix[current_val][20] = 0.002f;
            transition_matrix[current_val][14] = 0.002f;
            
            float row_sum = 0.0f;
            for (int j = 0; j < 31; j++)
                row_sum += transition_matrix[current_val][j];
            transition_matrix[current_val][current_val] = 1.0f - row_sum;
        }
        // STATES 25-30: High values (slower at start)
        else {
            if (current_val > 0)
                transition_matrix[current_val][current_val - 1] = p_tick * 0.8f;
            if (current_val >= 2)
                transition_matrix[current_val][current_val - 2] = 0.006f;
            if (current_val >= 3)
                transition_matrix[current_val][current_val - 3] = 0.004f;
            if (current_val >= 4)
                transition_matrix[current_val][current_val - 4] = 0.002f;
            if (current_val >= 5)
                transition_matrix[current_val][current_val - 5] = 0.001f;
            if (current_val < 30)
                transition_matrix[current_val][current_val + 1] = 0.002f;
            if (current_val < 29)
                transition_matrix[current_val][current_val + 2] = 0.001f;
            
            transition_matrix[current_val][30] = 0.040f;
            transition_matrix[current_val][24] = 0.003f;
            transition_matrix[current_val][20] = 0.002f;
            transition_matrix[current_val][14] = 0.001f;
            
            float row_sum = 0.0f;
            for (int j = 0; j < 31; j++)
                row_sum += transition_matrix[current_val][j];
            transition_matrix[current_val][current_val] = 1.0f - row_sum;
        }
    }
    
    // Normalize all rows to ensure they sum to 1.0
    for (int i = 0; i < 31; i++) {
        float row_sum = std::accumulate(transition_matrix[i].begin(), transition_matrix[i].end(), 0.0f);
        if (row_sum > 1e-10f) {
            for (auto& val : transition_matrix[i])
                val /= row_sum;
        }
    }
}

void BayesianShotClockFilter::createBlindTransitionMatrix() {
    blind_transition_matrix.resize(31, std::vector<float>(31, 0.0f));
    
    // From annotation code: assume clock is running during blocked periods
    float base_tick_prob = 1.0f / fps;
    float tick_prob = base_tick_prob * 5.0f;  // Assume running 5x faster when blocked
    
    for (int current_val = 0; current_val < 31; current_val++) {
        // STATE 0: Expired (from annotation code)
        if (current_val == 0) {
            blind_transition_matrix[0][0] = 0.90f;
            blind_transition_matrix[0][30] = 0.05f;
            blind_transition_matrix[0][24] = 0.02f;
            blind_transition_matrix[0][14] = 0.01f;
            blind_transition_matrix[0][20] = 0.01f;
        } else {
            // 85% chance to tick once, 10% to tick twice
            if (current_val > 0)
                blind_transition_matrix[current_val][current_val - 1] = 0.85f;
            if (current_val >= 2)
                blind_transition_matrix[current_val][current_val - 2] = 0.10f;
            
            // Very small reset probabilities (would see unblocking first)
            blind_transition_matrix[current_val][30] = 0.01f;
            blind_transition_matrix[current_val][24] = 0.005f;
            blind_transition_matrix[current_val][14] = 0.005f;
            blind_transition_matrix[current_val][20] = 0.005f;
            
            // Stay = 1.0 - sum
            float row_sum = 0.0f;
            for (int j = 0; j < 31; j++)
                row_sum += blind_transition_matrix[current_val][j];
            blind_transition_matrix[current_val][current_val] = 1.0f - row_sum;
        }
    }
    
    // Normalize all rows
    for (int i = 0; i < 31; i++) {
        float row_sum = std::accumulate(blind_transition_matrix[i].begin(), blind_transition_matrix[i].end(), 0.0f);
        if (row_sum > 1e-10f) {
            for (auto& val : blind_transition_matrix[i])
                val /= row_sum;
        }
    }
}

bool BayesianShotClockFilter::loadTransitionMatrixFromCSV(const std::string& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        blog(LOG_ERROR, "Failed to open transition matrix CSV: %s", csv_path.c_str());
        return false;
    }
    
    std::vector<std::vector<float>> new_matrix(31, std::vector<float>(31, 0.0f));
    std::string line;
    int row = 0;
    
    while (std::getline(file, line) && row < 31) {
        std::stringstream ss(line);
        std::string cell;
        int col = 0;
        
        while (std::getline(ss, cell, ',') && col < 31) {
            try {
                new_matrix[row][col] = std::stof(cell);
            } catch (...) {
                blog(LOG_ERROR, "Invalid value in CSV at row %d, col %d: %s", row, col, cell.c_str());
                return false;
            }
            col++;
        }
        
        if (col != 31) {
            blog(LOG_ERROR, "Invalid number of columns in row %d: expected 31, got %d", row, col);
            return false;
        }
        row++;
    }
    
    if (row != 31) {
        blog(LOG_ERROR, "Invalid number of rows in CSV: expected 31, got %d", row);
        return false;
    }
    
    // Validate that each row sums to approximately 1.0
    for (int i = 0; i < 31; i++) {
        float row_sum = std::accumulate(new_matrix[i].begin(), new_matrix[i].end(), 0.0f);
        if (std::abs(row_sum - 1.0f) > 0.01f) {
            blog(LOG_WARNING, "Row %d sum is %.4f (should be 1.0). Normalizing...", i, row_sum);
            if (row_sum > 1e-10f) {
                for (auto& val : new_matrix[i])
                    val /= row_sum;
            }
        }
    }
    
    transition_matrix = new_matrix;
    blog(LOG_INFO, "Successfully loaded shot clock transition matrix from: %s", csv_path.c_str());
    return true;
}

std::vector<float> BayesianShotClockFilter::applyTransition(
    const std::vector<float>& state,
    const std::vector<std::vector<float>>& matrix) {
    
    std::vector<float> result(31, 0.0f);
    for (int j = 0; j < 31; j++) {
        for (int i = 0; i < 31; i++) {
            result[j] += state[i] * matrix[i][j];
        }
    }
    
    // Normalize
    float sum = std::accumulate(result.begin(), result.end(), 0.0f);
    if (sum > 1e-10f) {
        for (auto& val : result)
            val /= sum;
    }
    
    return result;
}
ClockPrediction BayesianShotClockFilter::update(const std::vector<float>& cnn_probabilities, bool is_blocked) {
    frame_count++;
    
    // Predict step
    std::vector<float> prior;
    if (is_blocked) {
        consecutive_blocked++;
        prior = applyTransition(posterior, blind_transition_matrix);
    } else {
        consecutive_blocked = 0;
        prior = applyTransition(posterior, transition_matrix);
    }
    
    // Update step
    if (is_blocked) {
        posterior = prior;
    } else {
        // Bayesian update: posterior ∝ prior * likelihood
        // CNN probabilities are already averaged over N frames before inference
        for (int i = 0; i < 31; i++) {
            posterior[i] = prior[i] * cnn_probabilities[i];
        }
        
        float sum = std::accumulate(posterior.begin(), posterior.end(), 0.0f);
        if (sum > 1e-10f) {
            for (auto& val : posterior)
                val /= sum;
        }
    }
    
    // Normalize prior so sum is 1
    float prior_sum = std::accumulate(prior.begin(), prior.end(), 0.0f);
    if (prior_sum > 1e-10f) {
        for (auto& p : prior) p /= prior_sum;
    }
    
    // Get best estimate
    int best_val = std::max_element(posterior.begin(), posterior.end()) - posterior.begin();
    float confidence = posterior[best_val];
    
    ClockPrediction result;
    result.value = std::to_string(best_val);
    if (best_val < 10)
        result.value = "0" + result.value;
    result.confidence = confidence;
    result.is_blocked = is_blocked;
    result.is_special = false;
    result.probabilities = posterior;  // Posterior after update
    result.prior = prior;               // Prior before update
    result.cnn_raw = cnn_probabilities; // CNN output (already from averaged frames)
    
    return result;
}

void BayesianShotClockFilter::reset() {
    posterior.assign(31, 1.0f / 31.0f);
    frame_count = 0;
    consecutive_blocked = 0;
}

void BayesianShotClockFilter::resetPrior() {
    posterior.assign(31, 1.0f);
}

// ============================================================================
// BayesianGameClockFilter Implementation
// ============================================================================

BayesianGameClockFilter::BayesianGameClockFilter(int fps, int smoothing_frames)
    : fps(fps), frame_count(0), consecutive_blocked(0), smoothing_frames(smoothing_frames) {
    // Initialize uniform prior for 0-479 seconds (0:00 to 7:59)
    posterior.resize(480, 1.0f / 480.0f);
    createTransitionMatrix();
    createBlindTransitionMatrix();
}

void BayesianGameClockFilter::createTransitionMatrix() {
    transition_matrix.resize(480, std::vector<float>(480, 0.0f));
    
    float p_tick = 1.0f / fps;
    
    for (int current_seconds = 0; current_seconds < 480; current_seconds++) {
        // STATE 0: Game clock expired (from annotation code)
        if (current_seconds == 0) {
            transition_matrix[0][0] = 0.90f;
            transition_matrix[0][479] = 0.01f;  // Reset to 8:00
            transition_matrix[0][419] = 0.01f;  // Reset to 7:00
            transition_matrix[0][359] = 0.01f;  // Reset to 6:00
            transition_matrix[0][299] = 0.01f;  // Reset to 5:00
            for (int t = 0; t < 480; t++) {
                if (t != 0 && t != 479 && t != 419 && t != 359 && t != 299)
                    transition_matrix[0][t] = (1.0f - (0.9f + 4* 0.01f))/(480-5); // Small prob to jump to any other time
            }
        }
        // Normal countdown
        else {
            if (current_seconds > 0)
                transition_matrix[current_seconds][current_seconds - 1] = p_tick;
            
            // Small correction tails
            if (current_seconds >= 2)
                transition_matrix[current_seconds][current_seconds - 2] = 0.001f;
            if (current_seconds >= 3)
                transition_matrix[current_seconds][current_seconds - 3] = 0.0005f;
            
            if (current_seconds < 479)
                transition_matrix[current_seconds][current_seconds + 1] = 0.0002f;
            
            // Period resets (very small probability)
            transition_matrix[current_seconds][479] = 0.0001f;  // 8:00
            transition_matrix[current_seconds][419] = 0.0001f;  // 7:00
            transition_matrix[current_seconds][359] = 0.0001f;  // 6:00
            transition_matrix[current_seconds][299] = 0.0001f;  // 5:00
            
            // Stay = 1.0 - sum
            float row_sum = 0.0f;
            for (int j = 0; j < 480; j++)
                row_sum += transition_matrix[current_seconds][j];
            transition_matrix[current_seconds][current_seconds] = 1.0f - row_sum;
        }
    }
    
    // Normalize all rows to ensure they sum to 1.0
    for (int i = 0; i < 480; i++) {
        float row_sum = std::accumulate(transition_matrix[i].begin(), transition_matrix[i].end(), 0.0f);
        if (row_sum > 1e-10f) {
            for (auto& val : transition_matrix[i])
                val /= row_sum;
        }
    }
}

void BayesianGameClockFilter::createBlindTransitionMatrix() {
    blind_transition_matrix.resize(480, std::vector<float>(480, 0.0f));
    
    // From annotation code: simple countdown during blocked periods
    float p_tick = 1.0f / fps;
    
    for (int current_seconds = 0; current_seconds < 480; current_seconds++) {
        if (current_seconds == 0) {
            blind_transition_matrix[0][0] = 1.0f;  // Stay at 0 when blocked
        } else {
            // Simple tick countdown
            blind_transition_matrix[current_seconds][current_seconds - 1] = p_tick;
            blind_transition_matrix[current_seconds][current_seconds] = 1.0f - p_tick;
        }
    }
}

bool BayesianGameClockFilter::loadTransitionMatrixFromCSV(const std::string& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        blog(LOG_ERROR, "Failed to open transition matrix CSV: %s", csv_path.c_str());
        return false;
    }
    
    std::vector<std::vector<float>> new_matrix(480, std::vector<float>(480, 0.0f));
    std::string line;
    int row = 0;
    
    while (std::getline(file, line) && row < 480) {
        std::stringstream ss(line);
        std::string cell;
        int col = 0;
        
        while (std::getline(ss, cell, ',') && col < 480) {
            try {
                new_matrix[row][col] = std::stof(cell);
            } catch (...) {
                blog(LOG_ERROR, "Invalid value in CSV at row %d, col %d: %s", row, col, cell.c_str());
                return false;
            }
            col++;
        }
        
        if (col != 480) {
            blog(LOG_ERROR, "Invalid number of columns in row %d: expected 480, got %d", row, col);
            return false;
        }
        row++;
    }
    
    if (row != 480) {
        blog(LOG_ERROR, "Invalid number of rows in CSV: expected 480, got %d", row);
        return false;
    }
    
    // Validate that each row sums to approximately 1.0
    for (int i = 0; i < 480; i++) {
        float row_sum = std::accumulate(new_matrix[i].begin(), new_matrix[i].end(), 0.0f);
        if (std::abs(row_sum - 1.0f) > 0.01f) {
            blog(LOG_WARNING, "Row %d sum is %.4f (should be 1.0). Normalizing...", i, row_sum);
            if (row_sum > 1e-10f) {
                for (auto& val : new_matrix[i])
                    val /= row_sum;
            }
        }
    }
    
    transition_matrix = new_matrix;
    blog(LOG_INFO, "Successfully loaded game clock transition matrix from: %s", csv_path.c_str());
    return true;
}

std::vector<float> BayesianGameClockFilter::applyTransition(
    const std::vector<float>& state,
    const std::vector<std::vector<float>>& matrix) {
    
    std::vector<float> result(480, 0.0f);
    for (int j = 0; j < 480; j++) {
        for (int i = 0; i < 480; i++) {
            result[j] += state[i] * matrix[i][j];
        }
    }
    
    // Normalize
    float sum = std::accumulate(result.begin(), result.end(), 0.0f);
    if (sum > 1e-10f) {
        for (auto& val : result)
            val /= sum;
    }
    
    return result;
}

ClockPrediction BayesianGameClockFilter::update(const std::vector<float>& cnn_probabilities, bool is_blocked) {
    frame_count++;
    
    // Predict step
    std::vector<float> prior;
    if (is_blocked) {
        consecutive_blocked++;
        prior = applyTransition(posterior, blind_transition_matrix);
    } else {
        consecutive_blocked = 0;
        prior = applyTransition(posterior, transition_matrix);
    }
    
    // Update step
    if (is_blocked) {
        posterior = prior;
    } else {
        // Bayesian update: posterior ∝ prior * likelihood
        // CNN probabilities are already averaged over N frames before inference
        for (int i = 0; i < 480; i++) {
            posterior[i] = prior[i] * cnn_probabilities[i];
        }
        
        float sum = std::accumulate(posterior.begin(), posterior.end(), 0.0f);
        if (sum > 1e-10f) {
            for (auto& val : posterior)
                val /= sum;
        }
    }
    
    // Normalize prior so sum is 1
    float prior_sum = std::accumulate(prior.begin(), prior.end(), 0.0f);
    if (prior_sum > 1e-10f) {
        for (auto& p : prior) p /= prior_sum;
    }
    
    // Get best estimate
    int best_seconds = std::max_element(posterior.begin(), posterior.end()) - posterior.begin();
    float confidence = posterior[best_seconds];
    
    int minutes = best_seconds / 60;
    int seconds = best_seconds % 60;
    
    ClockPrediction result;
    result.value = std::to_string(minutes) + ":" + 
                   (seconds < 10 ? "0" : "") + std::to_string(seconds);
    result.confidence = confidence;
    result.is_blocked = is_blocked;
    result.is_special = (best_seconds == 0);
    result.probabilities = posterior;  // Posterior after update
    result.prior = prior;               // Prior before update
    result.cnn_raw = cnn_probabilities; // CNN output (already from averaged frames)
    
    return result;
}

void BayesianGameClockFilter::reset() {
    posterior.assign(480, 1.0f / 480.0f);
    frame_count = 0;
    consecutive_blocked = 0;
}

void BayesianGameClockFilter::resetPrior() {
    posterior.assign(480, 1.0f);
}

// ============================================================================
// ShotClockCNN Implementation
// ============================================================================

ShotClockCNN::ShotClockCNN() 
    : model_loaded(false), device(torch::kCPU) {
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        blog(LOG_INFO, "Shot Clock CNN: CUDA available, using GPU");
    } else {
        blog(LOG_INFO, "Shot Clock CNN: Using CPU");
    }
}

ShotClockCNN::~ShotClockCNN() {
}

bool ShotClockCNN::loadModel(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path, device);
        model.eval();
        model_loaded = true;
        blog(LOG_INFO, "Shot Clock CNN: Model loaded from %s", model_path.c_str());
        return true;
    } catch (const c10::Error& e) {
        blog(LOG_ERROR, "Shot Clock CNN: Failed to load model: %s", e.what());
        model_loaded = false;
        return false;
    }
}

torch::Tensor ShotClockCNN::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;
    
    // Handle different input types
    if (image.type() == CV_32F) {
        // Already float32, assume it's in [0, 255] range from averaging
        processed = image.clone();
    } else {
        // Convert to grayscale if needed, then to float32
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }
        gray.convertTo(processed, CV_32F);
    }
    
    // Resize to 64x32
    cv::Mat resized;
    cv::resize(processed, resized, cv::Size(64, 32));
    
    // Normalize to [0, 1] range
    if (resized.type() != CV_32F) {
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    } else {
        // Already float32, just scale to [0,1] if needed
        double minVal, maxVal;
        cv::minMaxLoc(resized, &minVal, &maxVal);
        if (maxVal > 1.1) {  // Assume [0,255] range
            resized /= 255.0;
        }
        // Otherwise assume already in [0,1] range
    }
    
    // Convert to tensor: shape [1, 1, 32, 64]
    torch::Tensor tensor = torch::from_blob(resized.data, {1, 1, 32, 64}, torch::kFloat32);
    tensor = tensor.clone(); // Ensure ownership
    tensor = tensor.to(device);
    
    return tensor;
}

ClockPrediction ShotClockCNN::predict(const cv::Mat& roi_image) {
    ClockPrediction result;
    result.value = "00";
    result.confidence = 0.0f;
    result.is_blocked = false;
    result.is_special = false;
    result.probabilities.resize(31, 1.0f / 31.0f);
    
    if (!model_loaded || roi_image.empty()) {
        return result;
    }
    
    try {
        torch::Tensor input = preprocessImage(roi_image);
        
        // Forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        auto outputs = model.forward(inputs).toTuple();
        
        // Parse outputs: digit1, digit2, blocked, inconclusive
        torch::Tensor digit1_out = outputs->elements()[0].toTensor();
        torch::Tensor digit2_out = outputs->elements()[1].toTensor();
        torch::Tensor blocked_out = outputs->elements()[2].toTensor();
        torch::Tensor inconclusive_out = outputs->elements()[3].toTensor();
        
        // Apply softmax to digit outputs
        digit1_out = torch::softmax(digit1_out, 1);
        digit2_out = torch::softmax(digit2_out, 1);
        
        // Move to CPU for processing
        digit1_out = digit1_out.to(torch::kCPU);
        digit2_out = digit2_out.to(torch::kCPU);
        blocked_out = blocked_out.to(torch::kCPU);
        inconclusive_out = inconclusive_out.to(torch::kCPU);
        
        float blocked_prob = blocked_out.item<float>();
        float inconclusive_prob = inconclusive_out.item<float>();
        
        // Calculate probabilities for each value 0-30
        std::vector<float> probabilities(31, 0.0);
        auto digit1_acc = digit1_out.accessor<float, 2>();
        auto digit2_acc = digit2_out.accessor<float, 2>();
        
        for (int value = 0; value <= 30; value++) {
            int d1 = value / 10;
            int d2 = value % 10;
            probabilities[value] = digit1_acc[0][d1] * digit2_acc[0][d2];
        }
        
        // Normalize probabilities to sum to 1 (proper probability distribution)
        float prob_sum = 0.0f;
        for (float p : probabilities) {
            prob_sum += p;
        }
        if (prob_sum > 1e-10f) {
            for (float& p : probabilities) {
                p /= prob_sum;
            }
        }
        
        // Find best prediction
        int best_value = std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
        float best_confidence = probabilities[best_value];
        
        result.probabilities = probabilities;
        
        // Check special cases
        if (blocked_prob > 0.5f) {
            result.value = "BLOCKED";
            result.confidence = blocked_prob;
            result.is_blocked = true;
            return result;
        }
        
        if (inconclusive_prob > 0.5f) {
            result.value = "??";
            result.confidence = inconclusive_prob;
            result.is_special = true;
            return result;
        }
        
        result.value = std::to_string(best_value);
        if (best_value < 10)
            result.value = "0" + result.value;
        result.confidence = best_confidence;
        
    } catch (const c10::Error& e) {
        blog(LOG_ERROR, "Shot Clock CNN prediction error: %s", e.what());
    }
    
    return result;
}

// ============================================================================
// GameClockCNN Implementation
// ============================================================================

GameClockCNN::GameClockCNN()
    : model_loaded(false), device(torch::kCPU) {
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        blog(LOG_INFO, "Game Clock CNN: CUDA available, using GPU");
    } else {
        blog(LOG_INFO, "Game Clock CNN: Using CPU");
    }
}

GameClockCNN::~GameClockCNN() {
}

bool GameClockCNN::loadModel(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path, device);
        model.eval();
        model_loaded = true;
        blog(LOG_INFO, "Game Clock CNN: Model loaded from %s", model_path.c_str());
        return true;
    } catch (const c10::Error& e) {
        blog(LOG_ERROR, "Game Clock CNN: Failed to load model: %s", e.what());
        model_loaded = false;
        return false;
    }
}

torch::Tensor GameClockCNN::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;
    
    // Handle different input types
    if (image.type() == CV_32F) {
        // Already float32, assume it's in [0, 255] range from averaging
        processed = image.clone();
    } else {
        // Convert to grayscale if needed, then to float32
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }
        gray.convertTo(processed, CV_32F);
    }
    
    // Resize to 64x32
    cv::Mat resized;
    cv::resize(processed, resized, cv::Size(64, 32));
    
    // Normalize to [0, 1] range
    if (resized.type() != CV_32F) {
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    } else {
        // Already float32, just scale to [0,1] if needed
        double minVal, maxVal;
        cv::minMaxLoc(resized, &minVal, &maxVal);
        if (maxVal > 1.1) {  // Assume [0,255] range
            resized /= 255.0;
        }
        // Otherwise assume already in [0,1] range
    }
    
    // Convert to tensor: shape [1, 1, 32, 64]
    torch::Tensor tensor = torch::from_blob(resized.data, {1, 1, 32, 64}, torch::kFloat32);
    tensor = tensor.clone();
    tensor = tensor.to(device);
    
    return tensor;
}

ClockPrediction GameClockCNN::predict(const cv::Mat& roi_image) {
    ClockPrediction result;
    result.value = "0:00";
    result.confidence = 0.0f;
    result.is_blocked = false;
    result.is_special = false;
    result.probabilities.resize(480, 1.0f / 480.0f);
    
    if (!model_loaded || roi_image.empty()) {
        return result;
    }
    
    try {
        torch::Tensor input = preprocessImage(roi_image);
        
        // Forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        auto outputs = model.forward(inputs).toTuple();
        
        // Parse outputs: minute, sec1, sec2, blocked, blank
        torch::Tensor minute_out = outputs->elements()[0].toTensor();
        torch::Tensor sec1_out = outputs->elements()[1].toTensor();
        torch::Tensor sec2_out = outputs->elements()[2].toTensor();
        torch::Tensor blocked_out = outputs->elements()[3].toTensor();
        torch::Tensor blank_out = outputs->elements()[4].toTensor();
        
        // Apply softmax
        minute_out = torch::softmax(minute_out, 1);
        sec1_out = torch::softmax(sec1_out, 1);
        sec2_out = torch::softmax(sec2_out, 1);
        
        // Move to CPU
        minute_out = minute_out.to(torch::kCPU);
        sec1_out = sec1_out.to(torch::kCPU);
        sec2_out = sec2_out.to(torch::kCPU);
        blocked_out = blocked_out.to(torch::kCPU);
        blank_out = blank_out.to(torch::kCPU);
        
        float blocked_prob = blocked_out.item<float>();
        float blank_prob = blank_out.item<float>();
        
        // Calculate probabilities for each time (0-479 seconds)
        std::vector<float> probabilities(480, 0.0f);
        auto minute_acc = minute_out.accessor<float, 2>();
        auto sec1_acc = sec1_out.accessor<float, 2>();
        auto sec2_acc = sec2_out.accessor<float, 2>();
        
        for (int total_seconds = 0; total_seconds < 480; total_seconds++) {
            int minute = total_seconds / 60;
            int seconds = total_seconds % 60;
            int s1 = seconds / 10;
            int s2 = seconds % 10;
            
            if (minute < 8 && s1 < 6) {
                probabilities[total_seconds] = minute_acc[0][minute] * 
                                                sec1_acc[0][s1] * 
                                                sec2_acc[0][s2];
            }
        }
        
        // Normalize probabilities to sum to 1 (proper probability distribution)
        float prob_sum = 0.0f;
        for (float p : probabilities) {
            prob_sum += p;
        }
        if (prob_sum > 1e-10f) {
            for (float& p : probabilities) {
                p /= prob_sum;
            }
        }
        
        // Find best prediction
        int best_seconds = std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
        float best_confidence = probabilities[best_seconds];
        
        result.probabilities = probabilities;
        
        // Check special cases
        if (blocked_prob > 0.5f) {
            result.value = "BLOCKED";
            result.confidence = blocked_prob;
            result.is_blocked = true;
            return result;
        }
        
        if (blank_prob > 0.5f) {
            result.value = "0:00";
            result.confidence = blank_prob;
            result.is_special = true;
            return result;
        }
        
        int minutes = best_seconds / 60;
        int seconds = best_seconds % 60;
        result.value = std::to_string(minutes) + ":" + 
                       (seconds < 10 ? "0" : "") + std::to_string(seconds);
        result.confidence = best_confidence;
        
    } catch (const c10::Error& e) {
        blog(LOG_ERROR, "Game Clock CNN prediction error: %s", e.what());
    }
    
    return result;
}

// ============================================================================
// ClockOCREngine Implementation
// ============================================================================

ClockOCREngine::ClockOCREngine()
    : use_bayesian(true), fps(30),
      last_shot_clock("00"), last_game_clock("0:00"),
      last_shot_clock_confidence(0.0f), last_game_clock_confidence(0.0f),
      max_buffered_frames(5) {
    
    shot_clock_cnn = std::make_unique<ShotClockCNN>();
    game_clock_cnn = std::make_unique<GameClockCNN>();
    // Default to averaging over 5 consecutive frames for smoother results
    shot_clock_filter = std::make_unique<BayesianShotClockFilter>(fps, 5);
    game_clock_filter = std::make_unique<BayesianGameClockFilter>(fps, 5);
}

ClockOCREngine::~ClockOCREngine() {
    closeCamera();
}

bool ClockOCREngine::loadShotClockModel(const std::string& model_path) {
    return shot_clock_cnn->loadModel(model_path);
}

bool ClockOCREngine::loadGameClockModel(const std::string& model_path) {
    return game_clock_cnn->loadModel(model_path);
}

bool ClockOCREngine::openCamera(int camera_index) {
    std::lock_guard<std::mutex> lock(frame_mutex);
    
    if (camera.isOpened()) {
        camera.release();
    }
    
    camera.open(camera_index);
    if (!camera.isOpened()) {
        blog(LOG_ERROR, "ClockOCREngine: Failed to open camera %d", camera_index);
        return false;
    }
    
    // Try to get FPS
    double detected_fps = camera.get(cv::CAP_PROP_FPS);
    if (detected_fps > 0) {
        fps = static_cast<int>(detected_fps);
        blog(LOG_INFO, "ClockOCREngine: Camera opened at %d fps", fps);
        
        // Update filters with new FPS
        shot_clock_filter = std::make_unique<BayesianShotClockFilter>(fps);
        game_clock_filter = std::make_unique<BayesianGameClockFilter>(fps);
    }
    
    return true;
}

bool ClockOCREngine::openCamera(const std::string& camera_url) {
    std::lock_guard<std::mutex> lock(frame_mutex);
    
    if (camera.isOpened()) {
        camera.release();
    }
    
    camera.open(camera_url);
    if (!camera.isOpened()) {
        blog(LOG_ERROR, "ClockOCREngine: Failed to open camera URL: %s", camera_url.c_str());
        return false;
    }
    
    double detected_fps = camera.get(cv::CAP_PROP_FPS);
    if (detected_fps > 0) {
        fps = static_cast<int>(detected_fps);
        blog(LOG_INFO, "ClockOCREngine: Camera stream opened at %d fps", fps);
        
        shot_clock_filter = std::make_unique<BayesianShotClockFilter>(fps);
        game_clock_filter = std::make_unique<BayesianGameClockFilter>(fps);
    }
    
    return true;
}

void ClockOCREngine::closeCamera() {
    std::lock_guard<std::mutex> lock(frame_mutex);
    if (camera.isOpened()) {
        camera.release();
    }
}

bool ClockOCREngine::isCameraOpen() const {
    return camera.isOpened();
}

void ClockOCREngine::setShotClockROI(int x, int y, int width, int height) {
    shot_clock_roi = cv::Rect(x, y, width, height);
    blog(LOG_INFO, "Shot Clock ROI set: (%d, %d, %d, %d)", x, y, width, height);
}

void ClockOCREngine::setGameClockROI(int x, int y, int width, int height) {
    game_clock_roi = cv::Rect(x, y, width, height);
    blog(LOG_INFO, "Game Clock ROI set: (%d, %d, %d, %d)", x, y, width, height);
}

bool ClockOCREngine::captureFrame() {
    std::lock_guard<std::mutex> lock(frame_mutex);
    
    if (!camera.isOpened()) {
        return false;
    }
    
    bool ret = camera.read(last_frame);
    if (!ret || last_frame.empty()) {
        blog(LOG_WARNING, "ClockOCREngine: Failed to capture frame");
        return false;
    }
    
    return true;
}

bool ClockOCREngine::validateROI(const cv::Rect& roi) const {
    if (last_frame.empty()) {
        return false;
    }
    
    return (roi.x >= 0 && roi.y >= 0 && 
            roi.x + roi.width <= last_frame.cols &&
            roi.y + roi.height <= last_frame.rows &&
            roi.width > 0 && roi.height > 0);
}

void ClockOCREngine::setFrame(const cv::Mat& frame) {
    // Legacy method: sets both buffers for backward compatibility
    setShotClockFrame(frame);
    setGameClockFrame(frame);
}

void ClockOCREngine::setShotClockFrame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex);
    last_frame = frame.clone();
    
    // Add to shot clock frame buffer for temporal averaging
    shot_clock_frames.push_back(frame.clone());
    
    // Keep only the most recent frames
    if (shot_clock_frames.size() > static_cast<size_t>(max_buffered_frames)) {
        shot_clock_frames.erase(shot_clock_frames.begin());
    }
}

void ClockOCREngine::setGameClockFrame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex);
    
    // Add to game clock frame buffer for temporal averaging
    game_clock_frames.push_back(frame.clone());
    
    // Keep only the most recent frames
    if (game_clock_frames.size() > static_cast<size_t>(max_buffered_frames)) {
        game_clock_frames.erase(game_clock_frames.begin());
    }
}

ClockPrediction ClockOCREngine::predictShotClock() {
    ClockPrediction result;
    result.value = "00";
    result.confidence = 0.0f;
    result.is_blocked = false;
    result.is_special = false;
    result.is_fresh_cnn = false;
    
    if (last_frame.empty() || !shot_clock_cnn->isLoaded()) {
        return result;
    }
    
    if (!validateROI(shot_clock_roi)) {
        blog(LOG_WARNING, "Invalid shot clock ROI");
        return result;
    }
    
    // Only run CNN when we have accumulated enough frames
    bool should_run_cnn = false;
    cv::Mat averaged_roi;
    
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        
        // Check if we have enough frames to process a batch
        if (shot_clock_frames.size() >= static_cast<size_t>(max_buffered_frames)) {
            should_run_cnn = true;
            
            // Crop ROI from each frame FIRST, then average the cropped regions
            cv::Mat roi_first = shot_clock_frames[0](shot_clock_roi);
            cv::Mat sum;
            roi_first.convertTo(sum, CV_32F);
            
            for (size_t i = 1; i < shot_clock_frames.size(); i++) {
                cv::Mat roi_crop = shot_clock_frames[i](shot_clock_roi);
                cv::Mat temp;
                roi_crop.convertTo(temp, CV_32F);
                cv::add(sum, temp, sum);
            }
            
            // Divide by number of frames and KEEP as float32 for CNN
            sum /= static_cast<float>(shot_clock_frames.size());
            averaged_roi = sum;  // Keep float precision for CNN processing
            
            // Store the averaged frame for visualization (convert to uint8 for display)
            cv::Mat display_frame;
            averaged_roi.convertTo(display_frame, CV_8U);
            last_shot_clock_averaged_frame = display_frame.clone();
            blog(LOG_INFO, "Stored shot clock averaged frame: %dx%d", display_frame.cols, display_frame.rows);
            
            // Clear the buffer to start collecting next batch
            shot_clock_frames.clear();
        }
    }
    
    // If we don't have enough frames yet, return the last cached prediction
    if (!should_run_cnn) {
        result.value = last_shot_clock;
        result.confidence = last_shot_clock_confidence;
        result.is_fresh_cnn = false;  // This is cached data
        return result;
    }
    
    // Use the averaged ROI directly (already cropped before averaging)
    cv::Mat roi = averaged_roi;
    
    // Get CNN prediction on the averaged frame (runs once per N frames)
    ClockPrediction cnn_result = shot_clock_cnn->predict(roi);
    
    // Apply Bayesian filtering if enabled
    if (use_bayesian) {
        result = shot_clock_filter->update(cnn_result.probabilities, cnn_result.is_blocked);
    } else {
        result = cnn_result;
    }
    
    // Mark as fresh CNN data
    result.is_fresh_cnn = true;
    
    // Cache result
    last_shot_clock = result.value;
    last_shot_clock_confidence = result.confidence;
    
    return result;
}

ClockPrediction ClockOCREngine::predictGameClock() {
    ClockPrediction result;
    result.value = "0:00";
    result.confidence = 0.0f;
    result.is_blocked = false;
    result.is_special = false;
    result.is_fresh_cnn = false;
    
    if (last_frame.empty() || !game_clock_cnn->isLoaded()) {
        return result;
    }
    
    if (!validateROI(game_clock_roi)) {
        blog(LOG_WARNING, "Invalid game clock ROI");
        return result;
    }
    
    // Only run CNN when we have accumulated enough frames
    bool should_run_cnn = false;
    cv::Mat averaged_roi;
    
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        
        // Check if we have enough frames to process a batch
        if (game_clock_frames.size() >= static_cast<size_t>(max_buffered_frames)) {
            should_run_cnn = true;
            
            // Crop ROI from each frame FIRST, then average the cropped regions
            cv::Mat roi_first = game_clock_frames[0](game_clock_roi);
            cv::Mat sum;
            roi_first.convertTo(sum, CV_32F);
            
            for (size_t i = 1; i < game_clock_frames.size(); i++) {
                cv::Mat roi_crop = game_clock_frames[i](game_clock_roi);
                cv::Mat temp;
                roi_crop.convertTo(temp, CV_32F);
                cv::add(sum, temp, sum);
            }
            
            // Divide by number of frames and KEEP as float32 for CNN
            sum /= static_cast<float>(game_clock_frames.size());
            averaged_roi = sum;  // Keep float precision for CNN processing
            
            // Store the averaged frame for visualization (convert to uint8 for display)
            cv::Mat display_frame;
            averaged_roi.convertTo(display_frame, CV_8U);
            last_game_clock_averaged_frame = display_frame.clone();
            blog(LOG_INFO, "Stored game clock averaged frame: %dx%d", display_frame.cols, display_frame.rows);
            
            // Clear the buffer to start collecting next batch
            game_clock_frames.clear();
        }
    }
    
    // If we don't have enough frames yet, return the last cached prediction
    if (!should_run_cnn) {
        result.value = last_game_clock;
        result.confidence = last_game_clock_confidence;
        result.is_fresh_cnn = false;  // This is cached data
        return result;
    }
    
    // Use the averaged ROI directly (already cropped before averaging)
    cv::Mat roi = averaged_roi;
    
    // Get CNN prediction on the averaged frame (runs once per N frames)
    ClockPrediction cnn_result = game_clock_cnn->predict(roi);
    
    // Apply Bayesian filtering if enabled
    if (use_bayesian) {
        result = game_clock_filter->update(cnn_result.probabilities, cnn_result.is_blocked);
    } else {
        result = cnn_result;
    }
    
    // Mark as fresh CNN data
    result.is_fresh_cnn = true;
    
    // Cache result
    last_game_clock = result.value;
    last_game_clock_confidence = result.confidence;
    
    return result;
}

cv::Mat ClockOCREngine::getShotClockROIImage() const {
    if (last_frame.empty() || !validateROI(shot_clock_roi)) {
        return cv::Mat();
    }
    return last_frame(shot_clock_roi).clone();
}

cv::Mat ClockOCREngine::getGameClockROIImage() const {
    if (last_frame.empty() || !validateROI(game_clock_roi)) {
        return cv::Mat();
    }
    return last_frame(game_clock_roi).clone();
}

cv::Mat ClockOCREngine::getShotClockAveragedFrame() const {
    std::lock_guard<std::mutex> lock(frame_mutex);
    blog(LOG_INFO, "getShotClockAveragedFrame: frame size %dx%d, empty: %s", 
         last_shot_clock_averaged_frame.cols, last_shot_clock_averaged_frame.rows,
         last_shot_clock_averaged_frame.empty() ? "true" : "false");
    return last_shot_clock_averaged_frame.clone();
}

cv::Mat ClockOCREngine::getGameClockAveragedFrame() const {
    std::lock_guard<std::mutex> lock(frame_mutex);
    blog(LOG_INFO, "getGameClockAveragedFrame: frame size %dx%d, empty: %s", 
         last_game_clock_averaged_frame.cols, last_game_clock_averaged_frame.rows,
         last_game_clock_averaged_frame.empty() ? "true" : "false");
    return last_game_clock_averaged_frame.clone();
}

void ClockOCREngine::setSmoothingFrames(int frames) {
    if (frames < 1) frames = 1;
    if (frames > 10) frames = 10;  // Reasonable upper limit
    
    std::lock_guard<std::mutex> lock(frame_mutex);
    max_buffered_frames = frames;
    
    // Trim both frame buffers if they're now too large
    while (shot_clock_frames.size() > static_cast<size_t>(max_buffered_frames)) {
        shot_clock_frames.erase(shot_clock_frames.begin());
    }
    while (game_clock_frames.size() > static_cast<size_t>(max_buffered_frames)) {
        game_clock_frames.erase(game_clock_frames.begin());
    }
    
    shot_clock_filter->setSmoothingFrames(frames);
    game_clock_filter->setSmoothingFrames(frames);
}

int ClockOCREngine::getSmoothingFrames() const {
    // Both filters should have same value, return shot clock's
    return shot_clock_filter->getSmoothingFrames();
}

bool ClockOCREngine::loadShotClockTransitionMatrix(const std::string& csv_path) {
    return shot_clock_filter->loadTransitionMatrixFromCSV(csv_path);
}

bool ClockOCREngine::loadGameClockTransitionMatrix(const std::string& csv_path) {
    return game_clock_filter->loadTransitionMatrixFromCSV(csv_path);
}

void ClockOCREngine::resetFilters() {
    shot_clock_filter->reset();
    game_clock_filter->reset();
}

void ClockOCREngine::resetPriors() {
    if (shot_clock_filter) shot_clock_filter->resetPrior();
    if (game_clock_filter) game_clock_filter->resetPrior();
}
