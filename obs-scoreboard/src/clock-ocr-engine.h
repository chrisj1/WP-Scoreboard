#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <deque>

// Forward declarations
struct ClockPrediction {
    std::string value;
    float confidence;
    bool is_blocked;
    bool is_special;  // For game clock: blank; for shot clock: inconclusive
    bool is_fresh_cnn;  // True if this prediction came from a new CNN inference
    std::vector<float> probabilities;  // Posterior distribution
    std::vector<float> prior;          // Prior distribution (before CNN update)
    std::vector<float> cnn_raw;        // Raw CNN probabilities
};

// Bayesian filter for shot clock (0-30 seconds)
class BayesianShotClockFilter {
public:
    BayesianShotClockFilter(int fps = 30, int smoothing_frames = 1);
    
    ClockPrediction update(const std::vector<float>& cnn_probabilities, bool is_blocked);
    void reset();
    void resetPrior();
    void setSmoothingFrames(int frames) { smoothing_frames = frames; }
    int getSmoothingFrames() const { return smoothing_frames; }
    
    // Load transition matrix from CSV file
    bool loadTransitionMatrixFromCSV(const std::string& csv_path);
    
private:
    std::vector<std::vector<float>> transition_matrix;
    std::vector<std::vector<float>> blind_transition_matrix;
    std::vector<float> posterior;
    int fps;
    int frame_count;
    int consecutive_blocked;
    int smoothing_frames;  // Number of frames to average before CNN inference
    
    void createTransitionMatrix();
    void createBlindTransitionMatrix();
    std::vector<float> applyTransition(const std::vector<float>& state, 
                                       const std::vector<std::vector<float>>& matrix);
};

// Bayesian filter for game clock (0:00 to 7:59 = 0-479 seconds)
class BayesianGameClockFilter {
public:
    BayesianGameClockFilter(int fps = 30, int smoothing_frames = 1);
    
    ClockPrediction update(const std::vector<float>& cnn_probabilities, bool is_blocked);
    void reset();
    void resetPrior();
    void setSmoothingFrames(int frames) { smoothing_frames = frames; }
    int getSmoothingFrames() const { return smoothing_frames; }
    
    // Load transition matrix from CSV file
    bool loadTransitionMatrixFromCSV(const std::string& csv_path);
    
private:
    std::vector<std::vector<float>> transition_matrix;
    std::vector<std::vector<float>> blind_transition_matrix;
    std::vector<float> posterior;
    int fps;
    int frame_count;
    int consecutive_blocked;
    int smoothing_frames;  // Number of frames to average before CNN inference
    
    void createTransitionMatrix();
    void createBlindTransitionMatrix();
    std::vector<float> applyTransition(const std::vector<float>& state,
                                       const std::vector<std::vector<float>>& matrix);
};

// CNN model wrapper for shot clock (0-30)
class ShotClockCNN {
public:
    ShotClockCNN();
    ~ShotClockCNN();
    
    bool loadModel(const std::string& model_path);
    ClockPrediction predict(const cv::Mat& roi_image);
    bool isLoaded() const { return model_loaded; }
    
private:
    torch::jit::script::Module model;
    bool model_loaded;
    torch::Device device;
    
    torch::Tensor preprocessImage(const cv::Mat& image);
};

// CNN model wrapper for game clock (0:00 to 7:59)
class GameClockCNN {
public:
    GameClockCNN();
    ~GameClockCNN();
    
    bool loadModel(const std::string& model_path);
    ClockPrediction predict(const cv::Mat& roi_image);
    bool isLoaded() const { return model_loaded; }
    
private:
    torch::jit::script::Module model;
    bool model_loaded;
    torch::Device device;
    
    torch::Tensor preprocessImage(const cv::Mat& image);
};

// Main OCR engine that handles camera input and clock detection
class ClockOCREngine {
public:
    ClockOCREngine();
    ~ClockOCREngine();
    
    // Model loading
    bool loadShotClockModel(const std::string& model_path);
    bool loadGameClockModel(const std::string& model_path);
    
    // Camera input
    bool openCamera(int camera_index = 0);
    bool openCamera(const std::string& camera_url);
    void closeCamera();
    bool isCameraOpen() const;
    
    // ROI configuration
    void setShotClockROI(int x, int y, int width, int height);
    void setGameClockROI(int x, int y, int width, int height);
    cv::Rect getShotClockROI() const { return shot_clock_roi; }
    cv::Rect getGameClockROI() const { return game_clock_roi; }
    
    // Frame processing
    bool captureFrame();
    cv::Mat getLastFrame() const { return last_frame.clone(); }
    void setFrame(const cv::Mat& frame); // Legacy: sets both buffers to same frame
    void setShotClockFrame(const cv::Mat& frame); // Set shot clock frame for averaging
    void setGameClockFrame(const cv::Mat& frame); // Set game clock frame for averaging
    
    // Get averaged frames for visualization
    cv::Mat getShotClockAveragedFrame() const;
    cv::Mat getGameClockAveragedFrame() const;
    
    // Clock prediction
    ClockPrediction predictShotClock();
    ClockPrediction predictGameClock();
    
    // Get ROI images for visualization
    cv::Mat getShotClockROIImage() const;
    cv::Mat getGameClockROIImage() const;
    
    // Enable/disable Bayesian filtering
    void enableBayesianFiltering(bool enable) { use_bayesian = enable; }
    bool isBayesianEnabled() const { return use_bayesian; }
    
    // Set/get frame smoothing
    void setSmoothingFrames(int frames);
    int getSmoothingFrames() const;
    
    // Load transition matrices from CSV
    bool loadShotClockTransitionMatrix(const std::string& csv_path);
    bool loadGameClockTransitionMatrix(const std::string& csv_path);
    
    // Reset filters
    void resetFilters();
    void resetPriors();
    
    // Get current predictions (cached)
    std::string getLastShotClockValue() const { return last_shot_clock; }
    std::string getLastGameClockValue() const { return last_game_clock; }
    float getLastShotClockConfidence() const { return last_shot_clock_confidence; }
    float getLastGameClockConfidence() const { return last_game_clock_confidence; }
    
private:
    // Models
    std::unique_ptr<ShotClockCNN> shot_clock_cnn;
    std::unique_ptr<GameClockCNN> game_clock_cnn;
    
    // Bayesian filters
    std::unique_ptr<BayesianShotClockFilter> shot_clock_filter;
    std::unique_ptr<BayesianGameClockFilter> game_clock_filter;
    bool use_bayesian;
    
    // Camera
    cv::VideoCapture camera;
    cv::Mat last_frame;
    mutable std::mutex frame_mutex;
    
    // Frame buffering for temporal averaging (separate buffers for each clock)
    std::vector<cv::Mat> shot_clock_frames;
    std::vector<cv::Mat> game_clock_frames;
    int max_buffered_frames;
    
    // ROI regions
    cv::Rect shot_clock_roi;
    cv::Rect game_clock_roi;
    
    // Cached predictions
    std::string last_shot_clock;
    std::string last_game_clock;
    float last_shot_clock_confidence;
    float last_game_clock_confidence;
    
    // Last averaged frames for visualization
    cv::Mat last_shot_clock_averaged_frame;
    cv::Mat last_game_clock_averaged_frame;
    
    // Frame rate detection
    int fps;
    
    // Helper methods
    bool validateROI(const cv::Rect& roi) const;
};
