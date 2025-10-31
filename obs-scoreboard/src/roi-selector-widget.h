#pragma once

#include <QWidget>
#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QSpinBox>
#include <QTimer>
#include <QMouseEvent>
#include <QPainter>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QMessageBox>

#ifdef USE_CNN_OCR
#include <opencv2/opencv.hpp>

struct ROI {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    
    bool isValid() const {
        return width > 0 && height > 0;
    }
    
    cv::Rect toRect() const {
        return cv::Rect(x, y, width, height);
    }
};

// Widget that displays camera frame and allows ROI selection
class ROICanvas : public QLabel {
    Q_OBJECT
    
public:
    explicit ROICanvas(QWidget* parent = nullptr);
    
    void setFrame(const QImage& frame);
    void setShotClockROI(const ROI& roi);
    void setGameClockROI(const ROI& roi);
    ROI getShotClockROI() const { return shot_clock_roi; }
    ROI getGameClockROI() const { return game_clock_roi; }
    
    void setSelectionMode(const QString& mode); // "shot" or "game"
    QString getSelectionMode() const { return selection_mode; }
    
    void clearCurrentSelection();
    void clearAllSelections();
    
signals:
    void shotClockROIChanged(const ROI& roi);
    void gameClockROIChanged(const ROI& roi);
    
protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    
private:
    QImage current_frame;
    QImage display_frame;
    
    // ROIs in image coordinates
    ROI shot_clock_roi;
    ROI game_clock_roi;
    
    // Selection state
    QString selection_mode; // "shot" or "game"
    bool is_selecting;
    QPoint selection_start;
    QPoint selection_end;
    
    // Scale factor for display
    double scale_factor;
    QPoint display_offset;
    
    void updateDisplayFrame();
    void drawROI(QPainter& painter, const ROI& roi, const QColor& color, const QString& label);
    QPoint screenToImage(const QPoint& screen_point) const;
    QPoint imageToScreen(const QPoint& image_point) const;
    ROI getCurrentSelection() const;
};

// Main dialog for ROI selection
class ROISelectorDialog : public QDialog {
    Q_OBJECT
    
public:
    explicit ROISelectorDialog(QWidget* parent = nullptr);
    ~ROISelectorDialog();
    
    // Set camera source
    bool openCamera(int camera_index);
    bool openCamera(const QString& camera_url);
    void closeCamera();
    
    // Get selected ROIs
    ROI getShotClockROI() const;
    ROI getGameClockROI() const;
    
    // Set existing ROIs (for editing)
    void setShotClockROI(const ROI& roi);
    void setGameClockROI(const ROI& roi);
    
    // Set static frame (instead of live camera)
    void setStaticFrame(const QImage& frame);
    void hideCameraControls();
    
    // Access to canvas for external frame setting
    ROICanvas* getCanvas() { return canvas; }
    
private slots:
    void updateFrame();
    void onSelectionModeChanged(const QString& mode);
    void onSaveShotClock();
    void onSaveGameClock();
    void onResetAll();
    void onCameraSourceChanged();
    void onTestROIs();
    
private:
    void setupUI();
    void updateStatus();
    
    // UI components
    ROICanvas* canvas;
    QComboBox* camera_combo;
    QSpinBox* camera_index_spin;
    QComboBox* mode_combo;
    QPushButton* save_shot_btn;
    QPushButton* save_game_btn;
    QPushButton* reset_btn;
    QPushButton* test_btn;
    QLabel* status_label;
    QLabel* shot_roi_label;
    QLabel* game_roi_label;
    QLabel* fps_label;
    
    // Camera
    cv::VideoCapture camera;
    QTimer* frame_timer;
    
    // FPS tracking
    QElapsedTimer fps_timer;
    int frame_count;
    
    // Saved ROIs
    ROI current_shot_roi;
    ROI current_game_roi;
};

#endif // USE_CNN_OCR
