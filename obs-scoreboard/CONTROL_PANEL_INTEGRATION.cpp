// Example integration of ROI Selector into Control Panel
// Add this to your control-panel.cpp file

#ifdef USE_CNN_OCR
#include "roi-selector-widget.h"
#include "clock-ocr-engine.h"
#endif

// In the ControlPanel class, add these members:
#ifdef USE_CNN_OCR
private:
    ClockOCREngine* cnn_engine;
    QTimer* detection_timer;
    
    // ROI configuration
    ROI shot_clock_roi;
    ROI game_clock_roi;
    
    // UI elements for CNN section
    QGroupBox* cnn_group;
    QCheckBox* enable_cnn_check;
    QPushButton* configure_rois_btn;
    QSpinBox* camera_index_spin;
    QLabel* shot_roi_label;
    QLabel* game_roi_label;
    QLabel* detection_status_label;
    QPushButton* start_detection_btn;
    QPushButton* stop_detection_btn;
    
private slots:
    void onConfigureROIs();
    void onStartDetection();
    void onStopDetection();
    void updateClocksFromCNN();
#endif

// In setupUI() method, add CNN detection section:
void ControlPanel::setupUI() {
    // ... existing UI code ...
    
#ifdef USE_CNN_OCR
    // CNN Detection Group
    cnn_group = new QGroupBox("🤖 CNN Clock Detection");
    QVBoxLayout* cnn_layout = new QVBoxLayout(cnn_group);
    
    // Enable CNN checkbox
    enable_cnn_check = new QCheckBox("Enable CNN Detection");
    cnn_layout->addWidget(enable_cnn_check);
    
    // Camera selection
    QHBoxLayout* camera_layout = new QHBoxLayout();
    camera_layout->addWidget(new QLabel("Camera Index:"));
    camera_index_spin = new QSpinBox();
    camera_index_spin->setMinimum(0);
    camera_index_spin->setMaximum(10);
    camera_index_spin->setValue(0);
    camera_layout->addWidget(camera_index_spin);
    camera_layout->addStretch();
    cnn_layout->addLayout(camera_layout);
    
    // Configure ROIs button
    configure_rois_btn = new QPushButton("📐 Configure Clock Regions");
    configure_rois_btn->setStyleSheet(
        "QPushButton { background-color: #3a7ca5; color: white; padding: 10px; font-weight: bold; }"
        "QPushButton:hover { background-color: #4a8cb5; }"
    );
    connect(configure_rois_btn, &QPushButton::clicked, this, &ControlPanel::onConfigureROIs);
    cnn_layout->addWidget(configure_rois_btn);
    
    // ROI status labels
    shot_roi_label = new QLabel("Shot Clock ROI: Not configured");
    shot_roi_label->setStyleSheet("QLabel { color: #888; padding: 5px; }");
    cnn_layout->addWidget(shot_roi_label);
    
    game_roi_label = new QLabel("Game Clock ROI: Not configured");
    game_roi_label->setStyleSheet("QLabel { color: #888; padding: 5px; }");
    cnn_layout->addWidget(game_roi_label);
    
    // Detection control buttons
    QHBoxLayout* detection_btns_layout = new QHBoxLayout();
    
    start_detection_btn = new QPushButton("▶️ Start Detection");
    start_detection_btn->setStyleSheet(
        "QPushButton { background-color: #2d7a2d; color: white; padding: 8px; }"
        "QPushButton:hover { background-color: #3d8a3d; }"
    );
    start_detection_btn->setEnabled(false);
    connect(start_detection_btn, &QPushButton::clicked, this, &ControlPanel::onStartDetection);
    detection_btns_layout->addWidget(start_detection_btn);
    
    stop_detection_btn = new QPushButton("⏹️ Stop Detection");
    stop_detection_btn->setStyleSheet(
        "QPushButton { background-color: #7a2d2d; color: white; padding: 8px; }"
        "QPushButton:hover { background-color: #8a3d3d; }"
    );
    stop_detection_btn->setEnabled(false);
    connect(stop_detection_btn, &QPushButton::clicked, this, &ControlPanel::onStopDetection);
    detection_btns_layout->addWidget(stop_detection_btn);
    
    cnn_layout->addLayout(detection_btns_layout);
    
    // Detection status
    detection_status_label = new QLabel("Status: Idle");
    detection_status_label->setStyleSheet(
        "QLabel { background-color: #2a2a2a; color: #aaa; padding: 10px; "
        "border-radius: 5px; font-family: monospace; }"
    );
    cnn_layout->addWidget(detection_status_label);
    
    // Add to main layout
    main_layout->addWidget(cnn_group);
    
    // Initialize CNN engine
    cnn_engine = new ClockOCREngine();
    
    // Detection timer (100ms = 10 FPS)
    detection_timer = new QTimer(this);
    connect(detection_timer, &QTimer::timeout, this, &ControlPanel::updateClocksFromCNN);
    
    // Load saved ROI settings
    QSettings settings("obs-studio", "obs-scoreboard");
    shot_clock_roi.x = settings.value("cnn/shot_roi_x", 0).toInt();
    shot_clock_roi.y = settings.value("cnn/shot_roi_y", 0).toInt();
    shot_clock_roi.width = settings.value("cnn/shot_roi_width", 0).toInt();
    shot_clock_roi.height = settings.value("cnn/shot_roi_height", 0).toInt();
    
    game_clock_roi.x = settings.value("cnn/game_roi_x", 0).toInt();
    game_clock_roi.y = settings.value("cnn/game_roi_y", 0).toInt();
    game_clock_roi.width = settings.value("cnn/game_roi_width", 0).toInt();
    game_clock_roi.height = settings.value("cnn/game_roi_height", 0).toInt();
    
    // Update ROI labels
    updateROILabels();
#endif
    
    // ... rest of existing UI code ...
}

#ifdef USE_CNN_OCR
void ControlPanel::onConfigureROIs() {
    // Create ROI selector dialog
    ROISelectorDialog dialog(this);
    
    // Set current ROIs if they exist
    if (shot_clock_roi.isValid()) {
        dialog.setShotClockROI(shot_clock_roi);
    }
    if (game_clock_roi.isValid()) {
        dialog.setGameClockROI(game_clock_roi);
    }
    
    // Open camera
    int camera_index = camera_index_spin->value();
    if (!dialog.openCamera(camera_index)) {
        detection_status_label->setText("❌ Failed to open camera");
        detection_status_label->setStyleSheet(
            "QLabel { background-color: #4a2a2a; color: #faa; padding: 10px; "
            "border-radius: 5px; font-family: monospace; }"
        );
        return;
    }
    
    // Show dialog
    if (dialog.exec() == QDialog::Accepted) {
        // Get selected ROIs
        shot_clock_roi = dialog.getShotClockROI();
        game_clock_roi = dialog.getGameClockROI();
        
        // Save to settings
        QSettings settings("obs-studio", "obs-scoreboard");
        settings.setValue("cnn/shot_roi_x", shot_clock_roi.x);
        settings.setValue("cnn/shot_roi_y", shot_clock_roi.y);
        settings.setValue("cnn/shot_roi_width", shot_clock_roi.width);
        settings.setValue("cnn/shot_roi_height", shot_clock_roi.height);
        
        settings.setValue("cnn/game_roi_x", game_clock_roi.x);
        settings.setValue("cnn/game_roi_y", game_clock_roi.y);
        settings.setValue("cnn/game_roi_width", game_clock_roi.width);
        settings.setValue("cnn/game_roi_height", game_clock_roi.height);
        
        // Update labels
        updateROILabels();
        
        // Enable start button if both ROIs are configured
        if (shot_clock_roi.isValid() && game_clock_roi.isValid()) {
            start_detection_btn->setEnabled(true);
            detection_status_label->setText("✅ ROIs configured - Ready to start");
            detection_status_label->setStyleSheet(
                "QLabel { background-color: #2a4a2a; color: #afa; padding: 10px; "
                "border-radius: 5px; font-family: monospace; }"
            );
        }
    }
}

void ControlPanel::onStartDetection() {
    // Load models (you should set these paths via UI or settings)
    QString shot_model = "C:/obs-scoreboard-models/shot_clock_model.pt";
    QString game_model = "C:/obs-scoreboard-models/game_clock_model.pt";
    
    if (!cnn_engine->loadShotClockModel(shot_model.toStdString())) {
        detection_status_label->setText("❌ Failed to load shot clock model");
        detection_status_label->setStyleSheet(
            "QLabel { background-color: #4a2a2a; color: #faa; padding: 10px; "
            "border-radius: 5px; font-family: monospace; }"
        );
        return;
    }
    
    if (!cnn_engine->loadGameClockModel(game_model.toStdString())) {
        detection_status_label->setText("❌ Failed to load game clock model");
        detection_status_label->setStyleSheet(
            "QLabel { background-color: #4a2a2a; color: #faa; padding: 10px; "
            "border-radius: 5px; font-family: monospace; }"
        );
        return;
    }
    
    // Open camera
    int camera_index = camera_index_spin->value();
    if (!cnn_engine->openCamera(camera_index)) {
        detection_status_label->setText("❌ Failed to open camera");
        detection_status_label->setStyleSheet(
            "QLabel { background-color: #4a2a2a; color: #faa; padding: 10px; "
            "border-radius: 5px; font-family: monospace; }"
        );
        return;
    }
    
    // Set ROIs
    cnn_engine->setShotClockROI(shot_clock_roi.x, shot_clock_roi.y, 
                                 shot_clock_roi.width, shot_clock_roi.height);
    cnn_engine->setGameClockROI(game_clock_roi.x, game_clock_roi.y,
                                 game_clock_roi.width, game_clock_roi.height);
    
    // Enable Bayesian filtering
    cnn_engine->enableBayesianFiltering(true);
    
    // Start detection timer
    detection_timer->start(100); // 10 Hz
    
    // Update UI
    start_detection_btn->setEnabled(false);
    stop_detection_btn->setEnabled(true);
    configure_rois_btn->setEnabled(false);
    camera_index_spin->setEnabled(false);
    
    detection_status_label->setText("🔄 Detection active - Updating clocks...");
    detection_status_label->setStyleSheet(
        "QLabel { background-color: #2a2a4a; color: #aaf; padding: 10px; "
        "border-radius: 5px; font-family: monospace; }"
    );
}

void ControlPanel::onStopDetection() {
    // Stop timer
    detection_timer->stop();
    
    // Close camera
    cnn_engine->closeCamera();
    
    // Update UI
    start_detection_btn->setEnabled(true);
    stop_detection_btn->setEnabled(false);
    configure_rois_btn->setEnabled(true);
    camera_index_spin->setEnabled(true);
    
    detection_status_label->setText("⏸️ Detection stopped");
    detection_status_label->setStyleSheet(
        "QLabel { background-color: #2a2a2a; color: #aaa; padding: 10px; "
        "border-radius: 5px; font-family: monospace; }"
    );
}

void ControlPanel::updateClocksFromCNN() {
    // Capture frame
    if (!cnn_engine->captureFrame()) {
        detection_status_label->setText("⚠️ Frame capture failed");
        return;
    }
    
    // Predict clocks
    ClockPrediction shot = cnn_engine->predictShotClock();
    ClockPrediction game = cnn_engine->predictGameClock();
    
    // Update shot clock if confidence is high
    if (shot.confidence > 0.7f && !shot.is_blocked) {
        try {
            int value = std::stoi(shot.value);
            if (value >= 0 && value <= 30) {
                shotClockSpin->setValue(value);
            }
        } catch (...) {
            // Invalid value, skip
        }
    }
    
    // Update game clock if confidence is high
    if (game.confidence > 0.7f && !game.is_blocked) {
        // Parse M:SS format
        size_t colon = game.value.find(':');
        if (colon != std::string::npos) {
            try {
                int minutes = std::stoi(game.value.substr(0, colon));
                int seconds = std::stoi(game.value.substr(colon + 1));
                
                if (minutes >= 0 && minutes <= 7 && seconds >= 0 && seconds <= 59) {
                    gameMinutesSpin->setValue(minutes);
                    gameSecondsSpin->setValue(seconds);
                }
            } catch (...) {
                // Invalid value, skip
            }
        }
    }
    
    // Update status
    QString status = QString("🔄 Shot: %1 (%2%) | Game: %3 (%4%)")
                    .arg(QString::fromStdString(shot.value))
                    .arg(shot.confidence * 100, 0, 'f', 1)
                    .arg(QString::fromStdString(game.value))
                    .arg(game.confidence * 100, 0, 'f', 1);
    
    detection_status_label->setText(status);
    
    // Update scoreboard
    updateScoreboard();
}

void ControlPanel::updateROILabels() {
    if (shot_clock_roi.isValid()) {
        shot_roi_label->setText(QString("Shot Clock ROI: (%1, %2, %3, %4)")
                               .arg(shot_clock_roi.x)
                               .arg(shot_clock_roi.y)
                               .arg(shot_clock_roi.width)
                               .arg(shot_clock_roi.height));
        shot_roi_label->setStyleSheet("QLabel { color: #4f4; padding: 5px; font-weight: bold; }");
    } else {
        shot_roi_label->setText("Shot Clock ROI: Not configured");
        shot_roi_label->setStyleSheet("QLabel { color: #888; padding: 5px; }");
    }
    
    if (game_clock_roi.isValid()) {
        game_roi_label->setText(QString("Game Clock ROI: (%1, %2, %3, %4)")
                               .arg(game_clock_roi.x)
                               .arg(game_clock_roi.y)
                               .arg(game_clock_roi.width)
                               .arg(game_clock_roi.height));
        game_roi_label->setStyleSheet("QLabel { color: #4af; padding: 5px; font-weight: bold; }");
    } else {
        game_roi_label->setText("Game Clock ROI: Not configured");
        game_roi_label->setStyleSheet("QLabel { color: #888; padding: 5px; }");
    }
}
#endif // USE_CNN_OCR
