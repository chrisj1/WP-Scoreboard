#include "roi-selector-widget.h"

#ifdef USE_CNN_OCR

#include <QGridLayout>
#include <QGroupBox>
#include <QMessageBox>
#include <QElapsedTimer>

// ============================================================================
// ROICanvas Implementation
// ============================================================================

ROICanvas::ROICanvas(QWidget* parent)
    : QLabel(parent)
    , is_selecting(false)
    , selection_mode("shot")
    , scale_factor(1.0)
{
    setMinimumSize(800, 600);
    setScaledContents(false);
    setAlignment(Qt::AlignCenter);
    setStyleSheet("QLabel { background-color: #2a2a2a; border: 2px solid #555; }");
    setMouseTracking(true);
    setCursor(Qt::CrossCursor);
}

void ROICanvas::setFrame(const QImage& frame) {
    current_frame = frame;
    updateDisplayFrame();
}

void ROICanvas::setShotClockROI(const ROI& roi) {
    shot_clock_roi = roi;
    updateDisplayFrame();
}

void ROICanvas::setGameClockROI(const ROI& roi) {
    game_clock_roi = roi;
    updateDisplayFrame();
}

void ROICanvas::setSelectionMode(const QString& mode) {
    selection_mode = mode;
    setCursor(is_selecting ? Qt::CrossCursor : Qt::CrossCursor);
}

void ROICanvas::clearCurrentSelection() {
    if (selection_mode == "shot") {
        shot_clock_roi = ROI();
        emit shotClockROIChanged(shot_clock_roi);
    } else {
        game_clock_roi = ROI();
        emit gameClockROIChanged(game_clock_roi);
    }
    updateDisplayFrame();
}

void ROICanvas::clearAllSelections() {
    shot_clock_roi = ROI();
    game_clock_roi = ROI();
    emit shotClockROIChanged(shot_clock_roi);
    emit gameClockROIChanged(game_clock_roi);
    updateDisplayFrame();
}

void ROICanvas::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && !current_frame.isNull()) {
        is_selecting = true;
        selection_start = screenToImage(event->pos());
        selection_end = selection_start;
        updateDisplayFrame();
    }
}

void ROICanvas::mouseMoveEvent(QMouseEvent* event) {
    if (is_selecting && !current_frame.isNull()) {
        selection_end = screenToImage(event->pos());
        updateDisplayFrame();
    }
}

void ROICanvas::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && is_selecting) {
        is_selecting = false;
        selection_end = screenToImage(event->pos());
        
        // Calculate final ROI
        ROI new_roi = getCurrentSelection();
        
        if (new_roi.width > 10 && new_roi.height > 10) {
            if (selection_mode == "shot") {
                shot_clock_roi = new_roi;
                emit shotClockROIChanged(shot_clock_roi);
            } else {
                game_clock_roi = new_roi;
                emit gameClockROIChanged(game_clock_roi);
            }
        }
        
        selection_start = QPoint();
        selection_end = QPoint();
        updateDisplayFrame();
    }
}

void ROICanvas::paintEvent(QPaintEvent* event) {
    QLabel::paintEvent(event);
}

void ROICanvas::updateDisplayFrame() {
    if (current_frame.isNull()) {
        display_frame = QImage();
        setPixmap(QPixmap());
        return;
    }
    
    // Calculate scale to fit in widget
    QSize canvas_size = size();
    QSize image_size = current_frame.size();
    
    double scale_w = (double)canvas_size.width() / image_size.width();
    double scale_h = (double)canvas_size.height() / image_size.height();
    scale_factor = qMin(scale_w, scale_h);
    
    QSize scaled_size = image_size * scale_factor;
    display_offset = QPoint(
        (canvas_size.width() - scaled_size.width()) / 2,
        (canvas_size.height() - scaled_size.height()) / 2
    );
    
    // Create display image
    display_frame = current_frame.scaled(scaled_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    
    // Draw on pixmap
    QPixmap pixmap = QPixmap::fromImage(display_frame);
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // Draw saved ROIs
    if (shot_clock_roi.isValid()) {
        drawROI(painter, shot_clock_roi, QColor(0, 255, 0, 180), "Shot Clock");
    }
    
    if (game_clock_roi.isValid()) {
        drawROI(painter, game_clock_roi, QColor(255, 0, 0, 180), "Game Clock");
    }
    
    // Draw current selection
    if (is_selecting) {
        ROI current_sel = getCurrentSelection();
        if (current_sel.width > 0 && current_sel.height > 0) {
            QColor color = (selection_mode == "shot") ? QColor(0, 255, 255, 200) : QColor(255, 255, 0, 200);
            drawROI(painter, current_sel, color, "Current Selection");
        }
    }
    
    painter.end();
    setPixmap(pixmap);
}

void ROICanvas::drawROI(QPainter& painter, const ROI& roi, const QColor& color, const QString& label) {
    if (!roi.isValid()) return;
    
    // Scale ROI to display coordinates
    QRect rect(
        roi.x * scale_factor,
        roi.y * scale_factor,
        roi.width * scale_factor,
        roi.height * scale_factor
    );
    
    // Draw rectangle
    painter.setPen(QPen(color, 3));
    painter.drawRect(rect);
    
    // Draw semi-transparent fill
    QColor fill_color = color;
    fill_color.setAlpha(30);
    painter.fillRect(rect, fill_color);
    
    // Draw label
    QFont font = painter.font();
    font.setPointSize(10);
    font.setBold(true);
    painter.setFont(font);
    
    QRect label_rect = painter.fontMetrics().boundingRect(label);
    label_rect.adjust(-5, -3, 5, 3);
    label_rect.moveTopLeft(QPoint(rect.x(), rect.y() - label_rect.height() - 5));
    
    painter.fillRect(label_rect, color);
    painter.setPen(Qt::white);
    painter.drawText(label_rect, Qt::AlignCenter, label);
    
    // Draw corner handles
    int handle_size = 8;
    painter.setPen(Qt::NoPen);
    painter.setBrush(color);
    painter.drawRect(rect.x() - handle_size/2, rect.y() - handle_size/2, handle_size, handle_size);
    painter.drawRect(rect.right() - handle_size/2, rect.y() - handle_size/2, handle_size, handle_size);
    painter.drawRect(rect.x() - handle_size/2, rect.bottom() - handle_size/2, handle_size, handle_size);
    painter.drawRect(rect.right() - handle_size/2, rect.bottom() - handle_size/2, handle_size, handle_size);
}

QPoint ROICanvas::screenToImage(const QPoint& screen_point) const {
    // Convert from widget coordinates to image coordinates
    QPoint relative = screen_point - display_offset;
    return QPoint(relative.x() / scale_factor, relative.y() / scale_factor);
}

QPoint ROICanvas::imageToScreen(const QPoint& image_point) const {
    // Convert from image coordinates to widget coordinates
    return QPoint(image_point.x() * scale_factor, image_point.y() * scale_factor) + display_offset;
}

ROI ROICanvas::getCurrentSelection() const {
    if (selection_start.isNull() || selection_end.isNull()) {
        return ROI();
    }
    
    ROI roi;
    roi.x = qMin(selection_start.x(), selection_end.x());
    roi.y = qMin(selection_start.y(), selection_end.y());
    roi.width = qAbs(selection_end.x() - selection_start.x());
    roi.height = qAbs(selection_end.y() - selection_start.y());
    
    // Clamp to image bounds
    if (!current_frame.isNull()) {
        roi.x = qMax(0, roi.x);
        roi.y = qMax(0, roi.y);
        roi.width = qMin(roi.width, current_frame.width() - roi.x);
        roi.height = qMin(roi.height, current_frame.height() - roi.y);
    }
    
    return roi;
}

// ============================================================================
// ROISelectorDialog Implementation
// ============================================================================

ROISelectorDialog::ROISelectorDialog(QWidget* parent)
    : QDialog(parent)
    , frame_count(0)
{
    setWindowTitle("Clock ROI Selector");
    setMinimumSize(1000, 800);
    
    setupUI();
    
    // Start frame timer
    frame_timer = new QTimer(this);
    connect(frame_timer, &QTimer::timeout, this, &ROISelectorDialog::updateFrame);
    frame_timer->start(33); // ~30 FPS
    
    fps_timer.start();
}

ROISelectorDialog::~ROISelectorDialog() {
    closeCamera();
}

void ROISelectorDialog::setupUI() {
    QVBoxLayout* main_layout = new QVBoxLayout(this);
    
    // Title
    QLabel* title = new QLabel("🎯 Clock Region Selector");
    QFont title_font;
    title_font.setPointSize(14);
    title_font.setBold(true);
    title->setFont(title_font);
    title->setStyleSheet("QLabel { color: #FFD700; padding: 10px; }");
    title->setAlignment(Qt::AlignCenter);
    main_layout->addWidget(title);
    
    // Instructions
    QLabel* instructions = new QLabel(
        "1. Select camera source below\n"
        "2. Choose 'Shot Clock' or 'Game Clock' mode\n"
        "3. Click and drag to draw a rectangle around the clock\n"
        "4. Click 'Save' button to confirm the selection\n"
        "5. Repeat for the other clock\n"
        "6. Click 'Test ROIs' to verify, then 'OK' to apply"
    );
    instructions->setStyleSheet("QLabel { background-color: #3a3a3a; padding: 10px; border-radius: 5px; }");
    main_layout->addWidget(instructions);
    
    // Camera settings
    QGroupBox* camera_group = new QGroupBox("Camera Source");
    QHBoxLayout* camera_layout = new QHBoxLayout(camera_group);
    
    camera_layout->addWidget(new QLabel("Source:"));
    camera_combo = new QComboBox();
    camera_combo->addItem("USB Camera", "usb");
    camera_combo->addItem("IP Camera (RTSP)", "ip");
    camera_layout->addWidget(camera_combo);
    
    camera_layout->addWidget(new QLabel("Index/URL:"));
    camera_index_spin = new QSpinBox();
    camera_index_spin->setMinimum(0);
    camera_index_spin->setMaximum(10);
    camera_index_spin->setValue(0);
    camera_layout->addWidget(camera_index_spin);
    
    QPushButton* open_camera_btn = new QPushButton("Open Camera");
    connect(open_camera_btn, &QPushButton::clicked, this, &ROISelectorDialog::onCameraSourceChanged);
    camera_layout->addWidget(open_camera_btn);
    
    camera_layout->addStretch();
    main_layout->addWidget(camera_group);
    
    // Canvas
    canvas = new ROICanvas();
    connect(canvas, &ROICanvas::shotClockROIChanged, [this](const ROI& roi) {
        current_shot_roi = roi;
        updateStatus();
    });
    connect(canvas, &ROICanvas::gameClockROIChanged, [this](const ROI& roi) {
        current_game_roi = roi;
        updateStatus();
    });
    main_layout->addWidget(canvas, 1);
    
    // Controls
    QHBoxLayout* controls_layout = new QHBoxLayout();
    
    // Mode selection
    QGroupBox* mode_group = new QGroupBox("Selection Mode");
    QHBoxLayout* mode_layout = new QHBoxLayout(mode_group);
    
    mode_combo = new QComboBox();
    mode_combo->addItem("🎯 Shot Clock", "shot");
    mode_combo->addItem("⏱️ Game Clock", "game");
    connect(mode_combo, &QComboBox::currentTextChanged, this, &ROISelectorDialog::onSelectionModeChanged);
    mode_layout->addWidget(mode_combo);
    
    controls_layout->addWidget(mode_group);
    
    // Save buttons
    save_shot_btn = new QPushButton("💾 Save Shot Clock");
    save_shot_btn->setStyleSheet("QPushButton { background-color: #2d7a2d; color: white; padding: 8px; }");
    connect(save_shot_btn, &QPushButton::clicked, this, &ROISelectorDialog::onSaveShotClock);
    controls_layout->addWidget(save_shot_btn);
    
    save_game_btn = new QPushButton("💾 Save Game Clock");
    save_game_btn->setStyleSheet("QPushButton { background-color: #2d5a7a; color: white; padding: 8px; }");
    connect(save_game_btn, &QPushButton::clicked, this, &ROISelectorDialog::onSaveGameClock);
    controls_layout->addWidget(save_game_btn);
    
    // Reset button
    reset_btn = new QPushButton("🔄 Reset All");
    reset_btn->setStyleSheet("QPushButton { background-color: #7a2d2d; color: white; padding: 8px; }");
    connect(reset_btn, &QPushButton::clicked, this, &ROISelectorDialog::onResetAll);
    controls_layout->addWidget(reset_btn);
    
    // Test button
    test_btn = new QPushButton("🧪 Test ROIs");
    test_btn->setStyleSheet("QPushButton { background-color: #7a6d2d; color: white; padding: 8px; }");
    connect(test_btn, &QPushButton::clicked, this, &ROISelectorDialog::onTestROIs);
    controls_layout->addWidget(test_btn);
    
    controls_layout->addStretch();
    main_layout->addLayout(controls_layout);
    
    // Status
    QHBoxLayout* status_layout = new QHBoxLayout();
    
    status_label = new QLabel("No camera opened");
    status_label->setStyleSheet("QLabel { padding: 5px; }");
    status_layout->addWidget(status_label);
    
    shot_roi_label = new QLabel("Shot Clock: Not set");
    shot_roi_label->setStyleSheet("QLabel { color: #4a4; padding: 5px; }");
    status_layout->addWidget(shot_roi_label);
    
    game_roi_label = new QLabel("Game Clock: Not set");
    game_roi_label->setStyleSheet("QLabel { color: #44a; padding: 5px; }");
    status_layout->addWidget(game_roi_label);
    
    fps_label = new QLabel("FPS: 0");
    fps_label->setStyleSheet("QLabel { padding: 5px; }");
    status_layout->addWidget(fps_label);
    
    status_layout->addStretch();
    main_layout->addLayout(status_layout);
    
    // Dialog buttons
    QHBoxLayout* button_layout = new QHBoxLayout();
    button_layout->addStretch();
    
    QPushButton* ok_btn = new QPushButton("✅ OK - Apply ROIs");
    ok_btn->setStyleSheet("QPushButton { background-color: #2d7a2d; color: white; padding: 10px 20px; font-weight: bold; }");
    ok_btn->setMinimumWidth(150);
    connect(ok_btn, &QPushButton::clicked, this, &QDialog::accept);
    button_layout->addWidget(ok_btn);
    
    QPushButton* cancel_btn = new QPushButton("❌ Cancel");
    cancel_btn->setStyleSheet("QPushButton { padding: 10px 20px; }");
    connect(cancel_btn, &QPushButton::clicked, this, &QDialog::reject);
    button_layout->addWidget(cancel_btn);
    
    main_layout->addLayout(button_layout);
}

bool ROISelectorDialog::openCamera(int camera_index) {
    closeCamera();
    
    camera.open(camera_index);
    if (!camera.isOpened()) {
        status_label->setText(QString("❌ Failed to open camera %1").arg(camera_index));
        return false;
    }
    
    status_label->setText(QString("✅ Camera %1 opened").arg(camera_index));
    return true;
}

bool ROISelectorDialog::openCamera(const QString& camera_url) {
    closeCamera();
    
    camera.open(camera_url.toStdString());
    if (!camera.isOpened()) {
        status_label->setText(QString("❌ Failed to open camera: %1").arg(camera_url));
        return false;
    }
    
    status_label->setText(QString("✅ Camera opened: %1").arg(camera_url));
    return true;
}

void ROISelectorDialog::closeCamera() {
    if (camera.isOpened()) {
        camera.release();
    }
}

ROI ROISelectorDialog::getShotClockROI() const {
    return current_shot_roi;
}

ROI ROISelectorDialog::getGameClockROI() const {
    return current_game_roi;
}

void ROISelectorDialog::setShotClockROI(const ROI& roi) {
    current_shot_roi = roi;
    canvas->setShotClockROI(roi);
    updateStatus();
}

void ROISelectorDialog::setGameClockROI(const ROI& roi) {
    current_game_roi = roi;
    canvas->setGameClockROI(roi);
    updateStatus();
}

void ROISelectorDialog::updateFrame() {
    if (!camera.isOpened()) {
        return;
    }
    
    cv::Mat frame;
    if (!camera.read(frame)) {
        status_label->setText("⚠️ Failed to read frame");
        return;
    }
    
    // Convert BGR to RGB
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    
    // Convert to QImage
    QImage qimg(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
    canvas->setFrame(qimg.copy());
    
    // Update FPS
    frame_count++;
    if (fps_timer.elapsed() > 1000) {
        double fps = frame_count * 1000.0 / fps_timer.elapsed();
        fps_label->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));
        frame_count = 0;
        fps_timer.restart();
    }
}

void ROISelectorDialog::onSelectionModeChanged(const QString& mode) {
    QString mode_data = mode_combo->currentData().toString();
    canvas->setSelectionMode(mode_data);
    
    if (mode_data == "shot") {
        mode_combo->setStyleSheet("QComboBox { background-color: #2d7a2d; color: white; }");
    } else {
        mode_combo->setStyleSheet("QComboBox { background-color: #2d5a7a; color: white; }");
    }
}

void ROISelectorDialog::onSaveShotClock() {
    ROI roi = canvas->getShotClockROI();
    if (!roi.isValid()) {
        QMessageBox::warning(this, "No Selection", 
                           "Please draw a rectangle around the shot clock first.");
        return;
    }
    
    current_shot_roi = roi;
    updateStatus();
    
    QMessageBox::information(this, "Saved", 
                           QString("Shot clock ROI saved:\n(%1, %2, %3, %4)")
                           .arg(roi.x).arg(roi.y).arg(roi.width).arg(roi.height));
    
    // Switch to game clock mode
    mode_combo->setCurrentIndex(1);
}

void ROISelectorDialog::onSaveGameClock() {
    ROI roi = canvas->getGameClockROI();
    if (!roi.isValid()) {
        QMessageBox::warning(this, "No Selection", 
                           "Please draw a rectangle around the game clock first.");
        return;
    }
    
    current_game_roi = roi;
    updateStatus();
    
    QMessageBox::information(this, "Saved", 
                           QString("Game clock ROI saved:\n(%1, %2, %3, %4)")
                           .arg(roi.x).arg(roi.y).arg(roi.width).arg(roi.height));
}

void ROISelectorDialog::onResetAll() {
    auto reply = QMessageBox::question(this, "Reset All", 
                                      "Are you sure you want to reset all ROI selections?",
                                      QMessageBox::Yes | QMessageBox::No);
    
    if (reply == QMessageBox::Yes) {
        canvas->clearAllSelections();
        current_shot_roi = ROI();
        current_game_roi = ROI();
        updateStatus();
    }
}

void ROISelectorDialog::onCameraSourceChanged() {
    QString source_type = camera_combo->currentData().toString();
    
    if (source_type == "usb") {
        int index = camera_index_spin->value();
        openCamera(index);
    } else {
        // For IP camera, would need a QLineEdit for URL
        // For now, just try index as camera device
        int index = camera_index_spin->value();
        openCamera(index);
    }
}

void ROISelectorDialog::onTestROIs() {
    if (!camera.isOpened()) {
        QMessageBox::warning(this, "No Camera", "Please open a camera first.");
        return;
    }
    
    if (!current_shot_roi.isValid() && !current_game_roi.isValid()) {
        QMessageBox::warning(this, "No ROIs", "Please select at least one ROI first.");
        return;
    }
    
    // Capture a frame and show ROIs
    cv::Mat frame;
    if (!camera.read(frame)) {
        QMessageBox::warning(this, "Error", "Failed to capture frame.");
        return;
    }
    
    QString message = "ROI Test:\n\n";
    
    if (current_shot_roi.isValid()) {
        cv::Rect rect = current_shot_roi.toRect();
        if (rect.x >= 0 && rect.y >= 0 && 
            rect.x + rect.width <= frame.cols && 
            rect.y + rect.height <= frame.rows) {
            message += QString("✅ Shot Clock ROI: (%1, %2, %3, %4) - Valid\n")
                      .arg(rect.x).arg(rect.y).arg(rect.width).arg(rect.height);
        } else {
            message += "❌ Shot Clock ROI: Out of bounds!\n";
        }
    }
    
    if (current_game_roi.isValid()) {
        cv::Rect rect = current_game_roi.toRect();
        if (rect.x >= 0 && rect.y >= 0 && 
            rect.x + rect.width <= frame.cols && 
            rect.y + rect.height <= frame.rows) {
            message += QString("✅ Game Clock ROI: (%1, %2, %3, %4) - Valid\n")
                      .arg(rect.x).arg(rect.y).arg(rect.width).arg(rect.height);
        } else {
            message += "❌ Game Clock ROI: Out of bounds!\n";
        }
    }
    
    message += QString("\nFrame size: %1 x %2").arg(frame.cols).arg(frame.rows);
    
    QMessageBox::information(this, "ROI Test Results", message);
}

void ROISelectorDialog::updateStatus() {
    if (current_shot_roi.isValid()) {
        shot_roi_label->setText(QString("Shot Clock: (%1, %2, %3, %4)")
                               .arg(current_shot_roi.x)
                               .arg(current_shot_roi.y)
                               .arg(current_shot_roi.width)
                               .arg(current_shot_roi.height));
        shot_roi_label->setStyleSheet("QLabel { color: #4f4; padding: 5px; font-weight: bold; }");
    } else {
        shot_roi_label->setText("Shot Clock: Not set");
        shot_roi_label->setStyleSheet("QLabel { color: #888; padding: 5px; }");
    }
    
    if (current_game_roi.isValid()) {
        game_roi_label->setText(QString("Game Clock: (%1, %2, %3, %4)")
                               .arg(current_game_roi.x)
                               .arg(current_game_roi.y)
                               .arg(current_game_roi.width)
                               .arg(current_game_roi.height));
        game_roi_label->setStyleSheet("QLabel { color: #4af; padding: 5px; font-weight: bold; }");
    } else {
        game_roi_label->setText("Game Clock: Not set");
        game_roi_label->setStyleSheet("QLabel { color: #888; padding: 5px; }");
    }
}

#endif // USE_CNN_OCR
