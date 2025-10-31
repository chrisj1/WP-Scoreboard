#include <QWidget>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QSpinBox>
#include <QPushButton>
#include <QLineEdit>
#include <QGroupBox>
#include <QTimer>
#include <QAction>
#include <QComboBox>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QColorDialog>
#include <QPushButton>
#include <QSettings>
#include <QCheckBox>
#include <QFileDialog>
#include <QDir>
#include <QMessageBox>
#include <QScrollArea>
#include <QInputDialog>
#include <obs.h>
#include <obs-module.h>
#include <obs-frontend-api.h>
#include <graphics/graphics.h>

#ifdef USE_CNN_OCR
#include "roi-selector-widget.h"
// Undefine Qt macros that conflict with LibTorch AFTER including Qt-dependent headers
#pragma push_macro("slots")
#undef slots
#include "clock-ocr-engine.h"
#include "histogram-viz-source.h"
#include "averaged-frame-viz-source.h"
#include <opencv2/opencv.hpp>
#pragma pop_macro("slots")
#endif

#include "shared-schedule.h"

// Forward declaration from scoreboard-source.cpp
void update_scoreboard_data(obs_data_t *data);

// Helper function to capture a single frame from an OBS video source
static QImage captureFrameFromOBSSource(obs_source_t* source) {
	if (!source) return QImage();
	
	uint32_t width = obs_source_get_width(source);
	uint32_t height = obs_source_get_height(source);
	
	if (width == 0 || height == 0) {
		blog(LOG_WARNING, "Source has no dimensions: %dx%d", width, height);
		return QImage();
	}
	
	blog(LOG_INFO, "Attempting to capture frame from source: %dx%d", width, height);
	
	QImage frame;
	
	obs_enter_graphics();
	
	// Create a render texture
	gs_texrender_t* texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	if (!texrender) {
		blog(LOG_ERROR, "Failed to create texrender");
		obs_leave_graphics();
		return QImage();
	}
	
	// Render the source to the texture
	gs_texrender_reset(texrender);
	if (gs_texrender_begin(texrender, width, height)) {
		struct vec4 clear_color;
		vec4_zero(&clear_color);
		gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);
		
		gs_ortho(0.0f, (float)width, 0.0f, (float)height, -100.0f, 100.0f);
		
		obs_source_video_render(source);
		
		gs_texrender_end(texrender);
		
		// Get the texture
		gs_texture_t* tex = gs_texrender_get_texture(texrender);
		if (tex) {
			// Create a staging surface to read pixels back to CPU
			gs_stagesurf_t* stagesurf = gs_stagesurface_create(width, height, GS_BGRA);
			if (stagesurf) {
				// Copy texture to staging surface
				gs_stage_texture(stagesurf, tex);
				
				// Map the staging surface to read pixels
				uint8_t* data = nullptr;
				uint32_t linesize = 0;
				
				if (gs_stagesurface_map(stagesurf, &data, &linesize)) {
					// Create QImage from the pixel data
					frame = QImage(width, height, QImage::Format_RGBA8888);
					
					// Copy pixel data line by line (handle different line sizes)
					for (uint32_t y = 0; y < height; y++) {
						uint8_t* src_line = data + (y * linesize);
						uint8_t* dst_line = frame.scanLine(y);
						
						// Convert BGRA to RGBA
						for (uint32_t x = 0; x < width; x++) {
							uint8_t b = src_line[x * 4 + 0];
							uint8_t g = src_line[x * 4 + 1];
							uint8_t r = src_line[x * 4 + 2];
							uint8_t a = src_line[x * 4 + 3];
							
							dst_line[x * 4 + 0] = r;
							dst_line[x * 4 + 1] = g;
							dst_line[x * 4 + 2] = b;
							dst_line[x * 4 + 3] = a;
						}
					}
					
					gs_stagesurface_unmap(stagesurf);
					blog(LOG_INFO, "Successfully captured frame: %dx%d", width, height);
				} else {
					blog(LOG_ERROR, "Failed to map staging surface");
				}
				
				gs_stagesurface_destroy(stagesurf);
			} else {
				blog(LOG_ERROR, "Failed to create staging surface");
			}
		} else {
			blog(LOG_ERROR, "Failed to get texture from texrender");
		}
	} else {
		blog(LOG_ERROR, "Failed to begin texrender");
	}
	
	gs_texrender_destroy(texrender);
	obs_leave_graphics();
	
	if (frame.isNull()) {
		blog(LOG_ERROR, "Frame capture resulted in null image");
	}
	
	return frame;
}

class ScoreboardControlPanel : public QWidget {
	Q_OBJECT

private:
	// Team names
	QLineEdit *homeTeamEdit;
	QLineEdit *awayTeamEdit;
	
	// Scores
	QSpinBox *homeScoreSpin;
	QSpinBox *awayScoreSpin;
	
	// Clocks
	QSpinBox *shotClockSpin;
	QSpinBox *gameMinutesSpin;
	QSpinBox *gameSecondsSpin;
	
	// Period
	QSpinBox *periodSpin;
	QComboBox *periodCombo;
	
	// Man-up indicators
	QCheckBox *homeManupCheck;
	QCheckBox *awayManupCheck;
	QTimer *homeManupTimer;
	QTimer *awayManupTimer;
	
	// Exclusions
	QSpinBox *homeExclusionsSpin;
	QSpinBox *awayExclusionsSpin;
	
	// Timeouts
	QSpinBox *homeTimeoutsSpin;
	QSpinBox *awayTimeoutsSpin;
	
	// Clock control
	QTimer *gameClockTimer;
	QTimer *shotClockTimer;
	bool gameClockRunning;
	bool shotClockRunning;
	QPushButton *startGameClockBtn;
	QPushButton *stopGameClockBtn;
	QPushButton *startShotClockBtn;
	QPushButton *stopShotClockBtn;
	QPushButton *resetShotClockBtn;
	
	// Schedule
	QComboBox *gameSelectCombo;
	QPushButton *loadScheduleBtn;
	QString configDir;
	
	// Colors
	QPushButton *homeColorBtn;
	QPushButton *awayColorBtn;
	uint32_t homeColor;
	uint32_t awayColor;
	uint32_t homeTextColor;
	uint32_t awayTextColor;
	
	// CNN Models
	QLineEdit *shotClockModelEdit;
	QLineEdit *gameClockModelEdit;
	QPushButton *browseShotModelBtn;
	QPushButton *browseGameModelBtn;
	QPushButton *loadModelsBtn;
	QPushButton *selectShotRoiBtn;
	QPushButton *selectGameRoiBtn;
	
	// Transition Matrices
	QLineEdit *shotClockMatrixEdit;
	QLineEdit *gameClockMatrixEdit;
	QPushButton *browseShotMatrixBtn;
	QPushButton *browseGameMatrixBtn;
	
	// Smoothing
	QSpinBox *smoothingFramesSpinBox;
	
	// Clock visibility controls
	QCheckBox *showGameClockCheck;
	QCheckBox *showShotClockCheck;
	
	QString modelsDir;
	
#ifdef USE_CNN_OCR
	// CNN OCR Engine
	std::unique_ptr<ClockOCREngine> ocrEngine;
	QTimer *ocrUpdateTimer;
	QPushButton *startDetectionBtn;
	QPushButton *stopDetectionBtn;
	obs_source_t* shotClockVideoSource;
	obs_source_t* gameClockVideoSource;
	QString shotClockSourceName;
	QString gameClockSourceName;
	bool detectionRunning;
#endif
	
	// Team color configurations loaded from teams.csv
	struct TeamColors {
		uint32_t home_bg;
		uint32_t home_text;
		uint32_t away_bg;
		uint32_t away_text;
	};
	QMap<QString, TeamColors> teamColorMap;

public:
	ScoreboardControlPanel(QWidget *parent = nullptr) : QWidget(parent) {
		setWindowTitle("Water Polo Scoreboard Control");
		setMinimumWidth(500);
		setMinimumHeight(400);
		setMaximumHeight(700);
		
		gameClockRunning = false;
		shotClockRunning = false;
		homeColor = 0xFF0080FF; // Blue
		awayColor = 0xFFFF8000; // Orange
		homeTextColor = 0xFFFFFFFF; // White
		awayTextColor = 0xFFFFFFFF; // White
		
#ifdef USE_CNN_OCR
		// Initialize CNN OCR
		ocrEngine = std::make_unique<ClockOCREngine>();
		ocrUpdateTimer = nullptr;
		shotClockVideoSource = nullptr;
		gameClockVideoSource = nullptr;
		detectionRunning = false;
#endif
		
		// Load saved config directory
		QSettings settings("WaterPoloScoreboard", "ControlPanel");
		configDir = settings.value("configDir", "").toString();
		
		// Create scroll area for the main content
		QScrollArea *scrollArea = new QScrollArea(this);
		scrollArea->setWidgetResizable(true);
		scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
		
		// Create a container widget for all controls
		QWidget *contentWidget = new QWidget();
		QVBoxLayout *mainLayout = new QVBoxLayout(contentWidget);
		
		// Set the main layout to just contain the scroll area
		QVBoxLayout *windowLayout = new QVBoxLayout(this);
		windowLayout->setContentsMargins(0, 0, 0, 0);
		windowLayout->addWidget(scrollArea);
		scrollArea->setWidget(contentWidget);
		
		// Schedule section
		QGroupBox *scheduleGroup = new QGroupBox("Game Schedule");
		QVBoxLayout *scheduleLayout = new QVBoxLayout();
		
		QHBoxLayout *scheduleRow = new QHBoxLayout();
		loadScheduleBtn = new QPushButton("Load Schedule...");
		scheduleRow->addWidget(loadScheduleBtn);
		
		gameSelectCombo = new QComboBox();
		gameSelectCombo->addItem("Select a game...");
		scheduleRow->addWidget(gameSelectCombo, 1);
		
		scheduleLayout->addLayout(scheduleRow);
		scheduleGroup->setLayout(scheduleLayout);
		mainLayout->addWidget(scheduleGroup);
		
		// CNN Models section
		QGroupBox *modelsGroup = new QGroupBox("CNN Clock Detection Models");
		QGridLayout *modelsLayout = new QGridLayout();
		
		modelsLayout->addWidget(new QLabel("Shot Clock Model:"), 0, 0);
		shotClockModelEdit = new QLineEdit();
		shotClockModelEdit->setPlaceholderText("shot_clock_model.pt");
		shotClockModelEdit->setReadOnly(true);
		modelsLayout->addWidget(shotClockModelEdit, 0, 1);
		
		browseShotModelBtn = new QPushButton("Browse...");
		modelsLayout->addWidget(browseShotModelBtn, 0, 2);
		
		modelsLayout->addWidget(new QLabel("Game Clock Model:"), 1, 0);
		gameClockModelEdit = new QLineEdit();
		gameClockModelEdit->setPlaceholderText("game_clock_model.pt");
		gameClockModelEdit->setReadOnly(true);
		modelsLayout->addWidget(gameClockModelEdit, 1, 1);
		
		browseGameModelBtn = new QPushButton("Browse...");
		modelsLayout->addWidget(browseGameModelBtn, 1, 2);
		
		loadModelsBtn = new QPushButton("Load CNN Models");
		loadModelsBtn->setStyleSheet("QPushButton { background-color: #0066cc; color: white; font-weight: bold; }");
		modelsLayout->addWidget(loadModelsBtn, 2, 0, 1, 3);
		
		// Transition Matrix section
		modelsLayout->addWidget(new QLabel("Shot Clock Transition Matrix:"), 3, 0);
		shotClockMatrixEdit = new QLineEdit();
		shotClockMatrixEdit->setPlaceholderText("Optional: shot_transition.csv");
		shotClockMatrixEdit->setReadOnly(true);
		modelsLayout->addWidget(shotClockMatrixEdit, 3, 1);
		
		browseShotMatrixBtn = new QPushButton("Browse...");
		modelsLayout->addWidget(browseShotMatrixBtn, 3, 2);
		
		modelsLayout->addWidget(new QLabel("Game Clock Transition Matrix:"), 4, 0);
		gameClockMatrixEdit = new QLineEdit();
		gameClockMatrixEdit->setPlaceholderText("Optional: game_transition.csv");
		gameClockMatrixEdit->setReadOnly(true);
		modelsLayout->addWidget(gameClockMatrixEdit, 4, 1);
		
		browseGameMatrixBtn = new QPushButton("Browse...");
		modelsLayout->addWidget(browseGameMatrixBtn, 4, 2);
		
		// Smoothing frames
		modelsLayout->addWidget(new QLabel("Frame Smoothing:"), 5, 0);
		smoothingFramesSpinBox = new QSpinBox();
		smoothingFramesSpinBox->setRange(1, 10);
		smoothingFramesSpinBox->setValue(3);
		smoothingFramesSpinBox->setToolTip("Number of frames to average CNN predictions over (1-10)");
		modelsLayout->addWidget(smoothingFramesSpinBox, 5, 1);
		QLabel *framesLabel = new QLabel("frames");
		modelsLayout->addWidget(framesLabel, 5, 2);
		
		// ROI Selector buttons
		selectShotRoiBtn = new QPushButton("Select ROI for Shot Clock");
		selectShotRoiBtn->setEnabled(false); // Disabled until models are loaded
		modelsLayout->addWidget(selectShotRoiBtn, 6, 0, 1, 3);
		
		selectGameRoiBtn = new QPushButton("Select ROI for Game Clock");
		selectGameRoiBtn->setEnabled(false); // Disabled until models are loaded
		modelsLayout->addWidget(selectGameRoiBtn, 7, 0, 1, 3);
		
#ifdef USE_CNN_OCR
		// Detection control buttons
		startDetectionBtn = new QPushButton("▶ Start Clock Detection");
		startDetectionBtn->setEnabled(false); // Disabled until ROIs are set
		startDetectionBtn->setStyleSheet("QPushButton { background-color: #00aa00; color: white; font-weight: bold; }");
		modelsLayout->addWidget(startDetectionBtn, 8, 0, 1, 3);
		
		stopDetectionBtn = new QPushButton("⏸ Stop Clock Detection");
		stopDetectionBtn->setEnabled(false);
		stopDetectionBtn->setStyleSheet("QPushButton { background-color: #cc0000; color: white; font-weight: bold; }");
		modelsLayout->addWidget(stopDetectionBtn, 9, 0, 1, 3);
#endif
		
		modelsGroup->setLayout(modelsLayout);
		mainLayout->addWidget(modelsGroup);
		
		// Teams section
		QGroupBox *teamsGroup = new QGroupBox("Teams");
		QGridLayout *teamsLayout = new QGridLayout();
		
		teamsLayout->addWidget(new QLabel("Home Team:"), 0, 0);
		homeTeamEdit = new QLineEdit("HOME");
		teamsLayout->addWidget(homeTeamEdit, 0, 1);
		
		homeColorBtn = new QPushButton("Home Color");
		homeColorBtn->setStyleSheet("background-color: #0080FF;");
		teamsLayout->addWidget(homeColorBtn, 0, 2);
		
		teamsLayout->addWidget(new QLabel("Away Team:"), 1, 0);
		awayTeamEdit = new QLineEdit("AWAY");
		teamsLayout->addWidget(awayTeamEdit, 1, 1);
		
		awayColorBtn = new QPushButton("Away Color");
		awayColorBtn->setStyleSheet("background-color: #FF8000;");
		teamsLayout->addWidget(awayColorBtn, 1, 2);
		
		teamsGroup->setLayout(teamsLayout);
		mainLayout->addWidget(teamsGroup);
		
		// Scores section
		QGroupBox *scoresGroup = new QGroupBox("Scores");
		QGridLayout *scoresLayout = new QGridLayout();
		
		scoresLayout->addWidget(new QLabel("Home Score:"), 0, 0);
		homeScoreSpin = new QSpinBox();
		homeScoreSpin->setRange(0, 99);
		homeScoreSpin->setValue(0);
		scoresLayout->addWidget(homeScoreSpin, 0, 1);
		
		QPushButton *homeScorePlusBtn = new QPushButton("+");
		QPushButton *homeScoreMinusBtn = new QPushButton("-");
		scoresLayout->addWidget(homeScorePlusBtn, 0, 2);
		scoresLayout->addWidget(homeScoreMinusBtn, 0, 3);
		
		scoresLayout->addWidget(new QLabel("Away Score:"), 1, 0);
		awayScoreSpin = new QSpinBox();
		awayScoreSpin->setRange(0, 99);
		awayScoreSpin->setValue(0);
		scoresLayout->addWidget(awayScoreSpin, 1, 1);
		
		QPushButton *awayScorePlusBtn = new QPushButton("+");
		QPushButton *awayScoreMinusBtn = new QPushButton("-");
		scoresLayout->addWidget(awayScorePlusBtn, 1, 2);
		scoresLayout->addWidget(awayScoreMinusBtn, 1, 3);
		
		scoresGroup->setLayout(scoresLayout);
		mainLayout->addWidget(scoresGroup);
		
		// Game Clock section
		QGroupBox *gameClockGroup = new QGroupBox("Game Clock");
		QGridLayout *gameClockLayout = new QGridLayout();
		
		gameClockLayout->addWidget(new QLabel("Minutes:"), 0, 0);
		gameMinutesSpin = new QSpinBox();
		gameMinutesSpin->setRange(0, 8);
		gameMinutesSpin->setValue(8);
		gameClockLayout->addWidget(gameMinutesSpin, 0, 1);
		
		gameClockLayout->addWidget(new QLabel("Seconds:"), 0, 2);
		gameSecondsSpin = new QSpinBox();
		gameSecondsSpin->setRange(0, 59);
		gameSecondsSpin->setValue(0);
		gameClockLayout->addWidget(gameSecondsSpin, 0, 3);
		
		startGameClockBtn = new QPushButton("Start");
		stopGameClockBtn = new QPushButton("Stop");
		QPushButton *resetGameClockBtn = new QPushButton("Reset");
		
		gameClockLayout->addWidget(startGameClockBtn, 1, 0);
		gameClockLayout->addWidget(stopGameClockBtn, 1, 1);
		gameClockLayout->addWidget(resetGameClockBtn, 1, 2);
		
		gameClockGroup->setLayout(gameClockLayout);
		mainLayout->addWidget(gameClockGroup);
		
		// Shot Clock section
		QGroupBox *shotClockGroup = new QGroupBox("Shot Clock (30 seconds)");
		QHBoxLayout *shotClockLayout = new QHBoxLayout();
		
		shotClockSpin = new QSpinBox();
		shotClockSpin->setRange(0, 30);
		shotClockSpin->setValue(30);
		shotClockLayout->addWidget(shotClockSpin);
		
		startShotClockBtn = new QPushButton("Start");
		stopShotClockBtn = new QPushButton("Stop");
		resetShotClockBtn = new QPushButton("Reset to 30");
		
		shotClockLayout->addWidget(startShotClockBtn);
		shotClockLayout->addWidget(stopShotClockBtn);
		shotClockLayout->addWidget(resetShotClockBtn);
		
		shotClockGroup->setLayout(shotClockLayout);
		mainLayout->addWidget(shotClockGroup);
		
		// Period, Exclusions, Timeouts section
		QGroupBox *gameInfoGroup = new QGroupBox("Game Info");
		QGridLayout *gameInfoLayout = new QGridLayout();
		
		gameInfoLayout->addWidget(new QLabel("Period:"), 0, 0);
		
		// Period combo box
		periodCombo = new QComboBox();
		periodCombo->addItem("Q1", 1);
		periodCombo->addItem("Q2", 2);
		periodCombo->addItem("Q3", 3);
		periodCombo->addItem("Q4", 4);
		periodCombo->addItem("5th", 5);
		periodCombo->addItem("Final", 0);
		periodCombo->addItem("Shootout", -1);
		periodCombo->setCurrentIndex(0);
		gameInfoLayout->addWidget(periodCombo, 0, 1);
		
		// Man-up indicators
		homeManupCheck = new QCheckBox("Home Man-Up");
		awayManupCheck = new QCheckBox("Away Man-Up");
		gameInfoLayout->addWidget(homeManupCheck, 0, 2);
		gameInfoLayout->addWidget(awayManupCheck, 0, 3);
		
		// Initialize man-up timers
		homeManupTimer = new QTimer(this);
		homeManupTimer->setSingleShot(true);
		awayManupTimer = new QTimer(this);
		awayManupTimer->setSingleShot(true);
		
		gameInfoLayout->addWidget(new QLabel("Home Exclusions:"), 1, 0);
		homeExclusionsSpin = new QSpinBox();
		homeExclusionsSpin->setRange(0, 10);
		homeExclusionsSpin->setValue(0);
		gameInfoLayout->addWidget(homeExclusionsSpin, 1, 1);
		
		gameInfoLayout->addWidget(new QLabel("Away Exclusions:"), 1, 2);
		awayExclusionsSpin = new QSpinBox();
		awayExclusionsSpin->setRange(0, 10);
		awayExclusionsSpin->setValue(0);
		gameInfoLayout->addWidget(awayExclusionsSpin, 1, 3);
		
		gameInfoLayout->addWidget(new QLabel("Home Timeouts:"), 2, 0);
		homeTimeoutsSpin = new QSpinBox();
		homeTimeoutsSpin->setRange(0, 2);
		homeTimeoutsSpin->setValue(2);
		gameInfoLayout->addWidget(homeTimeoutsSpin, 2, 1);
		
		gameInfoLayout->addWidget(new QLabel("Away Timeouts:"), 2, 2);
		awayTimeoutsSpin = new QSpinBox();
		awayTimeoutsSpin->setRange(0, 2);
		awayTimeoutsSpin->setValue(2);
		gameInfoLayout->addWidget(awayTimeoutsSpin, 2, 3);
		
		gameInfoGroup->setLayout(gameInfoLayout);
		mainLayout->addWidget(gameInfoGroup);
		
		// Clock visibility controls
		QGroupBox *clockVisibilityGroup = new QGroupBox("Clock Visibility");
		QHBoxLayout *clockVisibilityLayout = new QHBoxLayout();
		showGameClockCheck = new QCheckBox("Show Game Clock");
		showShotClockCheck = new QCheckBox("Show Shot Clock");
		showGameClockCheck->setChecked(true);
		showShotClockCheck->setChecked(true);
		blog(LOG_INFO, "Clock visibility controls created and initialized");
		clockVisibilityLayout->addWidget(showGameClockCheck);
		clockVisibilityLayout->addWidget(showShotClockCheck);
		clockVisibilityGroup->setLayout(clockVisibilityLayout);
		mainLayout->addWidget(clockVisibilityGroup);
		blog(LOG_INFO, "Clock visibility group added to main layout");

#ifdef USE_CNN_OCR
		// Reset priors button
		QPushButton *resetPriorsBtn = new QPushButton("Reset Bayesian Priors");
		resetPriorsBtn->setStyleSheet("QPushButton { background-color: #cc6600; color: white; font-weight: bold; padding: 8px; }");
		mainLayout->addWidget(resetPriorsBtn);
#endif
		
		// Update button
		QPushButton *updateBtn = new QPushButton("Update Scoreboard");
		updateBtn->setStyleSheet("QPushButton { background-color: #0e8a0e; color: white; font-weight: bold; padding: 10px; }");
		mainLayout->addWidget(updateBtn);
		
		// Setup timers
		gameClockTimer = new QTimer(this);
		gameClockTimer->setInterval(1000); // 1 second
		
		shotClockTimer = new QTimer(this);
		shotClockTimer->setInterval(1000); // 1 second
		
		// Connect signals
		connect(updateBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::updateScoreboard);
		
		connect(homeScorePlusBtn, &QPushButton::clicked, [this]() {
			homeScoreSpin->setValue(homeScoreSpin->value() + 1);
			updateScoreboard();
		});
		connect(homeScoreMinusBtn, &QPushButton::clicked, [this]() {
			homeScoreSpin->setValue(homeScoreSpin->value() - 1);
			updateScoreboard();
		});
		connect(awayScorePlusBtn, &QPushButton::clicked, [this]() {
			awayScoreSpin->setValue(awayScoreSpin->value() + 1);
			updateScoreboard();
		});
		connect(awayScoreMinusBtn, &QPushButton::clicked, [this]() {
			awayScoreSpin->setValue(awayScoreSpin->value() - 1);
			updateScoreboard();
		});
		
		connect(startGameClockBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::startGameClock);
		connect(stopGameClockBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::stopGameClock);
		connect(resetGameClockBtn, &QPushButton::clicked, [this]() {
			gameMinutesSpin->setValue(8);
			gameSecondsSpin->setValue(0);
			updateScoreboard();
		});
		
		connect(startShotClockBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::startShotClock);
		connect(stopShotClockBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::stopShotClock);
		connect(resetShotClockBtn, &QPushButton::clicked, [this]() {
			shotClockSpin->setValue(30);
			updateScoreboard();
		});
		
		connect(gameClockTimer, &QTimer::timeout, this, &ScoreboardControlPanel::onGameClockTick);
		connect(shotClockTimer, &QTimer::timeout, this, &ScoreboardControlPanel::onShotClockTick);
		
		// Auto-update on any change
		connect(homeTeamEdit, &QLineEdit::textChanged, this, &ScoreboardControlPanel::updateScoreboard);
		connect(awayTeamEdit, &QLineEdit::textChanged, this, &ScoreboardControlPanel::updateScoreboard);
		connect(homeScoreSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &ScoreboardControlPanel::updateScoreboard);
		connect(awayScoreSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &ScoreboardControlPanel::updateScoreboard);
		connect(periodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ScoreboardControlPanel::updateScoreboard);
		connect(homeExclusionsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &ScoreboardControlPanel::updateScoreboard);
		connect(awayExclusionsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &ScoreboardControlPanel::updateScoreboard);
		connect(homeTimeoutsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &ScoreboardControlPanel::updateScoreboard);
		connect(awayTimeoutsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &ScoreboardControlPanel::updateScoreboard);
		
		// Man-up indicators
		connect(homeManupCheck, &QCheckBox::toggled, this, &ScoreboardControlPanel::onHomeManupToggled);
		connect(awayManupCheck, &QCheckBox::toggled, this, &ScoreboardControlPanel::onAwayManupToggled);
		connect(homeManupTimer, &QTimer::timeout, this, [this]() {
			homeManupCheck->setChecked(false);
		});
		connect(awayManupTimer, &QTimer::timeout, this, [this]() {
			awayManupCheck->setChecked(false);
		});
		
		// Schedule and color connections
		connect(loadScheduleBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::loadSchedule);
		connect(gameSelectCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ScoreboardControlPanel::onGameSelected);
		connect(homeColorBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::chooseHomeColor);
		connect(awayColorBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::chooseAwayColor);
		
		// CNN model connections
		connect(browseShotModelBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::browseShotClockModel);
		connect(browseGameModelBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::browseGameClockModel);
		connect(browseShotMatrixBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::browseShotClockMatrix);
		connect(browseGameMatrixBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::browseGameClockMatrix);
		connect(loadModelsBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::loadCNNModels);
		connect(selectShotRoiBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::selectShotClockRoi);
		connect(selectGameRoiBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::selectGameClockRoi);
		connect(smoothingFramesSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &ScoreboardControlPanel::onSmoothingFramesChanged);
		
		// Connect clock visibility checkboxes
		connect(showGameClockCheck, &QCheckBox::toggled, this, [this](bool checked) {
			blog(LOG_INFO, "Game clock checkbox toggled: %s", checked ? "true" : "false");
			qDebug() << "Game clock checkbox toggled:" << checked;
			updateScoreboard();
		});
		connect(showShotClockCheck, &QCheckBox::toggled, this, [this](bool checked) {
			blog(LOG_INFO, "Shot clock checkbox toggled: %s", checked ? "true" : "false");
			qDebug() << "Shot clock checkbox toggled:" << checked;
			updateScoreboard();
		});

#ifdef USE_CNN_OCR
		// Connect reset priors button
		connect(resetPriorsBtn, &QPushButton::clicked, this, [this]() {
			if (ocrEngine) {
				ocrEngine->resetPriors();
				QMessageBox::information(this, "Bayesian Priors Reset", 
					"Both shot clock and game clock priors have been reset to uniform.");
			} else {
				QMessageBox::warning(this, "OCR Engine Not Available", 
					"CNN OCR engine is not loaded. Cannot reset priors.");
			}
		});
#endif

#ifdef USE_CNN_OCR
		connect(startDetectionBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::startClockDetection);
		connect(stopDetectionBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::stopClockDetection);
#endif
		
		// Initialize models directory path using environment variable
		QString appData = QString::fromLocal8Bit(qgetenv("APPDATA"));
		if (!appData.isEmpty()) {
			modelsDir = appData + "/obs-studio/plugin_config/obs-scoreboard/models";
			QDir dir(modelsDir);
			if (!dir.exists()) {
				dir.mkpath(".");
			}
			
			// Load saved model paths
			QSettings modelSettings("WaterPoloScoreboard", "CNNModels");
			QString shotModelPath = modelSettings.value("shotClockModel", modelsDir + "/shot_clock_model.pt").toString();
			QString gameModelPath = modelSettings.value("gameClockModel", modelsDir + "/game_clock_model.pt").toString();
			QString shotMatrixPath = modelSettings.value("shotClockMatrix", "").toString();
			QString gameMatrixPath = modelSettings.value("gameClockMatrix", "").toString();
			
			shotClockModelEdit->setText(shotModelPath);
			gameClockModelEdit->setText(gameModelPath);
			
			if (!shotMatrixPath.isEmpty()) {
				shotClockMatrixEdit->setText(shotMatrixPath);
			}
			if (!gameMatrixPath.isEmpty()) {
				gameClockMatrixEdit->setText(gameMatrixPath);
			}
		}
		
		// Auto-load schedule if config directory is set (with delay to ensure OBS is ready)
		if (!configDir.isEmpty()) {
			blog(LOG_INFO, "Auto-loading config from: %s", configDir.toUtf8().constData());
			QTimer::singleShot(500, this, [this]() {
				loadScheduleFromPath(configDir);
			});
		}
	}

private slots:
	void updateScoreboard() {
		blog(LOG_INFO, "Control panel updateScoreboard() called");
		obs_data_t *data = obs_data_create();
		
		obs_data_set_string(data, "home_team", homeTeamEdit->text().toUtf8().constData());
		obs_data_set_string(data, "away_team", awayTeamEdit->text().toUtf8().constData());
		obs_data_set_int(data, "home_score", homeScoreSpin->value());
		obs_data_set_int(data, "away_score", awayScoreSpin->value());
		obs_data_set_int(data, "shot_clock", shotClockSpin->value());
		obs_data_set_int(data, "game_clock_minutes", gameMinutesSpin->value());
		obs_data_set_int(data, "game_clock_seconds", gameSecondsSpin->value());
		
		// Period - use combo box value and text
		int periodValue = periodCombo->currentData().toInt();
		QString periodText = periodCombo->currentText();
		obs_data_set_int(data, "period", abs(periodValue));
		if (periodValue <= 0) {
			// Special periods (Final, 5th, Shootout) - send text
			obs_data_set_string(data, "period_text", periodText.toUtf8().constData());
		} else {
			// Regular quarters - clear period_text
			obs_data_set_string(data, "period_text", "");
		}
		
		obs_data_set_int(data, "home_exclusions", homeExclusionsSpin->value());
		obs_data_set_int(data, "away_exclusions", awayExclusionsSpin->value());
		obs_data_set_int(data, "home_timeouts", homeTimeoutsSpin->value());
		obs_data_set_int(data, "away_timeouts", awayTimeoutsSpin->value());
		obs_data_set_int(data, "home_color", homeColor);
		obs_data_set_int(data, "away_color", awayColor);
		obs_data_set_int(data, "home_text_color", homeTextColor);
		obs_data_set_int(data, "away_text_color", awayTextColor);
		
		// Man-up indicators
		obs_data_set_bool(data, "home_manup", homeManupCheck->isChecked());
		obs_data_set_bool(data, "away_manup", awayManupCheck->isChecked());
		
		// Clock visibility
		bool showGameClock = showGameClockCheck->isChecked();
		bool showShotClock = showShotClockCheck->isChecked();
		blog(LOG_INFO, "Control panel: Setting clock visibility - game=%s, shot=%s", 
		     showGameClock ? "true" : "false", showShotClock ? "true" : "false");
		obs_data_set_bool(data, "show_game_clock", showGameClock);
		obs_data_set_bool(data, "show_shot_clock", showShotClock);
		
		// Send logo paths
		if (!configDir.isEmpty()) {
			QString homeTeam = homeTeamEdit->text().toLower().replace(" ", "");
			QString awayTeam = awayTeamEdit->text().toLower().replace(" ", "");
			
			// Try PNG first, fallback to SVG (though GDI+ doesn't support SVG well)
			QString homeLogo = configDir + "/logos/" + homeTeam + ".png";
			QString awayLogo = configDir + "/logos/" + awayTeam + ".png";
			
			// Check if PNG exists, otherwise try SVG
			if (!QFile::exists(homeLogo)) {
				homeLogo = configDir + "/logos/" + homeTeam + ".svg";
			}
			if (!QFile::exists(awayLogo)) {
				awayLogo = configDir + "/logos/" + awayTeam + ".svg";
			}
			
			blog(LOG_INFO, "Home logo path: %s", homeLogo.toUtf8().constData());
			blog(LOG_INFO, "Away logo path: %s", awayLogo.toUtf8().constData());
			
			obs_data_set_string(data, "home_logo_path", homeLogo.toUtf8().constData());
			obs_data_set_string(data, "away_logo_path", awayLogo.toUtf8().constData());
			
			// Get next game from combo box
			int currentIndex = gameSelectCombo->currentIndex();
			if (currentIndex >= 0 && currentIndex < gameSelectCombo->count() - 1) {
				// There is a next game
				QStringList nextTeams = gameSelectCombo->itemData(currentIndex + 1).toStringList();
				if (nextTeams.size() >= 2) {
					QString nextHome = nextTeams[0].toLower().replace(" ", "");
					QString nextAway = nextTeams[1].toLower().replace(" ", "");
					
					QString nextHomeLogo = configDir + "/logos/" + nextHome + ".png";
					QString nextAwayLogo = configDir + "/logos/" + nextAway + ".png";
					
					if (!QFile::exists(nextHomeLogo)) nextHomeLogo = configDir + "/logos/" + nextHome + ".svg";
					if (!QFile::exists(nextAwayLogo)) nextAwayLogo = configDir + "/logos/" + nextAway + ".svg";
					
					obs_data_set_string(data, "next_home_team", nextTeams[0].toUtf8().constData());
					obs_data_set_string(data, "next_away_team", nextTeams[1].toUtf8().constData());
					obs_data_set_string(data, "next_home_logo_path", nextHomeLogo.toUtf8().constData());
					obs_data_set_string(data, "next_away_logo_path", nextAwayLogo.toUtf8().constData());
				}
			} else {
				// No next game - set empty strings
				obs_data_set_string(data, "next_home_team", "");
				obs_data_set_string(data, "next_away_team", "");
				obs_data_set_string(data, "next_home_logo_path", "");
				obs_data_set_string(data, "next_away_logo_path", "");
			}
		}
		
		blog(LOG_INFO, "Control panel: About to call update_scoreboard_data");
		blog(LOG_ERROR, "Control panel: About to call update_scoreboard_data (ERROR LEVEL)");
		update_scoreboard_data(data);
		blog(LOG_INFO, "Control panel: update_scoreboard_data completed");
		blog(LOG_ERROR, "Control panel: update_scoreboard_data completed (ERROR LEVEL)");
		
		obs_data_release(data);
	}
	
	void startGameClock() {
		if (!gameClockRunning) {
			gameClockRunning = true;
			gameClockTimer->start();
			startGameClockBtn->setEnabled(false);
			stopGameClockBtn->setEnabled(true);
		}
	}
	
	void stopGameClock() {
		if (gameClockRunning) {
			gameClockRunning = false;
			gameClockTimer->stop();
			startGameClockBtn->setEnabled(true);
			stopGameClockBtn->setEnabled(false);
		}
	}
	
	void startShotClock() {
		if (!shotClockRunning) {
			shotClockRunning = true;
			shotClockTimer->start();
			startShotClockBtn->setEnabled(false);
			stopShotClockBtn->setEnabled(true);
		}
	}
	
	void stopShotClock() {
		if (shotClockRunning) {
			shotClockRunning = false;
			shotClockTimer->stop();
			startShotClockBtn->setEnabled(true);
			stopShotClockBtn->setEnabled(false);
		}
	}
	
	void onGameClockTick() {
		int minutes = gameMinutesSpin->value();
		int seconds = gameSecondsSpin->value();
		
		if (seconds == 0) {
			if (minutes == 0) {
				// Clock expired
				stopGameClock();
				return;
			}
			minutes--;
			seconds = 59;
		} else {
			seconds--;
		}
		
		gameMinutesSpin->setValue(minutes);
		gameSecondsSpin->setValue(seconds);
		updateScoreboard();
	}
	
	void onShotClockTick() {
		int shotClock = shotClockSpin->value();
		
		if (shotClock <= 0) {
			stopShotClock();
			return;
		}
		
		shotClockSpin->setValue(shotClock - 1);
		updateScoreboard();
	}
	
	void loadSchedule() {
		QString dir = QFileDialog::getExistingDirectory(this, "Select Configuration Directory", configDir);
		if (dir.isEmpty()) return;
		
		configDir = dir;
		
		// Save config directory to settings
		QSettings settings("WaterPoloScoreboard", "ControlPanel");
		settings.setValue("configDir", configDir);
		
		loadScheduleFromPath(configDir);
	}
	
	void loadScheduleFromPath(const QString &dir) {
		if (dir.isEmpty()) return;
		
		// Load teams.csv first
		loadTeamColors(dir + "/teams.csv");
		
		// Update global schedule data (shared with schedule source)
		std::string config_dir = dir.toUtf8().constData();
		update_global_schedule_data(config_dir);
		
		// Then load schedule.csv into the combo box for UI
		QString schedulePath = dir + "/schedule.csv";
		
		QFile file(schedulePath);
		if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
			blog(LOG_WARNING, "Could not open schedule.csv at %s", schedulePath.toUtf8().constData());
			return;
		}
		
		gameSelectCombo->clear();
		gameSelectCombo->addItem("Select a game...");
		
		QTextStream in(&file);
		QString header = in.readLine(); // Skip header
		
		while (!in.atEnd()) {
			QString line = in.readLine();
			QStringList parts = line.split(',');
			
			if (parts.size() >= 3) {
				QString time = parts[0].trimmed();
				QString home = parts[1].trimmed();
				QString away = parts[2].trimmed();
				QString displayText = QString("%1: %2 vs %3").arg(time, home, away);
				
				gameSelectCombo->addItem(displayText);
				gameSelectCombo->setItemData(gameSelectCombo->count() - 1, QStringList() << home << away);
			}
		}
		
		file.close();
		blog(LOG_INFO, "Loaded schedule from %s", schedulePath.toUtf8().constData());
		
		// Notify schedule sources that data has been updated
		notify_schedule_data_updated();
		
		// Auto-select the first game if available
		if (gameSelectCombo->count() > 1) {
			blog(LOG_INFO, "Auto-selecting first game (index 1)");
			gameSelectCombo->setCurrentIndex(1);
			// Force an update in case the signal doesn't fire
			onGameSelected(1);
		}
	}
	
	void onGameSelected(int index) {
		if (index <= 0) return; // "Select a game..." or invalid
		
		QStringList teams = gameSelectCombo->itemData(index).toStringList();
		if (teams.size() >= 2) {
			blog(LOG_INFO, "Game selected: %s vs %s", teams[0].toUtf8().constData(), teams[1].toUtf8().constData());
			
			homeTeamEdit->setText(teams[0]);
			awayTeamEdit->setText(teams[1]);
			
			// Set team-specific colors (you can customize this per team)
			setTeamColors(teams[0], teams[1]);
			
			updateScoreboard();
		}
	}
	
	void loadTeamColors(const QString &teamsPath) {
		QFile file(teamsPath);
		if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
			blog(LOG_WARNING, "Could not open teams.csv at %s", teamsPath.toUtf8().constData());
			return;
		}
		
		teamColorMap.clear();
		QTextStream in(&file);
		QString header = in.readLine(); // Skip header: name,home_bg,home_text,away_bg,away_text
		
		while (!in.atEnd()) {
			QString line = in.readLine();
			QStringList parts = line.split(',');
			
			if (parts.size() >= 5) {
				QString teamName = parts[0].trimmed();
				QString homeBgHex = parts[1].trimmed();
				QString homeTextHex = parts[2].trimmed();
				QString awayBgHex = parts[3].trimmed();
				QString awayTextHex = parts[4].trimmed();
				
				TeamColors colors;
				colors.home_bg = hexToColor(homeBgHex);
				colors.home_text = hexToColor(homeTextHex);
				colors.away_bg = hexToColor(awayBgHex);
				colors.away_text = hexToColor(awayTextHex);
				
				teamColorMap[teamName] = colors;
			}
		}
		
		file.close();
		blog(LOG_INFO, "Loaded %d team color configurations from %s", 
		     teamColorMap.size(), teamsPath.toUtf8().constData());
	}
	
	uint32_t hexToColor(const QString &hex) {
		// Convert #RRGGBB to 0xFFRRGGBB
		QString cleanHex = hex;
		if (cleanHex.startsWith('#')) {
			cleanHex = cleanHex.mid(1);
		}
		
		bool ok;
		uint32_t rgb = cleanHex.toUInt(&ok, 16);
		if (ok) {
			return 0xFF000000 | rgb; // Add full alpha
		}
		return 0xFFFFFFFF; // Default to white if parsing fails
	}
	
	void setTeamColors(const QString &home, const QString &away) {
		// Use loaded team colors from teams.csv
		if (teamColorMap.contains(home)) {
			homeColor = teamColorMap[home].home_bg;
			homeTextColor = teamColorMap[home].home_text;
		} else {
			homeColor = 0xFF0080FF; // Default blue
			homeTextColor = 0xFFFFFFFF; // Default white
		}
		
		if (teamColorMap.contains(away)) {
			awayColor = teamColorMap[away].away_bg;
			awayTextColor = teamColorMap[away].away_text;
		} else {
			awayColor = 0xFFFF8000; // Default orange
			awayTextColor = 0xFFFFFFFF; // Default white
		}
		
		// Update button colors
		updateColorButtons();
		updateScoreboard();
	}
	
	void chooseHomeColor() {
		QColor current = QColor((homeColor >> 16) & 0xFF, (homeColor >> 8) & 0xFF, homeColor & 0xFF);
		QColor color = QColorDialog::getColor(current, this, "Choose Home Team Color");
		
		if (color.isValid()) {
			homeColor = 0xFF000000 | (color.red() << 16) | (color.green() << 8) | color.blue();
			updateColorButtons();
			updateScoreboard();
		}
	}
	
	void chooseAwayColor() {
		QColor current = QColor((awayColor >> 16) & 0xFF, (awayColor >> 8) & 0xFF, awayColor & 0xFF);
		QColor color = QColorDialog::getColor(current, this, "Choose Away Team Color");
		
		if (color.isValid()) {
			awayColor = 0xFF000000 | (color.red() << 16) | (color.green() << 8) | color.blue();
			updateColorButtons();
			updateScoreboard();
		}
	}
	
	void onHomeManupToggled(bool checked) {
		if (checked) {
			homeManupTimer->start(30000); // 30 seconds
		} else {
			homeManupTimer->stop();
		}
		updateScoreboard();
	}
	
	void onAwayManupToggled(bool checked) {
		if (checked) {
			awayManupTimer->start(30000); // 30 seconds
		} else {
			awayManupTimer->stop();
		}
		updateScoreboard();
	}
	
	void browseShotClockModel() {
		QString defaultDir = modelsDir.isEmpty() ? QDir::homePath() : modelsDir;
		QString filename = QFileDialog::getOpenFileName(
			this,
			"Select Shot Clock CNN Model",
			defaultDir,
			"TorchScript Models (*.pt);;All Files (*.*)"
		);
		
		if (!filename.isEmpty()) {
			shotClockModelEdit->setText(filename);
			QSettings settings("WaterPoloScoreboard", "CNNModels");
			settings.setValue("shotClockModel", filename);
		}
	}
	
	void browseGameClockModel() {
		QString defaultDir = modelsDir.isEmpty() ? QDir::homePath() : modelsDir;
		QString filename = QFileDialog::getOpenFileName(
			this,
			"Select Game Clock CNN Model",
			defaultDir,
			"TorchScript Models (*.pt);;All Files (*.*)"
		);
		
		if (!filename.isEmpty()) {
			gameClockModelEdit->setText(filename);
			QSettings settings("WaterPoloScoreboard", "CNNModels");
			settings.setValue("gameClockModel", filename);
		}
	}
	
	void browseShotClockMatrix() {
		QString defaultDir = modelsDir.isEmpty() ? QDir::homePath() : modelsDir;
		QString filename = QFileDialog::getOpenFileName(
			this,
			"Select Shot Clock Transition Matrix CSV",
			defaultDir,
			"CSV Files (*.csv);;All Files (*.*)"
		);
		
		if (!filename.isEmpty()) {
			shotClockMatrixEdit->setText(filename);
			QSettings settings("WaterPoloScoreboard", "CNNModels");
			settings.setValue("shotClockMatrix", filename);
			
#ifdef USE_CNN_OCR
			// Try to load immediately if OCR engine exists
			if (ocrEngine) {
				if (ocrEngine->loadShotClockTransitionMatrix(filename.toStdString())) {
					blog(LOG_INFO, "Shot clock transition matrix loaded successfully");
				} else {
					blog(LOG_ERROR, "Failed to load shot clock transition matrix");
				}
			}
#endif
		}
	}
	
	void browseGameClockMatrix() {
		QString defaultDir = modelsDir.isEmpty() ? QDir::homePath() : modelsDir;
		QString filename = QFileDialog::getOpenFileName(
			this,
			"Select Game Clock Transition Matrix CSV",
			defaultDir,
			"CSV Files (*.csv);;All Files (*.*)"
		);
		
		if (!filename.isEmpty()) {
			gameClockMatrixEdit->setText(filename);
			QSettings settings("WaterPoloScoreboard", "CNNModels");
			settings.setValue("gameClockMatrix", filename);
			
#ifdef USE_CNN_OCR
			// Try to load immediately if OCR engine exists
			if (ocrEngine) {
				if (ocrEngine->loadGameClockTransitionMatrix(filename.toStdString())) {
					blog(LOG_INFO, "Game clock transition matrix loaded successfully");
				} else {
					blog(LOG_ERROR, "Failed to load game clock transition matrix");
				}
			}
#endif
		}
	}
	
	void onSmoothingFramesChanged(int frames) {
#ifdef USE_CNN_OCR
		if (ocrEngine) {
			ocrEngine->setSmoothingFrames(frames);
			blog(LOG_INFO, "Smoothing frames set to: %d", frames);
		}
#endif
	}
	
	void loadCNNModels() {
#ifdef USE_CNN_OCR
		QString shotModel = shotClockModelEdit->text();
		QString gameModel = gameClockModelEdit->text();
		
		bool shotOk = false, gameOk = false;
		QString message;
		
		if (!shotModel.isEmpty() && QFile::exists(shotModel)) {
			shotOk = ocrEngine->loadShotClockModel(shotModel.toStdString());
			if (shotOk) {
				blog(LOG_INFO, "Shot clock model loaded successfully: %s", shotModel.toUtf8().constData());
			} else {
				blog(LOG_ERROR, "Failed to load shot clock model: %s", shotModel.toUtf8().constData());
			}
		}
		
		if (!gameModel.isEmpty() && QFile::exists(gameModel)) {
			gameOk = ocrEngine->loadGameClockModel(gameModel.toStdString());
			if (gameOk) {
				blog(LOG_INFO, "Game clock model loaded successfully: %s", gameModel.toUtf8().constData());
			} else {
				blog(LOG_ERROR, "Failed to load game clock model: %s", gameModel.toUtf8().constData());
			}
		}
		
		// Enable Bayesian filtering with Markov model (enabled by default, but ensure it's on)
		if (shotOk || gameOk) {
			ocrEngine->enableBayesianFiltering(true);
			blog(LOG_INFO, "Bayesian filtering with Markov model enabled for multi-frame averaging");
		}
		
		if (shotOk && gameOk) {
			message = "✓ Both CNN models loaded successfully!\n\n"
			          "Bayesian filtering enabled:\n"
			          "• Markov transition matrices\n"
			          "• Multi-frame temporal smoothing\n"
			          "• Handles blocked/obscured frames";
			// Enable ROI selector buttons
			selectShotRoiBtn->setEnabled(true);
			selectGameRoiBtn->setEnabled(true);
			QMessageBox::information(this, "CNN Models", message);
		} else if (shotOk || gameOk) {
			message = QString("⚠ Partial success:\n%1%2\n\nBayesian filtering enabled for loaded model(s)")
				.arg(shotOk ? "✓ Shot clock model loaded\n" : "✗ Shot clock model failed\n")
				.arg(gameOk ? "✓ Game clock model loaded" : "✗ Game clock model failed");
			if (shotOk) {
				selectShotRoiBtn->setEnabled(true);
			}
			if (gameOk) {
				selectGameRoiBtn->setEnabled(true);
			}
			QMessageBox::warning(this, "CNN Models", message);
		} else {
			message = "✗ Failed to load CNN models.\nPlease check the file paths.";
			QMessageBox::critical(this, "CNN Models", message);
		}
#else
		QMessageBox msgBox(this);
		msgBox.setWindowTitle("CNN Support Not Available");
		msgBox.setIcon(QMessageBox::Information);
		msgBox.setText("CNN clock detection is not compiled in this build.");
		msgBox.setInformativeText(
			"To enable automatic clock detection:\n\n"
			"1. Download LibTorch (PyTorch C++):\n"
			"   https://pytorch.org/get-started/locally/\n"
			"   Select: LibTorch, Windows, C++/Java, CPU or CUDA\n"
			"   Extract to C:\\libtorch\n\n"
			"2. Install OpenCV with vcpkg:\n"
			"   vcpkg install opencv:x64-windows\n\n"
			"3. Rebuild the plugin:\n"
			"   cd obs-scoreboard\n"
			"   .\\build-direct.ps1\n\n"
			"The build script will auto-detect LibTorch and OpenCV."
		);
		msgBox.setDetailedText(
			"Current model paths:\n"
			"Shot Clock: " + shotClockModelEdit->text() + "\n"
			"Game Clock: " + gameClockModelEdit->text() + "\n\n"
			"These paths are saved and will be used once CNN support is compiled."
		);
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.exec();
#endif
	}
	
	void selectShotClockRoi() {
#ifdef USE_CNN_OCR
		// Get list of video sources from OBS
		QStringList videoSourceNames;
		QMap<QString, obs_source_t*> videoSourceMap;
		
		auto enumSources = [](void* param, obs_source_t* source) -> bool {
			auto* data = static_cast<QPair<QStringList*, QMap<QString, obs_source_t*>*>*>(param);
			const char* id = obs_source_get_id(source);
			const char* name = obs_source_get_name(source);
			
			// Check if it's a video source (not just audio)
			uint32_t flags = obs_source_get_output_flags(source);
			if (id && name && (flags & OBS_SOURCE_VIDEO)) {
				QString sourceName = QString::fromUtf8(name);
				data->first->append(sourceName);
				obs_source_get_ref(source);
				data->second->insert(sourceName, source);
			}
			return true;
		};
		
		QPair<QStringList*, QMap<QString, obs_source_t*>*> enumData(&videoSourceNames, &videoSourceMap);
		obs_enum_sources(enumSources, &enumData);
		
		if (videoSourceNames.isEmpty()) {
			QMessageBox::warning(this, "No Video Sources", 
				"No video sources found in OBS.\n\n"
				"Please add a video capture device to your OBS scene first.");
			return;
		}
		
		// Show source selection dialog
		bool ok;
		QString selectedSource = QInputDialog::getItem(this, 
			"Select Video Source",
			"Choose a video source to capture frame from:",
			videoSourceNames, 0, false, &ok);
		
		if (!ok || selectedSource.isEmpty()) {
			// Release all source references
			for (obs_source_t* source : videoSourceMap.values()) {
				obs_source_release(source);
			}
			return;
		}
		
		obs_source_t* source = videoSourceMap[selectedSource];
		
		// Capture a frame from the source using our helper function
		QImage frame = captureFrameFromOBSSource(source);
		
		// Release source references
		for (obs_source_t* src : videoSourceMap.values()) {
			obs_source_release(src);
		}
		
		if (frame.isNull()) {
			QMessageBox::warning(this, "Capture Failed", 
				"Failed to capture frame from video source.\n"
				"Make sure the source is active and visible.");
			return;
		}
		
		// Create ROI selector dialog with the captured frame
		ROISelectorDialog* dialog = new ROISelectorDialog(this);
		dialog->setWindowTitle("Select Shot Clock ROI - " + selectedSource);
		
		// Load existing ROI if available
		QSettings roiSettings("WaterPoloScoreboard", "CNNModels");
		ROI existingRoi;
		existingRoi.x = roiSettings.value("shotClockROI_x", 0).toInt();
		existingRoi.y = roiSettings.value("shotClockROI_y", 0).toInt();
		existingRoi.width = roiSettings.value("shotClockROI_width", 0).toInt();
		existingRoi.height = roiSettings.value("shotClockROI_height", 0).toInt();
		if (existingRoi.width > 0 && existingRoi.height > 0) {
			dialog->setShotClockROI(existingRoi);
		}
		
		// Set the captured frame (dialog will use this instead of live camera)
		dialog->getCanvas()->setFrame(frame);
		dialog->getCanvas()->setSelectionMode("shot");
		
		// Hide camera controls since we're using a static frame
		// Hide camera controls not needed for static frame
		
		
		// Show dialog
		if (dialog->exec() == QDialog::Accepted) {
			ROI shotRoi = dialog->getShotClockROI();
			
			if (shotRoi.width > 0 && shotRoi.height > 0) {
				roiSettings.setValue("shotClockROI_x", shotRoi.x);
				roiSettings.setValue("shotClockROI_y", shotRoi.y);
				roiSettings.setValue("shotClockROI_width", shotRoi.width);
				roiSettings.setValue("shotClockROI_height", shotRoi.height);
				roiSettings.setValue("shotClockROI_source", selectedSource);
				
				// Save source name for detection
				shotClockSourceName = selectedSource;
				
				// Set ROI in OCR engine
				ocrEngine->setShotClockROI(shotRoi.x, shotRoi.y, shotRoi.width, shotRoi.height);
				
				// Check if we can enable detection (both ROIs set)
				checkEnableDetection();
				
				QMessageBox::information(this, "ROI Saved", 
					QString("Shot Clock ROI saved:\n"
					        "Source: %1\n"
					        "X: %2, Y: %3\n"
					        "Width: %4, Height: %5")
						.arg(selectedSource)
						.arg(shotRoi.x).arg(shotRoi.y)
						.arg(shotRoi.width).arg(shotRoi.height));
				
				blog(LOG_INFO, "Shot Clock ROI saved from source '%s': x=%d, y=%d, w=%d, h=%d", 
					selectedSource.toUtf8().constData(),
					shotRoi.x, shotRoi.y, shotRoi.width, shotRoi.height);
			}
		}
		
		delete dialog;
#else
		QMessageBox::information(this, "CNN Not Available", 
			"CNN clock detection is not compiled in this build.\n"
			"ROI selection requires CNN support.");
#endif
	}
	
	void selectGameClockRoi() {
#ifdef USE_CNN_OCR
		// Get list of video sources from OBS
		QStringList videoSourceNames;
		QMap<QString, obs_source_t*> videoSourceMap;
		
		auto enumSources = [](void* param, obs_source_t* source) -> bool {
			auto* data = static_cast<QPair<QStringList*, QMap<QString, obs_source_t*>*>*>(param);
			const char* id = obs_source_get_id(source);
			const char* name = obs_source_get_name(source);
			
			// Check if it's a video source (not just audio)
			uint32_t flags = obs_source_get_output_flags(source);
			if (id && name && (flags & OBS_SOURCE_VIDEO)) {
				QString sourceName = QString::fromUtf8(name);
				data->first->append(sourceName);
				obs_source_get_ref(source);
				data->second->insert(sourceName, source);
			}
			return true;
		};
		
		QPair<QStringList*, QMap<QString, obs_source_t*>*> enumData(&videoSourceNames, &videoSourceMap);
		obs_enum_sources(enumSources, &enumData);
		
		if (videoSourceNames.isEmpty()) {
			QMessageBox::warning(this, "No Video Sources", 
				"No video sources found in OBS.\n\n"
				"Please add a video capture device to your OBS scene first.");
			return;
		}
		
		// Show source selection dialog
		bool ok;
		QString selectedSource = QInputDialog::getItem(this, 
			"Select Video Source",
			"Choose a video source to capture frame from:",
			videoSourceNames, 0, false, &ok);
		
		if (!ok || selectedSource.isEmpty()) {
			// Release all source references
			for (obs_source_t* source : videoSourceMap.values()) {
				obs_source_release(source);
			}
			return;
		}
		
		obs_source_t* source = videoSourceMap[selectedSource];
		
		// Capture a frame from the source using our helper function
		QImage frame = captureFrameFromOBSSource(source);
		
		// Release source references
		for (obs_source_t* src : videoSourceMap.values()) {
			obs_source_release(src);
		}
		
		if (frame.isNull()) {
			QMessageBox::warning(this, "Capture Failed", 
				"Failed to capture frame from video source.\n"
				"Make sure the source is active and visible.");
			return;
		}
		
		// Create ROI selector dialog with the captured frame
		ROISelectorDialog* dialog = new ROISelectorDialog(this);
		dialog->setWindowTitle("Select Game Clock ROI - " + selectedSource);
		
		// Load existing ROI if available
		QSettings roiSettings("WaterPoloScoreboard", "CNNModels");
		ROI existingRoi;
		existingRoi.x = roiSettings.value("gameClockROI_x", 0).toInt();
		existingRoi.y = roiSettings.value("gameClockROI_y", 0).toInt();
		existingRoi.width = roiSettings.value("gameClockROI_width", 0).toInt();
		existingRoi.height = roiSettings.value("gameClockROI_height", 0).toInt();
		if (existingRoi.width > 0 && existingRoi.height > 0) {
			dialog->setGameClockROI(existingRoi);
		}
		
		// Set the captured frame (dialog will use this instead of live camera)
		dialog->getCanvas()->setFrame(frame);
		dialog->getCanvas()->setSelectionMode("game");
		
		// Hide camera controls since we're using a static frame
		// Hide camera controls not needed for static frame
		
		
		// Show dialog
		if (dialog->exec() == QDialog::Accepted) {
			ROI gameRoi = dialog->getGameClockROI();
			
			if (gameRoi.width > 0 && gameRoi.height > 0) {
				roiSettings.setValue("gameClockROI_x", gameRoi.x);
				roiSettings.setValue("gameClockROI_y", gameRoi.y);
				roiSettings.setValue("gameClockROI_width", gameRoi.width);
				roiSettings.setValue("gameClockROI_height", gameRoi.height);
				roiSettings.setValue("gameClockROI_source", selectedSource);
				
				// Save source name for detection
				gameClockSourceName = selectedSource;
				
				// Set ROI in OCR engine
				ocrEngine->setGameClockROI(gameRoi.x, gameRoi.y, gameRoi.width, gameRoi.height);
				
				// Check if we can enable detection (both ROIs set)
				checkEnableDetection();
				
				QMessageBox::information(this, "ROI Saved", 
					QString("Game Clock ROI saved:\n"
					        "Source: %1\n"
					        "X: %2, Y: %3\n"
					        "Width: %4, Height: %5")
						.arg(selectedSource)
						.arg(gameRoi.x).arg(gameRoi.y)
						.arg(gameRoi.width).arg(gameRoi.height));
				
				blog(LOG_INFO, "Game Clock ROI saved from source '%s': x=%d, y=%d, w=%d, h=%d", 
					selectedSource.toUtf8().constData(),
					gameRoi.x, gameRoi.y, gameRoi.width, gameRoi.height);
			}
		}
		
		delete dialog;
#else
		QMessageBox::information(this, "CNN Not Available", 
			"CNN clock detection is not compiled in this build.\n"
			"ROI selection requires CNN support.");
#endif
	}
	
	void updateColorButtons() {
		QString homeStyle = QString("background-color: rgb(%1, %2, %3);")
			.arg((homeColor >> 16) & 0xFF)
			.arg((homeColor >> 8) & 0xFF)
			.arg(homeColor & 0xFF);
		homeColorBtn->setStyleSheet(homeStyle);
		
		QString awayStyle = QString("background-color: rgb(%1, %2, %3);")
			.arg((awayColor >> 16) & 0xFF)
			.arg((awayColor >> 8) & 0xFF)
			.arg(awayColor & 0xFF);
		awayColorBtn->setStyleSheet(awayStyle);
	}
	
#ifdef USE_CNN_OCR
	void checkEnableDetection() {
		// Enable detection button if both ROIs are set
		QSettings roiSettings("WaterPoloScoreboard", "CNNModels");
		bool shotRoiSet = roiSettings.value("shotClockROI_width", 0).toInt() > 0;
		bool gameRoiSet = roiSettings.value("gameClockROI_width", 0).toInt() > 0;
		
		if (shotRoiSet && gameRoiSet && !shotClockSourceName.isEmpty() && !gameClockSourceName.isEmpty()) {
			startDetectionBtn->setEnabled(true);
		}
	}
	
	void startClockDetection() {
		blog(LOG_INFO, "startClockDetection() called");
		
		if (detectionRunning) {
			blog(LOG_INFO, "Detection already running, ignoring");
			return;
		}
		
		blog(LOG_INFO, "Looking for sources: shot='%s', game='%s'", 
			shotClockSourceName.toUtf8().constData(),
			gameClockSourceName.toUtf8().constData());
		
		// Get the OBS sources
		shotClockVideoSource = obs_get_source_by_name(shotClockSourceName.toUtf8().constData());
		gameClockVideoSource = obs_get_source_by_name(gameClockSourceName.toUtf8().constData());
		
		if (!shotClockVideoSource || !gameClockVideoSource) {
			blog(LOG_WARNING, "Source lookup failed: shot=%p, game=%p", 
				shotClockVideoSource, gameClockVideoSource);
			QMessageBox::warning(this, "Source Not Found",
				"Could not find one or more video sources.\n"
				"Please reconfigure the ROIs.");
			if (shotClockVideoSource) obs_source_release(shotClockVideoSource);
			if (gameClockVideoSource) obs_source_release(gameClockVideoSource);
			shotClockVideoSource = nullptr;
			gameClockVideoSource = nullptr;
			return;
		}
		
		blog(LOG_INFO, "Sources found successfully, creating timer");
		
		// Create and start update timer at 30 FPS to match CNN training conditions
		// This ensures frames are captured at the same rate as training data
		ocrUpdateTimer = new QTimer(this);
		connect(ocrUpdateTimer, &QTimer::timeout, this, &ScoreboardControlPanel::updateClocksFromOCR);
		ocrUpdateTimer->start(33); // 30 FPS (~33ms per frame)
		
		detectionRunning = true;
		startDetectionBtn->setEnabled(false);
		stopDetectionBtn->setEnabled(true);
		
		blog(LOG_INFO, "Started clock detection: shot=%s, game=%s", 
			shotClockSourceName.toUtf8().constData(),
			gameClockSourceName.toUtf8().constData());
	}
	
	void stopClockDetection() {
		if (!detectionRunning) return;
		
		if (ocrUpdateTimer) {
			ocrUpdateTimer->stop();
			delete ocrUpdateTimer;
			ocrUpdateTimer = nullptr;
		}
		
		if (shotClockVideoSource) {
			obs_source_release(shotClockVideoSource);
			shotClockVideoSource = nullptr;
		}
		
		if (gameClockVideoSource) {
			obs_source_release(gameClockVideoSource);
			gameClockVideoSource = nullptr;
		}
		
		detectionRunning = false;
		startDetectionBtn->setEnabled(true);
		stopDetectionBtn->setEnabled(false);
		
		blog(LOG_INFO, "Stopped clock detection");
	}
	
	void updateClocksFromOCR() {
		// Capture frames from both sources (they might be the same source)
		// NOTE: This runs at 10 FPS, and each frame is processed through:
		// 1. CNN model to get raw digit probabilities
		// 2. Bayesian filter with Markov transition matrix for temporal smoothing
		// 3. Multi-frame averaging over time (30 FPS model, running at 10 FPS capture)
		// This approach handles:
		// - Noisy/uncertain predictions
		// - Blocked/obscured frames
		// - Natural clock transitions (counting down)
		// - Reset events (30->29, 24->23, etc.)
		
		QImage shotClockFrame = captureFrameFromOBSSource(shotClockVideoSource);
		QImage gameClockFrame;
		
		// If using same source for both, reuse the frame
		if (shotClockSourceName == gameClockSourceName && shotClockVideoSource == gameClockVideoSource) {
			gameClockFrame = shotClockFrame;
		} else {
			gameClockFrame = captureFrameFromOBSSource(gameClockVideoSource);
		}
		
		if (shotClockFrame.isNull() || gameClockFrame.isNull()) {
			// Skip this update if frames couldn't be captured
			return;
		}
		
		// Convert QImage to cv::Mat
		cv::Mat shotMat = qImageToMat(shotClockFrame);
		cv::Mat gameMat = qImageToMat(gameClockFrame);
		
		// Set frames for frame buffering (separate buffers for each clock)
		// This allows different sources to be averaged independently
		ocrEngine->setShotClockFrame(shotMat);
		ocrEngine->setGameClockFrame(gameMat);
		
		// Get ROI rectangles
		cv::Rect shotRoi = ocrEngine->getShotClockROI();
		cv::Rect gameRoi = ocrEngine->getGameClockROI();
		
		// Store predictions for histogram visualization
		ClockPrediction shotPred, gamePred;
		
		// Process shot clock with Bayesian filtering
		if (shotRoi.x >= 0 && shotRoi.y >= 0 && 
		    shotRoi.x + shotRoi.width <= shotMat.cols &&
		    shotRoi.y + shotRoi.height <= shotMat.rows) {
			
			// Get CNN prediction with Bayesian filtering (frame already set above)
			// This internally:
			// - Runs CNN to get digit probabilities
			// - Applies Markov transition matrix (time evolution)
			// - Updates posterior probability distribution
			// - Returns most likely value with confidence
			shotPred = ocrEngine->predictShotClock();
			
			if (shotPred.confidence > 0.7) {
				try {
					// Update shot clock display
					int shotValue = std::stoi(shotPred.value);
					shotClockSpin->setValue(shotValue);
				} catch (...) {
					// Ignore parse errors
				}
			}
		}
		
		// Process game clock with Bayesian filtering
		if (gameRoi.x >= 0 && gameRoi.y >= 0 &&
		    gameRoi.x + gameRoi.width <= gameMat.cols &&
		    gameRoi.y + gameRoi.height <= gameMat.rows) {
			
			// Get CNN prediction with Bayesian filtering (frame already set above)
			// Game clock filter handles 0:00 to 7:59 (480 states)
			// with appropriate transition probabilities for quarter/period endings
			gamePred = ocrEngine->predictGameClock();
			
			if (gamePred.confidence > 0.7) {
				try {
					// Parse MM:SS format
					size_t colonPos = gamePred.value.find(':');
					if (colonPos != std::string::npos) {
						int minutes = std::stoi(gamePred.value.substr(0, colonPos));
						int seconds = std::stoi(gamePred.value.substr(colonPos + 1));
						
						gameMinutesSpin->setValue(minutes);
						gameSecondsSpin->setValue(seconds);
					}
				} catch (...) {
					// Ignore parse errors
				}
			}
		}
		
		// Update histogram visualization only when we have fresh CNN data
		// This prevents flickering from repeated updates with cached predictions
		if (shotPred.is_fresh_cnn || gamePred.is_fresh_cnn) {
			update_histogram_viz_data(shotPred, gamePred);
			
			// Also update averaged frame visualization
			cv::Mat shot_averaged = ocrEngine->getShotClockAveragedFrame();
			cv::Mat game_averaged = ocrEngine->getGameClockAveragedFrame();
			update_averaged_frame_viz_data(shot_averaged, game_averaged);
		}
		
		// Update the scoreboard
		updateScoreboard();
	}
	
	cv::Mat qImageToMat(const QImage& image) {
		// Convert QImage to cv::Mat
		cv::Mat mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
		cv::Mat result;
		cv::cvtColor(mat, result, cv::COLOR_RGBA2BGR);
		return result;
	}
#endif
};

#include "control-panel.moc"

// Global control panel instance
static ScoreboardControlPanel *g_control_panel = nullptr;

void init_control_panel()
{
	if (!g_control_panel) {
		// Create control panel as a dock widget
		QMainWindow *main_window = (QMainWindow *)obs_frontend_get_main_window();
		if (main_window) {
			g_control_panel = new ScoreboardControlPanel();
			
			QAction *action = (QAction *)obs_frontend_add_tools_menu_qaction("Water Polo Scoreboard Control");
			action->connect(action, &QAction::triggered, [=]() {
				g_control_panel->show();
				g_control_panel->raise();
			});
			
			blog(LOG_INFO, "Control panel initialized");
		}
	}
}

void shutdown_control_panel()
{
	if (g_control_panel) {
		g_control_panel->hide();
		delete g_control_panel;
		g_control_panel = nullptr;
	}
}
