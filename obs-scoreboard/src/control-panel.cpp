#include <mutex>

#include "scoreboard-source.h"

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
#include <QSettings>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QCheckBox>
#include <QDir>
#include <QMessageBox>
#include <QScrollArea>
#include <QInputDialog>
#include <QMutex>
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
#ifdef __APPLE__
#undef NO
#undef YES
#endif
#include <opencv2/opencv.hpp>
#pragma pop_macro("slots")
#endif

#include "shared-schedule.h"

// Forward declaration from scoreboard-source.cpp
void update_scoreboard_data(obs_data_t *data);

// ── Frame capture via obs_enter_graphics ─────────────────────────────────────
//
// obs_enter_graphics() acquires OBS's graphics mutex, guaranteeing we are
// between render frames (no active Metal/GL render encoder). This makes it
// safe to call obs_source_video_render() and gs_stagesurface_map() from any
// thread — including the Qt main thread and our own capture thread.
//
// Capture source → BGRA → RGBA bytes. Caller owns obs_enter_graphics context.
static bool captureSourceToBytes(obs_source_t *source,
                                  std::vector<uint8_t> &out,
                                  uint32_t &outW, uint32_t &outH)
{
	uint32_t w = obs_source_get_width(source);
	uint32_t h = obs_source_get_height(source);
	if (w == 0 || h == 0) return false;

	gs_texrender_t *tr = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	if (!tr) return false;

	bool ok = false;
	gs_texrender_reset(tr);
	if (gs_texrender_begin(tr, w, h)) {
		struct vec4 clr; vec4_zero(&clr);
		gs_clear(GS_CLEAR_COLOR, &clr, 0.0f, 0);
		gs_ortho(0.0f, (float)w, 0.0f, (float)h, -100.0f, 100.0f);
		obs_source_video_render(source);
		gs_texrender_end(tr);

		gs_texture_t *tex = gs_texrender_get_texture(tr);
		if (tex) {
			gs_stagesurf_t *ss = gs_stagesurface_create(w, h, GS_BGRA);
			if (ss) {
				gs_stage_texture(ss, tex);
				uint8_t *ptr = nullptr; uint32_t ls = 0;
				if (gs_stagesurface_map(ss, &ptr, &ls)) {
					out.resize(w * h * 4);
					for (uint32_t y = 0; y < h; y++) {
						const uint8_t *src = ptr + y * ls;
						uint8_t *dst = out.data() + y * w * 4;
						for (uint32_t x = 0; x < w; x++) {
							dst[x*4+0] = src[x*4+2]; // BGRA→RGBA
							dst[x*4+1] = src[x*4+1];
							dst[x*4+2] = src[x*4+0];
							dst[x*4+3] = src[x*4+3];
						}
					}
					gs_stagesurface_unmap(ss);
					outW = w; outH = h;
					ok = true;
				}
				gs_stagesurface_destroy(ss);
			}
		}
	}
	gs_texrender_destroy(tr);
	return ok;
}

// One-shot capture from any thread (e.g. Qt main thread for ROI selector).
static QImage captureFrameFromOBSSource(obs_source_t *source) {
	if (!source) return QImage();

	std::vector<uint8_t> bytes;
	uint32_t w = 0, h = 0;

	obs_enter_graphics();
	bool ok = captureSourceToBytes(source, bytes, w, h);
	obs_leave_graphics();

	if (!ok || bytes.empty()) return QImage();

	// Build QImage on the calling thread (safe — no GPU work here)
	QImage img(w, h, QImage::Format_RGBA8888);
	memcpy(img.bits(), bytes.data(), bytes.size());
	return img;
}

// ============================================================================
// TeamColorEditorDialog — edit team colors and write back to teams.csv
// ============================================================================
class TeamColorEditorDialog : public QDialog {
	Q_OBJECT

signals:
	// Emitted immediately after any color is changed so the scoreboard can update live.
	void teamColorChanged(const QString& teamName,
	                      const QColor& home_bg, const QColor& home_text,
	                      const QColor& away_bg, const QColor& away_text);

private:
	struct TeamEntry {
		QString name;
		QColor home_bg, home_text, away_bg, away_text;
	};

	QComboBox*   teamCombo;
	QPushButton* homeBgBtn;
	QPushButton* homeTextBtn;
	QPushButton* awayBgBtn;
	QPushButton* awayTextBtn;
	QLabel*      previewHome;
	QLabel*      previewAway;

	QString          csvPath;
	QList<TeamEntry> teams;

	// ---------- helpers ----------

	static void setSwatchStyle(QPushButton* btn, const QColor& bg, const QColor& fg) {
		btn->setText(bg.name().toUpper());
		btn->setStyleSheet(
			QString("QPushButton { background-color: %1; color: %2; padding: 4px 8px; }")
			.arg(bg.name()).arg(fg.name()));
	}

	TeamEntry& current() { return teams[teamCombo->currentIndex()]; }

	void loadCsv() {
		QFile f(csvPath);
		if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return;
		QTextStream in(&f);
		in.readLine(); // header
		while (!in.atEnd()) {
			QString line = in.readLine().trimmed();
			if (line.isEmpty()) continue;
			QStringList p = line.split(',');
			if (p.size() < 5) continue;
			TeamEntry e;
			e.name      = p[0].trimmed();
			e.home_bg   = QColor(p[1].trimmed());
			e.home_text = QColor(p[2].trimmed());
			e.away_bg   = QColor(p[3].trimmed());
			e.away_text = QColor(p[4].trimmed());
			teams.append(e);
		}
	}

	void saveCsv() {
		QFile f(csvPath);
		if (!f.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
			QMessageBox::critical(this, "Error",
				QString("Could not write to %1").arg(csvPath));
			return;
		}
		QTextStream out(&f);
		out << "name,home_bg,home_text,away_bg,away_text\n";
		for (const TeamEntry& e : teams) {
			out << e.name << ","
			    << e.home_bg.name().toUpper() << ","
			    << e.home_text.name().toUpper() << ","
			    << e.away_bg.name().toUpper() << ","
			    << e.away_text.name().toUpper() << "\n";
		}
	}

	void refreshSwatches() {
		const TeamEntry& e = teams[teamCombo->currentIndex()];
		setSwatchStyle(homeBgBtn,   e.home_bg,   e.home_text);
		setSwatchStyle(homeTextBtn, e.home_text,  e.home_bg);
		setSwatchStyle(awayBgBtn,   e.away_bg,   e.away_text);
		setSwatchStyle(awayTextBtn, e.away_text,  e.away_bg);

		previewHome->setStyleSheet(
			QString("QLabel { background-color: %1; color: %2; padding: 8px; font-weight: bold; border-radius: 4px; }")
			.arg(e.home_bg.name()).arg(e.home_text.name()));
		previewHome->setText("HOME  " + e.name);

		previewAway->setStyleSheet(
			QString("QLabel { background-color: %1; color: %2; padding: 8px; font-weight: bold; border-radius: 4px; }")
			.arg(e.away_bg.name()).arg(e.away_text.name()));
		previewAway->setText("AWAY  " + e.name);
	}

	void pickColor(QColor& target, QPushButton* btn, const QString& title, bool isBg) {
		QColor original = target;

		QColorDialog dlg(target, this);
		dlg.setWindowTitle(title);
		// DontUseNativeDialog is required: the macOS native picker doesn't emit
		// currentColorChanged while dragging, so live preview wouldn't work.
		dlg.setOption(QColorDialog::DontUseNativeDialog);

		auto applyColor = [&](const QColor& color) {
			target = color;
			QColor pair = isBg ? teams[teamCombo->currentIndex()].home_text
			                   : teams[teamCombo->currentIndex()].home_bg;
			setSwatchStyle(btn, target, pair);
			refreshSwatches();
			const TeamEntry& e = teams[teamCombo->currentIndex()];
			emit teamColorChanged(e.name, e.home_bg, e.home_text, e.away_bg, e.away_text);
		};

		connect(&dlg, &QColorDialog::currentColorChanged, this, applyColor);

		if (dlg.exec() != QDialog::Accepted) {
			// Restore original color on cancel
			applyColor(original);
		}
	}

	void setupUI() {
		setWindowTitle("Edit Team Colors");
		setMinimumWidth(460);

		QVBoxLayout* layout = new QVBoxLayout(this);

		// --- Team selector ---
		QHBoxLayout* row = new QHBoxLayout();
		row->addWidget(new QLabel("Team:"));
		teamCombo = new QComboBox();
		for (const TeamEntry& e : teams)
			teamCombo->addItem(e.name);
		row->addWidget(teamCombo, 1);
		layout->addLayout(row);

		// --- Color grid ---
		QGroupBox* colorBox = new QGroupBox("Colors");
		QGridLayout* grid = new QGridLayout(colorBox);

		auto makeLabel = [](const QString& txt) {
			QLabel* l = new QLabel(txt);
			l->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
			return l;
		};

		grid->addWidget(makeLabel("Home background:"), 0, 0);
		homeBgBtn = new QPushButton(); homeBgBtn->setMinimumWidth(130);
		grid->addWidget(homeBgBtn, 0, 1);

		grid->addWidget(makeLabel("Home text:"), 1, 0);
		homeTextBtn = new QPushButton(); homeTextBtn->setMinimumWidth(130);
		grid->addWidget(homeTextBtn, 1, 1);

		grid->addWidget(makeLabel("Away background:"), 2, 0);
		awayBgBtn = new QPushButton(); awayBgBtn->setMinimumWidth(130);
		grid->addWidget(awayBgBtn, 2, 1);

		grid->addWidget(makeLabel("Away text:"), 3, 0);
		awayTextBtn = new QPushButton(); awayTextBtn->setMinimumWidth(130);
		grid->addWidget(awayTextBtn, 3, 1);

		layout->addWidget(colorBox);

		// --- Preview ---
		QGroupBox* previewBox = new QGroupBox("Preview");
		QVBoxLayout* pv = new QVBoxLayout(previewBox);
		previewHome = new QLabel(); previewHome->setAlignment(Qt::AlignCenter); previewHome->setMinimumHeight(36);
		previewAway = new QLabel(); previewAway->setAlignment(Qt::AlignCenter); previewAway->setMinimumHeight(36);
		pv->addWidget(previewHome);
		pv->addWidget(previewAway);
		layout->addWidget(previewBox);

		// --- Buttons ---
		QHBoxLayout* btns = new QHBoxLayout();
		btns->addStretch();
		QPushButton* saveBtn = new QPushButton("Save to CSV");
		saveBtn->setStyleSheet("QPushButton { background-color: #2d7a2d; color: white; padding: 6px 16px; font-weight: bold; }");
		QPushButton* cancelBtn = new QPushButton("Cancel");
		cancelBtn->setStyleSheet("QPushButton { padding: 6px 16px; }");
		btns->addWidget(saveBtn);
		btns->addWidget(cancelBtn);
		layout->addLayout(btns);

		// --- Connections ---
		connect(teamCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
		        this, [this]{ refreshSwatches(); });

		connect(homeBgBtn, &QPushButton::clicked, this, [this]{
			pickColor(current().home_bg, homeBgBtn, "Home Background Color", true);
		});
		connect(homeTextBtn, &QPushButton::clicked, this, [this]{
			pickColor(current().home_text, homeTextBtn, "Home Text Color", false);
		});
		connect(awayBgBtn, &QPushButton::clicked, this, [this]{
			pickColor(current().away_bg, awayBgBtn, "Away Background Color", true);
		});
		connect(awayTextBtn, &QPushButton::clicked, this, [this]{
			pickColor(current().away_text, awayTextBtn, "Away Text Color", false);
		});

		connect(saveBtn, &QPushButton::clicked, this, [this]{
			saveCsv();
			accept();
		});
		connect(cancelBtn, &QPushButton::clicked, this, &QDialog::reject);

		if (!teams.isEmpty()) refreshSwatches();
	}

public:
	explicit TeamColorEditorDialog(const QString& csvPath, QWidget* parent = nullptr)
		: QDialog(parent), csvPath(csvPath)
	{
		loadCsv();
		setupUI();
	}
};

// ============================================================================

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
	QSpinBox *defaultQuarterMinutesSpin;
	QSpinBox *defaultQuarterSecondsSpin;
	
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
	bool clockSyncGuard = false; // prevents re-entrancy when syncing clocks
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
	QPushButton *editTeamColorsBtn;
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
	// Raw video callback stores CPU-side RGBA frames (no GPU ops needed).
	// OBS pushes frames on its video output thread; Qt timer reads them.
	std::mutex             rawVideoMutex;
	std::vector<uint8_t>   rawVideoBytes; // RGBA8888
	uint32_t               rawVideoW = 0, rawVideoH = 0;

	// --- Clock sync state ---
	static constexpr int kPauseFrameThreshold = 20; // ~2 s at 10 Hz OCR
	int  shotOCRLast    = -1;
	int  shotOCRSameRun = 0;
	bool shotOCRPaused  = false;
	int  gameOCRLast    = -1; // total seconds
	int  gameOCRSameRun = 0;
	bool gameOCRPaused  = false;

	// Sync mode:
	//   0 = Event-based  — trust 1 s timer during play, sync only on stop/start
	//   1 = Rate-based   — 50 ms internal timer, adjust tick rate to converge to OCR
	int          clockSyncMode   = 0;
	QComboBox   *clockSyncModeCombo = nullptr;

	// Rate-based sync: high-frequency timers + float clocks
	QTimer      *shotRateTimer   = nullptr;
	QTimer      *gameRateTimer   = nullptr;
	double       shotClockMs     = 0.0;  // milliseconds remaining
	double       gameClockMs     = 0.0;
	double       shotClockRate   = 1.0;  // 1.0 = real-time
	double       gameClockRate   = 1.0;
	static constexpr double kRateConvergenceMs = 4000.0; // converge over ~4 s
	static constexpr int    kRateTickMs        = 50;     // 50 ms = 20 Hz
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
	Q_INVOKABLE void nextGame() {
		int cur = gameSelectCombo->currentIndex();
		if (cur > 0 && cur < gameSelectCombo->count() - 1)
			gameSelectCombo->setCurrentIndex(cur + 1);
	}

	Q_INVOKABLE void prevGame() {
		int cur = gameSelectCombo->currentIndex();
		if (cur > 1)
			gameSelectCombo->setCurrentIndex(cur - 1);
	}

	// ── Remote API (called by WebSocket handler, always on Qt main thread) ───

	QJsonObject getScheduleJson() {
		QJsonArray games;
		if (configDir.isEmpty()) {
			QJsonObject r; r["error"] = "No schedule loaded"; return r;
		}
		QFile file(configDir + "/schedule.csv");
		if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
			QJsonObject r; r["error"] = "Could not open schedule.csv"; return r;
		}
		QTextStream in(&file);
		in.readLine(); // skip header
		int index = 0;
		while (!in.atEnd()) {
			QString line = in.readLine().trimmed();
			if (line.isEmpty()) { index++; continue; }
			QStringList p = line.split(',');
			QJsonObject g;
			g["index"]      = index++;
			g["start_time"] = p.value(0).trimmed();
			g["home"]       = p.value(1).trimmed();
			g["away"]       = p.value(2).trimmed();
			QString hs = p.value(3).trimmed(), as_ = p.value(4).trimmed();
			g["home_score"] = hs.isEmpty()  ? QJsonValue(QJsonValue::Null) : QJsonValue(hs.toInt());
			g["away_score"] = as_.isEmpty() ? QJsonValue(QJsonValue::Null) : QJsonValue(as_.toInt());
			g["winner"]     = p.value(5).trimmed();
			games.append(g);
		}
		QJsonObject result;
		result["games"] = games;
		return result;
	}

	void setGameScoreAtIndex(int gameIndex, int homeScore, int awayScore, const QString &winner) {
		if (configDir.isEmpty()) return;
		QString schedulePath = configDir + "/schedule.csv";
		QFile file(schedulePath);
		if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return;

		QStringList lines;
		QTextStream in(&file);
		QString header = in.readLine();
		if (!header.split(',').contains("home_score"))
			header += ",home_score,away_score,winner";
		lines.append(header);

		int idx = 0;
		while (!in.atEnd()) {
			QString line = in.readLine();
			if (line.trimmed().isEmpty()) { lines.append(line); continue; }
			QStringList p = line.split(',');
			if (idx == gameIndex && p.size() >= 3) {
				lines.append(p.value(0).trimmed() + "," + p.value(1).trimmed() + "," +
				             p.value(2).trimmed() + "," + QString::number(homeScore) +
				             "," + QString::number(awayScore) + "," + winner);
			} else {
				lines.append(line);
			}
			idx++;
		}
		file.close();

		QFile out(schedulePath);
		if (!out.open(QIODevice::WriteOnly | QIODevice::Text)) return;
		QTextStream outStream(&out);
		for (const QString &l : lines) outStream << l << "\n";
		out.close();

		update_global_schedule_data(configDir.toUtf8().constData());
		notify_schedule_data_updated();
	}

	QJsonObject getSettingsJson() {
		QJsonObject s;
		s["show_game_clock"]         = showGameClockCheck->isChecked();
		s["show_shot_clock"]         = showShotClockCheck->isChecked();
		s["default_quarter_minutes"] = defaultQuarterMinutesSpin->value();
		s["default_quarter_seconds"] = defaultQuarterSecondsSpin->value();
		s["smoothing_frames"]        = smoothingFramesSpinBox->value();
		s["shot_clock_model_path"]   = shotClockModelEdit->text();
		s["game_clock_model_path"]   = gameClockModelEdit->text();
		s["config_dir"]              = configDir;
#ifdef USE_CNN_OCR
		s["clock_sync_mode"]         = clockSyncMode;
		s["shot_clock_matrix_path"]  = shotClockMatrixEdit->text();
		s["game_clock_matrix_path"]  = gameClockMatrixEdit->text();
		s["cnn_available"]           = true;
#else
		s["cnn_available"]           = false;
#endif
		return s;
	}

	void applySettingsJson(const QJsonObject &s) {
		if (s.contains("show_game_clock"))
			showGameClockCheck->setChecked(s["show_game_clock"].toBool());
		if (s.contains("show_shot_clock"))
			showShotClockCheck->setChecked(s["show_shot_clock"].toBool());
		if (s.contains("default_quarter_minutes"))
			defaultQuarterMinutesSpin->setValue(s["default_quarter_minutes"].toInt());
		if (s.contains("default_quarter_seconds"))
			defaultQuarterSecondsSpin->setValue(s["default_quarter_seconds"].toInt());
		if (s.contains("smoothing_frames"))
			smoothingFramesSpinBox->setValue(s["smoothing_frames"].toInt());
#ifdef USE_CNN_OCR
		if (s.contains("clock_sync_mode"))
			clockSyncModeCombo->setCurrentIndex(s["clock_sync_mode"].toInt());
#endif
	}

	QJsonObject getRoisJson() {
		QJsonObject result;
#ifdef USE_CNN_OCR
		if (ocrEngine) {
			auto shot = ocrEngine->getShotClockROI();
			auto game = ocrEngine->getGameClockROI();
			QJsonObject so; so["x"] = shot.x; so["y"] = shot.y; so["width"] = shot.width; so["height"] = shot.height;
			QJsonObject go; go["x"] = game.x; go["y"] = game.y; go["width"] = game.width; go["height"] = game.height;
			result["shot_clock"] = so;
			result["game_clock"] = go;
		}
#endif
		QSettings rs("WaterPoloScoreboard", "CNNModels");
		result["shot_clock_source"] = rs.value("shotClockROI_source", "").toString();
		result["game_clock_source"] = rs.value("gameClockROI_source", "").toString();
		return result;
	}

	// Sync all UI spinboxes/edits from g_scoreboard without triggering
	// updateScoreboard() — call this after a WebSocket update so the clock
	// timers don't overwrite the new values on their next tick.
	Q_INVOKABLE void syncUIFromState() {
		struct scoreboard_source *sb = get_global_scoreboard();
		if (!sb) return;
		QSignalBlocker b1(homeScoreSpin), b2(awayScoreSpin);
		QSignalBlocker b3(homeExclusionsSpin), b4(awayExclusionsSpin);
		QSignalBlocker b5(homeTimeoutsSpin), b6(awayTimeoutsSpin);
		QSignalBlocker b7(gameMinutesSpin), b8(gameSecondsSpin);
		QSignalBlocker b9(shotClockSpin);
		QSignalBlocker b10(homeTeamEdit), b11(awayTeamEdit);
		homeScoreSpin->setValue(sb->home_score);
		awayScoreSpin->setValue(sb->away_score);
		homeExclusionsSpin->setValue(sb->home_exclusions);
		awayExclusionsSpin->setValue(sb->away_exclusions);
		homeTimeoutsSpin->setValue(sb->home_timeouts);
		awayTimeoutsSpin->setValue(sb->away_timeouts);
		gameMinutesSpin->setValue(sb->game_clock_minutes);
		gameSecondsSpin->setValue(sb->game_clock_seconds);
		shotClockSpin->setValue(sb->shot_clock);
		homeTeamEdit->setText(QString::fromStdString(sb->home_team));
		awayTeamEdit->setText(QString::fromStdString(sb->away_team));
	}

	void setRoiData(const QString &clock, int x, int y, int w, int h) {
#ifdef USE_CNN_OCR
		if (!ocrEngine) return;
		QSettings rs("WaterPoloScoreboard", "CNNModels");
		if (clock == "shot") {
			ocrEngine->setShotClockROI(x, y, w, h);
			rs.setValue("shotClockROI_x", x); rs.setValue("shotClockROI_y", y);
			rs.setValue("shotClockROI_width", w); rs.setValue("shotClockROI_height", h);
		} else {
			ocrEngine->setGameClockROI(x, y, w, h);
			rs.setValue("gameClockROI_x", x); rs.setValue("gameClockROI_y", y);
			rs.setValue("gameClockROI_width", w); rs.setValue("gameClockROI_height", h);
		}
#else
		(void)clock; (void)x; (void)y; (void)w; (void)h;
#endif
	}

	QJsonArray getTeamsJson() {
		QJsonArray teams;
		if (configDir.isEmpty()) return teams;
		QFile file(configDir + "/teams.csv");
		if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return teams;
		QTextStream in(&file);
		in.readLine(); // header
		while (!in.atEnd()) {
			QString line = in.readLine().trimmed();
			if (line.isEmpty()) continue;
			QStringList p = line.split(',');
			if (p.size() >= 5) {
				QJsonObject t;
				t["name"]      = p[0].trimmed();
				t["home_bg"]   = p[1].trimmed();
				t["home_text"] = p[2].trimmed();
				t["away_bg"]   = p[3].trimmed();
				t["away_text"] = p[4].trimmed();
				teams.append(t);
			}
		}
		return teams;
	}

	void setTeamColorData(const QString &name, const QString &homeBg, const QString &homeText,
	                      const QString &awayBg, const QString &awayText) {
		if (configDir.isEmpty()) return;
		QString teamsPath = configDir + "/teams.csv";
		QFile file(teamsPath);
		if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return;
		QStringList lines;
		QTextStream in(&file);
		lines.append(in.readLine()); // header
		while (!in.atEnd()) {
			QString line = in.readLine().trimmed();
			if (line.isEmpty()) continue;
			QStringList p = line.split(',');
			if (!p.isEmpty() && p[0].trimmed() == name)
				lines.append(name + "," + homeBg + "," + homeText + "," + awayBg + "," + awayText);
			else
				lines.append(line);
		}
		file.close();
		QFile out(teamsPath);
		if (!out.open(QIODevice::WriteOnly | QIODevice::Text)) return;
		QTextStream outStream(&out);
		for (const QString &l : lines) outStream << l << "\n";
		out.close();
		loadTeamColors(teamsPath);
		update_global_schedule_data(configDir.toUtf8().constData());
		notify_schedule_data_updated();
	}

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

		// Restore source names and ROIs from previous session
		{
			QSettings roiSettings("WaterPoloScoreboard", "CNNModels");
			shotClockSourceName = roiSettings.value("shotClockROI_source", "").toString();
			gameClockSourceName = roiSettings.value("gameClockROI_source", "").toString();

			int sx = roiSettings.value("shotClockROI_x", 0).toInt();
			int sy = roiSettings.value("shotClockROI_y", 0).toInt();
			int sw = roiSettings.value("shotClockROI_width", 0).toInt();
			int sh = roiSettings.value("shotClockROI_height", 0).toInt();
			if (sw > 0 && sh > 0)
				ocrEngine->setShotClockROI(sx, sy, sw, sh);

			int gx = roiSettings.value("gameClockROI_x", 0).toInt();
			int gy = roiSettings.value("gameClockROI_y", 0).toInt();
			int gw = roiSettings.value("gameClockROI_width", 0).toInt();
			int gh = roiSettings.value("gameClockROI_height", 0).toInt();
			if (gw > 0 && gh > 0)
				ocrEngine->setGameClockROI(gx, gy, gw, gh);
		}
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

		modelsLayout->addWidget(new QLabel("Sync Mode:"), 10, 0);
		clockSyncModeCombo = new QComboBox();
		clockSyncModeCombo->addItem("Event-based (stop/start sync)", 0);
		clockSyncModeCombo->addItem("Rate-based (sub-second smear)", 1);
		{
			QSettings s("WaterPoloScoreboard", "CNNModels");
			clockSyncMode = s.value("clockSyncMode", 0).toInt();
			clockSyncModeCombo->setCurrentIndex(clockSyncMode);
		}
		modelsLayout->addWidget(clockSyncModeCombo, 10, 1, 1, 2);
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

		editTeamColorsBtn = new QPushButton("Edit Team Colors...");
		editTeamColorsBtn->setStyleSheet("QPushButton { padding: 4px 8px; }");
		teamsLayout->addWidget(editTeamColorsBtn, 2, 0, 1, 3);

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
		
		// Add save score button
		QPushButton *saveScoreBtn = new QPushButton("Save Score to Schedule");
		scoresLayout->addWidget(saveScoreBtn, 2, 0, 1, 4);
		
		scoresGroup->setLayout(scoresLayout);
		mainLayout->addWidget(scoresGroup);
		
		// Game Clock section
		QGroupBox *gameClockGroup = new QGroupBox("Game Clock");
		QGridLayout *gameClockLayout = new QGridLayout();
		
		// Default quarter time row
		QLabel *defaultTimeLabel = new QLabel("Default:");
		defaultTimeLabel->setStyleSheet("QLabel { color: #aaa; }");
		gameClockLayout->addWidget(defaultTimeLabel, 0, 0);

		defaultQuarterMinutesSpin = new QSpinBox();
		defaultQuarterMinutesSpin->setRange(0, 99);
		defaultQuarterMinutesSpin->setValue(8);
		defaultQuarterMinutesSpin->setSuffix("m");
		defaultQuarterMinutesSpin->setToolTip("Default quarter length (minutes)");
		gameClockLayout->addWidget(defaultQuarterMinutesSpin, 0, 1);

		defaultQuarterSecondsSpin = new QSpinBox();
		defaultQuarterSecondsSpin->setRange(0, 59);
		defaultQuarterSecondsSpin->setValue(0);
		defaultQuarterSecondsSpin->setSuffix("s");
		defaultQuarterSecondsSpin->setToolTip("Default quarter length (seconds)");
		gameClockLayout->addWidget(defaultQuarterSecondsSpin, 0, 2);

		// Current clock time row
		gameClockLayout->addWidget(new QLabel("Time:"), 1, 0);
		gameMinutesSpin = new QSpinBox();
		gameMinutesSpin->setRange(0, 99);
		gameMinutesSpin->setValue(8);
		gameClockLayout->addWidget(gameMinutesSpin, 1, 1);

		gameClockLayout->addWidget(new QLabel(":"), 1, 2, Qt::AlignCenter);
		gameSecondsSpin = new QSpinBox();
		gameSecondsSpin->setRange(0, 59);
		gameSecondsSpin->setValue(0);
		gameClockLayout->addWidget(gameSecondsSpin, 1, 3);

		startGameClockBtn = new QPushButton("Start");
		stopGameClockBtn = new QPushButton("Stop");
		QPushButton *resetGameClockBtn = new QPushButton("Reset");

		gameClockLayout->addWidget(startGameClockBtn, 2, 0);
		gameClockLayout->addWidget(stopGameClockBtn, 2, 1);
		gameClockLayout->addWidget(resetGameClockBtn, 2, 2);
		
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
		
		connect(saveScoreBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::saveScoreToSchedule);
		
		connect(startGameClockBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::startGameClock);
		connect(stopGameClockBtn, &QPushButton::clicked, this, &ScoreboardControlPanel::stopGameClock);
		connect(resetGameClockBtn, &QPushButton::clicked, [this]() {
			gameMinutesSpin->setValue(defaultQuarterMinutesSpin->value());
			gameSecondsSpin->setValue(defaultQuarterSecondsSpin->value());
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
		connect(editTeamColorsBtn, &QPushButton::clicked, this, [this]() {
			if (configDir.isEmpty()) {
				QMessageBox::warning(this, "No Directory Loaded",
					"Please load a schedule directory first so the teams.csv path is known.");
				return;
			}
			TeamColorEditorDialog dlg(configDir + "/teams.csv", this);
			connect(&dlg, &TeamColorEditorDialog::teamColorChanged, this,
				[this](const QString& team, const QColor& hBg, const QColor& hText,
				       const QColor& aBg, const QColor& aText) {
					bool changed = false;
					if (homeTeamEdit->text().trimmed() == team) {
						homeColor     = 0xFF000000 | (hBg.red()   << 16) | (hBg.green()   << 8) | hBg.blue();
						homeTextColor = 0xFF000000 | (hText.red() << 16) | (hText.green() << 8) | hText.blue();
						changed = true;
					}
					if (awayTeamEdit->text().trimmed() == team) {
						awayColor     = 0xFF000000 | (aBg.red()   << 16) | (aBg.green()   << 8) | aBg.blue();
						awayTextColor = 0xFF000000 | (aText.red() << 16) | (aText.green() << 8) | aText.blue();
						changed = true;
					}
					if (changed) {
						updateColorButtons();
						updateScoreboard();
					}
				});
			if (dlg.exec() == QDialog::Accepted) {
				loadTeamColors(configDir + "/teams.csv");
			}
		});
		
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
		connect(clockSyncModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
			clockSyncMode = idx;
			QSettings s("WaterPoloScoreboard", "CNNModels");
			s.setValue("clockSyncMode", clockSyncMode);
		});

		// Rate-based timers (only active when sync mode = 1 and clock is running)
		shotRateTimer = new QTimer(this);
		shotRateTimer->setInterval(kRateTickMs);
		connect(shotRateTimer, &QTimer::timeout, this, &ScoreboardControlPanel::onShotClockRateTick);
		gameRateTimer = new QTimer(this);
		gameRateTimer->setInterval(kRateTickMs);
		connect(gameRateTimer, &QTimer::timeout, this, &ScoreboardControlPanel::onGameClockRateTick);

		// Enable start button if ROIs + sources were already saved in a previous session
		checkEnableDetection();
#endif

		// Initialize models directory: prefer configDir, fall back to APPDATA (Windows)
		if (!configDir.isEmpty()) {
			modelsDir = configDir;
		} else {
			QString appData = QString::fromLocal8Bit(qgetenv("APPDATA"));
			if (!appData.isEmpty()) {
				modelsDir = appData + "/obs-studio/plugin_config/obs-scoreboard/models";
				QDir dir(modelsDir);
				if (!dir.exists()) {
					dir.mkpath(".");
				}
			}
		}

		{
			// Load saved model paths, defaulting to models in the config/models dir
			QSettings modelSettings("WaterPoloScoreboard", "CNNModels");
			QString defaultShotModel = modelsDir.isEmpty() ? "shot_clock_model.pt" : modelsDir + "/shot_clock_model.pt";
			QString defaultGameModel = modelsDir.isEmpty() ? "game_clock_model.pt" : modelsDir + "/game_clock_model.pt";
			QString shotModelPath = modelSettings.value("shotClockModel", defaultShotModel).toString();
			QString gameModelPath = modelSettings.value("gameClockModel", defaultGameModel).toString();
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
#ifdef USE_CNN_OCR
			if (clockSyncMode == 1 && gameRateTimer) {
				gameClockMs   = (gameMinutesSpin->value() * 60 + gameSecondsSpin->value()) * 1000.0;
				gameClockRate = 1.0;
				gameRateTimer->start();
			} else {
#endif
				gameClockTimer->start();
#ifdef USE_CNN_OCR
			}
#endif
			startGameClockBtn->setEnabled(false);
			stopGameClockBtn->setEnabled(true);
			// Keep shot clock in sync
			if (!clockSyncGuard) { clockSyncGuard = true; startShotClock(); clockSyncGuard = false; }
		}
	}

	void stopGameClock() {
		if (gameClockRunning) {
			gameClockRunning = false;
			gameClockTimer->stop();
#ifdef USE_CNN_OCR
			if (gameRateTimer) gameRateTimer->stop();
#endif
			startGameClockBtn->setEnabled(true);
			stopGameClockBtn->setEnabled(false);
			// Keep shot clock in sync
			if (!clockSyncGuard) { clockSyncGuard = true; stopShotClock(); clockSyncGuard = false; }
		}
	}

	void startShotClock() {
		if (!shotClockRunning) {
			shotClockRunning = true;
#ifdef USE_CNN_OCR
			if (clockSyncMode == 1 && shotRateTimer) {
				shotClockMs   = shotClockSpin->value() * 1000.0;
				shotClockRate = 1.0;
				shotRateTimer->start();
			} else {
#endif
				shotClockTimer->start();
#ifdef USE_CNN_OCR
			}
#endif
			startShotClockBtn->setEnabled(false);
			stopShotClockBtn->setEnabled(true);
			// Keep game clock in sync
			if (!clockSyncGuard) { clockSyncGuard = true; startGameClock(); clockSyncGuard = false; }
		}
	}

	void stopShotClock() {
		if (shotClockRunning) {
			shotClockRunning = false;
			shotClockTimer->stop();
#ifdef USE_CNN_OCR
			if (shotRateTimer) shotRateTimer->stop();
#endif
			startShotClockBtn->setEnabled(true);
			stopShotClockBtn->setEnabled(false);
			// Keep game clock in sync
			if (!clockSyncGuard) { clockSyncGuard = true; stopGameClock(); clockSyncGuard = false; }
		}
	}
	
	void onGameClockTick() {
		int total = gameMinutesSpin->value() * 60 + gameSecondsSpin->value();
		if (total <= 0) { stopGameClock(); return; }
		total--;
		if (total <= 0) {
			gameMinutesSpin->setValue(0);
			gameSecondsSpin->setValue(0);
			stopGameClock();
			return;
		}
		gameMinutesSpin->setValue(total / 60);
		gameSecondsSpin->setValue(total % 60);
		updateScoreboard();
	}
	
	void onShotClockTick() {
		int v = shotClockSpin->value();
		if (v <= 0) { stopShotClock(); return; }
		v--;
		if (v <= 0) { shotClockSpin->setValue(0); stopShotClock(); return; }
		shotClockSpin->setValue(v);
		updateScoreboard();
	}

#ifdef USE_CNN_OCR
	// Rate-based tick: fires every 50 ms, advances clock at shotClockRate speed.
	// The displayed integer only changes when the floor crosses a second boundary,
	// so the viewer always sees clean 1-second decrements regardless of the rate.
	void onShotClockRateTick() {
		shotClockMs -= kRateTickMs * shotClockRate;
		if (shotClockMs <= 0.0) {
			shotClockSpin->setValue(0);
			stopShotClock();
			updateScoreboard();
			return;
		}
		int display = (int)(shotClockMs / 1000.0);
		if (display != shotClockSpin->value()) {
			shotClockSpin->setValue(display);
			updateScoreboard();
		}
	}

	void onGameClockRateTick() {
		gameClockMs -= kRateTickMs * gameClockRate;
		if (gameClockMs <= 0.0) {
			gameMinutesSpin->setValue(0);
			gameSecondsSpin->setValue(0);
			stopGameClock();
			updateScoreboard();
			return;
		}
		int total   = (int)(gameClockMs / 1000.0);
		int display_m = total / 60;
		int display_s = total % 60;
		if (display_m != gameMinutesSpin->value() || display_s != gameSecondsSpin->value()) {
			gameMinutesSpin->setValue(display_m);
			gameSecondsSpin->setValue(display_s);
			updateScoreboard();
		}
	}
#endif

	void loadSchedule() {
		QString startDir = configDir.isEmpty() ? QDir::homePath() : configDir;
		QString csvPath = QFileDialog::getOpenFileName(
			this, "Select Schedule CSV", startDir,
			"Schedule files (*.csv);;All files (*)");
		if (csvPath.isEmpty()) return;

		// configDir is the folder containing the CSV — logos and teams.csv live here
		configDir = QFileInfo(csvPath).absolutePath();

		QSettings settings("WaterPoloScoreboard", "ControlPanel");
		settings.setValue("configDir", configDir);

		loadScheduleFromPath(configDir, csvPath);
	}

	void loadScheduleFromPath(const QString &dir, const QString &csvPath = QString()) {
		if (dir.isEmpty()) return;

		// Resolve which CSV file to use
		QString schedulePath = csvPath.isEmpty() ? dir + "/schedule.csv" : csvPath;

		// Load teams.csv first
		loadTeamColors(dir + "/teams.csv");

		// Update global schedule data (shared with schedule source)
		std::string config_dir = dir.toUtf8().constData();
		update_global_schedule_data(config_dir);

		QFile file(schedulePath);
		if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
			blog(LOG_WARNING, "Could not open schedule at %s", schedulePath.toUtf8().constData());
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
				
				// Resolve placeholder team names to actual teams
				std::string home_std = home.toUtf8().constData();
				std::string away_std = away.toUtf8().constData();
				home_std = resolve_team_placeholder(home_std, false);
				away_std = resolve_team_placeholder(away_std, false);
				home = QString::fromStdString(home_std);
				away = QString::fromStdString(away_std);
				
				QString displayText = QString("%1: %2 vs %3").arg(time, home, away);
				
				gameSelectCombo->addItem(displayText);
				gameSelectCombo->setItemData(gameSelectCombo->count() - 1, QStringList() << home << away);
			}
		}
		
		blog(LOG_INFO, "Loaded schedule from %s", schedulePath.toUtf8().constData());
		notify_schedule_data_updated();

		if (gameSelectCombo->count() > 1) {
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
	
	void saveScoreToSchedule() {
		if (configDir.isEmpty()) {
			QMessageBox::warning(this, "No Schedule Loaded", 
				"Please load a schedule directory first.");
			return;
		}
		
		QString homeTeam = homeTeamEdit->text().trimmed();
		QString awayTeam = awayTeamEdit->text().trimmed();
		int homeScore = homeScoreSpin->value();
		int awayScore = awayScoreSpin->value();
		
		if (homeTeam.isEmpty() || awayTeam.isEmpty()) {
			QMessageBox::warning(this, "Missing Team Names", 
				"Both home and away team names must be set.");
			return;
		}
		
		// Determine winner
		QString winner;
		if (homeScore > awayScore) {
			winner = homeTeam;
		} else if (awayScore > homeScore) {
			winner = awayTeam;
		} else {
			// Tie - prompt for winner
			QStringList options;
			options << homeTeam << awayTeam << "Tie";
			
			bool ok;
			QString selection = QInputDialog::getItem(this, "Game Tied", 
				QString("The score is tied %1-%2. Select the winner (or Tie):").arg(homeScore).arg(awayScore),
				options, 0, false, &ok);
			
			if (!ok) {
				return; // User cancelled
			}
			
			winner = selection;
		}
		
		// Read the schedule.csv file
		QString schedulePath = configDir + "/schedule.csv";
		QFile inputFile(schedulePath);
		if (!inputFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
			QMessageBox::critical(this, "Error", 
				QString("Could not open schedule file: %1").arg(schedulePath));
			return;
		}
		
		QStringList lines;
		QTextStream in(&inputFile);
		QString header = in.readLine();
		
		// Check if we need to add the score and winner columns
		QStringList headerParts = header.split(',');
		bool hasScoreColumns = headerParts.contains("home_score");
		
		if (!hasScoreColumns) {
			header += ",home_score,away_score,winner";
		}
		lines.append(header);
		
		bool foundGame = false;
		
		// Process each line - only update scores, don't change team names
		while (!in.atEnd()) {
			QString line = in.readLine();
			QStringList parts = line.split(',');
			
			if (parts.size() >= 3) {
				QString lineHome = parts[1].trimmed();
				QString lineAway = parts[2].trimmed();
				
				// Check if this is the current game
				if (lineHome == homeTeam && lineAway == awayTeam) {
					foundGame = true;
					
					// Build the updated line with scores only
					QString updatedLine;
					if (hasScoreColumns && parts.size() >= 6) {
						// Update existing score columns
						parts[3] = QString::number(homeScore);
						parts[4] = QString::number(awayScore);
						parts[5] = winner;
						updatedLine = parts.join(',');
					} else {
						// Add score columns
						updatedLine = QString("%1,%2,%3").arg(line).arg(homeScore).arg(awayScore);
						updatedLine += "," + winner;
					}
					lines.append(updatedLine);
					
					blog(LOG_INFO, "[SaveScore] Updated game: %s vs %s = %d-%d (Winner: %s)",
						homeTeam.toUtf8().constData(), awayTeam.toUtf8().constData(),
						homeScore, awayScore, winner.toUtf8().constData());
				} else {
					// Keep the line as is, but add empty columns if needed
					if (!hasScoreColumns) {
						line += ",,,"; // Empty score and winner columns
					}
					lines.append(line);
				}
			} else {
				lines.append(line);
			}
		}
		
		inputFile.close();
		
		if (!foundGame) {
			QMessageBox::warning(this, "Game Not Found", 
				QString("Could not find game '%1 vs %2' in schedule.").arg(homeTeam, awayTeam));
			return;
		}
		
		// Write back to the file (truncate to overwrite completely)
		QFile outputFile(schedulePath);
		if (!outputFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
			QMessageBox::critical(this, "Error", 
				QString("Could not write to schedule file: %1").arg(schedulePath));
			return;
		}
		
		QTextStream out(&outputFile);
		for (const QString &line : lines) {
			out << line << "\n";
		}
		
		outputFile.close();
		
		// Reload the global schedule data so the schedule view updates
		if (!configDir.isEmpty()) {
			blog(LOG_INFO, "[SaveScore] Calling update_global_schedule_data with dir: %s", 
				configDir.toUtf8().constData());
			update_global_schedule_data(configDir.toStdString());
			blog(LOG_INFO, "[SaveScore] Global schedule data reloaded, timestamp updated");
			
			// Force all schedule sources to refresh by triggering their update
			int source_count = 0;
			obs_enum_sources([](void* param, obs_source_t* source) {
				int* count = (int*)param;
				const char* id = obs_source_get_id(source);
				if (id && strcmp(id, "water_polo_schedule") == 0) {
					(*count)++;
					blog(LOG_INFO, "[SaveScore] Found schedule source, forcing refresh");
					// Get current settings and trigger update
					obs_data_t* settings = obs_source_get_settings(source);
					obs_source_update(source, settings);
					obs_data_release(settings);
				}
				return true;
			}, &source_count);
			blog(LOG_INFO, "[SaveScore] Refreshed %d schedule sources", source_count);
		}
		
		QMessageBox::information(this, "Score Saved", 
			QString("Score saved successfully:\n%1 %2 - %3 %4\nWinner: %5")
				.arg(homeTeam).arg(homeScore).arg(awayScore).arg(awayTeam).arg(winner));
		
		blog(LOG_INFO, "[SaveScore] Score saved to schedule.csv successfully");
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

                loadModelsBtn->setEnabled(false);
                loadModelsBtn->setText("Loading...");

                // Load models asynchronously to avoid blocking the main UI thread
                std::thread([this, shotModel, gameModel]() {
                        bool shotOk = false, gameOk = false;

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

                        // Call back to main UI thread safely
                        QMetaObject::invokeMethod(this, [this, shotOk, gameOk]() {
                                loadModelsBtn->setEnabled(true);
                                loadModelsBtn->setText("Load CNN Models");

                                if (shotOk || gameOk) {
                                        ocrEngine->enableBayesianFiltering(true);
                                        blog(LOG_INFO, "Bayesian filtering with Markov model enabled");
                                }

                                QString message;
                                if (shotOk && gameOk) {
                                        message = "✓ Both CNN models loaded successfully!\n\nBayesian filtering enabled:\n• Markov transition matrices\n• Multi-frame temporal smoothing\n• Handles blocked/obscured frames";
                                        selectShotRoiBtn->setEnabled(true);
                                        selectGameRoiBtn->setEnabled(true);
                                        QMessageBox::information(this, "CNN Models", message);
                                } else if (shotOk || gameOk) {
                                        message = QString("⚠ Partial success:\n%1%2\n\nBayesian filtering enabled for loaded model(s)")
                                                .arg(shotOk ? "✓ Shot clock model loaded\n" : "✗ Shot clock model failed\n")
                                                .arg(gameOk ? "✓ Game clock model loaded" : "✗ Game clock model failed");
                                        if (shotOk) selectShotRoiBtn->setEnabled(true);
                                        if (gameOk) selectGameRoiBtn->setEnabled(true);
                                        QMessageBox::warning(this, "CNN Models", message);
                                } else {
                                        message = "✗ Failed to load CNN models.\nPlease check the file paths.";
                                        QMessageBox::critical(this, "CNN Models", message);
                                }
                        }, Qt::QueuedConnection);
                }).detach();
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

		// Always prompt so the user can confirm or change the source
		QSettings roiSettings2("WaterPoloScoreboard", "CNNModels");
		QString savedSource = roiSettings2.value("shotClockROI_source", "").toString();
		int defaultIndex = 0;
		if (!savedSource.isEmpty()) {
			int idx = videoSourceNames.indexOf(savedSource);
			if (idx >= 0) defaultIndex = idx;
		}
		bool ok;
		QString selectedSource = QInputDialog::getItem(this,
			"Select Video Source",
			"Choose a video source to capture frame from:",
			videoSourceNames, defaultIndex, false, &ok);
		if (!ok || selectedSource.isEmpty()) {
			for (obs_source_t* src : videoSourceMap.values()) obs_source_release(src);
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
		dialog->hideCameraControls();
		
		
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

		// Always prompt so the user can confirm or change the source
		QSettings roiSettings2("WaterPoloScoreboard", "CNNModels");
		QString savedSource = roiSettings2.value("gameClockROI_source", "").toString();
		if (savedSource.isEmpty()) savedSource = roiSettings2.value("shotClockROI_source", "").toString();
		int defaultIndex = 0;
		if (!savedSource.isEmpty()) {
			int idx = videoSourceNames.indexOf(savedSource);
			if (idx >= 0) defaultIndex = idx;
		}
		bool ok;
		QString selectedSource = QInputDialog::getItem(this,
			"Select Video Source",
			"Choose a video source to capture frame from:",
			videoSourceNames, defaultIndex, false, &ok);
		if (!ok || selectedSource.isEmpty()) {
			for (obs_source_t* src : videoSourceMap.values()) obs_source_release(src);
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
		dialog->hideCameraControls();
		
		
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
	
	// Raw video callback: called by OBS's video output thread with CPU-side BGRA data.
	// Converts BGRA→RGBA and stores under mutex for the Qt timer to read.
	static void rawVideoCallback(void *param, struct video_data *frame) {
		auto *self = static_cast<ScoreboardControlPanel*>(param);
		if (!frame || !frame->data[0]) return;

		struct obs_video_info ovi;
		if (!obs_get_video_info(&ovi)) return;
		uint32_t w = ovi.base_width;
		uint32_t h = ovi.base_height;
		uint32_t stride = frame->linesize[0];

		std::vector<uint8_t> bytes(w * h * 4);
		for (uint32_t row = 0; row < h; row++) {
			const uint8_t *src = frame->data[0] + row * stride;
			uint8_t *dst = bytes.data() + row * w * 4;
			for (uint32_t x = 0; x < w; x++) {
				dst[x*4+0] = src[x*4+2]; // BGRA → RGBA
				dst[x*4+1] = src[x*4+1];
				dst[x*4+2] = src[x*4+0];
				dst[x*4+3] = src[x*4+3];
			}
		}

		std::lock_guard<std::mutex> lock(self->rawVideoMutex);
		self->rawVideoBytes = std::move(bytes);
		self->rawVideoW = w;
		self->rawVideoH = h;
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
		
		blog(LOG_INFO, "Starting detection with sources: shot='%s', game='%s'",
			shotClockSourceName.toUtf8().constData(),
			gameClockSourceName.toUtf8().constData());

		// No global canvas callback needed — frames are captured per-source
		// directly in updateClocksFromOCR using captureFrameFromOBSSource.

		// Timer drives OCR inference at 10 FPS
		ocrUpdateTimer = new QTimer(this);
		connect(ocrUpdateTimer, &QTimer::timeout, this, &ScoreboardControlPanel::updateClocksFromOCR);
		ocrUpdateTimer->start(100);
		
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
		
		shotClockVideoSource = nullptr;
		gameClockVideoSource = nullptr;
		
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
		
		// Capture directly from each named OBS source so ROI coordinates
		// (which are in source-frame space) match the captured pixels.
		QImage shotClockFrame, gameClockFrame;
		{
			obs_source_t *shotSrc = obs_get_source_by_name(shotClockSourceName.toUtf8().constData());
			if (shotSrc) {
				shotClockFrame = captureFrameFromOBSSource(shotSrc);
				obs_source_release(shotSrc);
			}
		}
		{
			obs_source_t *gameSrc = obs_get_source_by_name(gameClockSourceName.toUtf8().constData());
			if (gameSrc) {
				gameClockFrame = captureFrameFromOBSSource(gameSrc);
				obs_source_release(gameSrc);
			}
		}

		if (shotClockFrame.isNull() || gameClockFrame.isNull())
			return;
		
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

			shotPred = ocrEngine->predictShotClock();

			if (shotPred.confidence > 0.7 && shotPred.is_blocked) {
				// Blocked frame: reset same-run counter so blockage doesn't
				// accumulate toward pause detection. Timer keeps running.
				shotOCRSameRun = 0;
			} else if (shotPred.confidence > 0.7 && !shotPred.is_blocked) {
				try {
					int ocrVal = std::stoi(shotPred.value);

					// --- Pause / auto-start (shared by both modes) ---
					if (ocrVal == shotOCRLast) {
						shotOCRSameRun++;
						if (!shotOCRPaused && shotOCRSameRun >= kPauseFrameThreshold && shotClockRunning) {
							stopShotClock();
							shotOCRPaused = true;
						}
						// Hard-sync after ~3 s paused
						if (shotOCRPaused && shotOCRSameRun >= kPauseFrameThreshold * 3 / 2) {
							shotClockSpin->setValue(ocrVal);
							if (clockSyncMode == 1) shotClockMs = ocrVal * 1000.0;
						}
					} else if (ocrVal > shotOCRLast) {
						// Clock increased → reset occurred. Snap to new value and stay
						// paused until we see it actually start counting down.
						shotOCRSameRun = 0;
						shotClockSpin->setValue(ocrVal);
						if (clockSyncMode == 1) shotClockMs = ocrVal * 1000.0;
						if (shotClockRunning) stopShotClock();
						shotOCRPaused = true;
					} else {
						// Clock decreased → actively counting down
						shotOCRSameRun = 0;
						if (!shotClockRunning) {
							shotClockSpin->setValue(ocrVal);
							startShotClock(); // initialises shotClockMs in rate mode
						}
						shotOCRPaused = false;

						if (clockSyncMode == 0) {
							// --- Event-based: safety snap only ---
							int internal   = shotClockSpin->value();
							int diff       = std::abs(ocrVal - internal);
							bool nearZero  = (ocrVal <= 5 || internal <= 5);
							if (diff > (nearZero ? 1 : 10))
								shotClockSpin->setValue(ocrVal);
						} else {
							// --- Rate-based: adjust playback rate to converge ---
							double ocrMs  = ocrVal * 1000.0;
							double diffMs = shotClockMs - ocrMs; // + = ahead, - = behind
							if (std::abs(diffMs) > 10000.0) {
								// Way off — hard snap
								shotClockMs   = ocrMs;
								shotClockRate = 1.0;
							} else {
								// Near zero tighten convergence so accuracy is preserved
								bool nearZero = (ocrVal <= 5);
								double conv   = nearZero ? kRateConvergenceMs / 2.0 : kRateConvergenceMs;
								shotClockRate = 1.0 + diffMs / conv;
								shotClockRate = std::max(0.5, std::min(2.0, shotClockRate));
							}
						}
					}
					shotOCRLast = ocrVal;
				} catch (...) {}
			}
		}

		// Process game clock with Bayesian filtering
		if (gameRoi.x >= 0 && gameRoi.y >= 0 &&
		    gameRoi.x + gameRoi.width <= gameMat.cols &&
		    gameRoi.y + gameRoi.height <= gameMat.rows) {

			gamePred = ocrEngine->predictGameClock();

			if (gamePred.confidence > 0.7 && gamePred.is_blocked) {
				gameOCRSameRun = 0;
			} else if (gamePred.confidence > 0.7 && !gamePred.is_blocked) {
				try {
					size_t colonPos = gamePred.value.find(':');
					if (colonPos != std::string::npos) {
						int minutes  = std::stoi(gamePred.value.substr(0, colonPos));
						int seconds  = std::stoi(gamePred.value.substr(colonPos + 1));
						int ocrTotal = minutes * 60 + seconds;

						if (ocrTotal == gameOCRLast) {
							gameOCRSameRun++;
							if (!gameOCRPaused && gameOCRSameRun >= kPauseFrameThreshold && gameClockRunning) {
								stopGameClock();
								gameOCRPaused = true;
							}
							// Hard-sync after ~3 s paused
							if (gameOCRPaused && gameOCRSameRun >= kPauseFrameThreshold * 3 / 2) {
								gameMinutesSpin->setValue(ocrTotal / 60);
								gameSecondsSpin->setValue(ocrTotal % 60);
								if (clockSyncMode == 1) gameClockMs = ocrTotal * 1000.0;
							}
						} else if (ocrTotal > gameOCRLast) {
							// Clock increased → reset. Stay paused, snap to new value.
							gameOCRSameRun = 0;
							gameMinutesSpin->setValue(ocrTotal / 60);
							gameSecondsSpin->setValue(ocrTotal % 60);
							if (clockSyncMode == 1) gameClockMs = ocrTotal * 1000.0;
							if (gameClockRunning) stopGameClock();
							gameOCRPaused = true;
						} else {
							// Clock decreased → counting down
							gameOCRSameRun = 0;
							if (!gameClockRunning) {
								gameMinutesSpin->setValue(ocrTotal / 60);
								gameSecondsSpin->setValue(ocrTotal % 60);
								startGameClock(); // initialises gameClockMs in rate mode
							}
							gameOCRPaused = false;

							if (clockSyncMode == 0) {
								int internalTotal = gameMinutesSpin->value() * 60 + gameSecondsSpin->value();
								int diff          = std::abs(ocrTotal - internalTotal);
								bool nearZero     = (ocrTotal <= 10 || internalTotal <= 10);
								if (diff > (nearZero ? 1 : 10)) {
									gameMinutesSpin->setValue(ocrTotal / 60);
									gameSecondsSpin->setValue(ocrTotal % 60);
								}
							} else {
								double ocrMs  = ocrTotal * 1000.0;
								double diffMs = gameClockMs - ocrMs;
								if (std::abs(diffMs) > 10000.0) {
									gameClockMs   = ocrMs;
									gameClockRate = 1.0;
								} else {
									bool nearZero = (ocrTotal <= 10);
									double conv   = nearZero ? kRateConvergenceMs / 2.0 : kRateConvergenceMs;
									gameClockRate = 1.0 + diffMs / conv;
									gameClockRate = std::max(0.5, std::min(2.0, gameClockRate));
								}
							}
						}
						gameOCRLast = ocrTotal;
					}
				} catch (...) {}
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

void select_next_game()
{
	if (g_control_panel)
		QMetaObject::invokeMethod(g_control_panel, "nextGame", Qt::QueuedConnection);
}

void select_prev_game()
{
	if (g_control_panel)
		QMetaObject::invokeMethod(g_control_panel, "prevGame", Qt::QueuedConnection);
}

QJsonObject get_schedule_json()
{
	return g_control_panel ? g_control_panel->getScheduleJson() : QJsonObject();
}

void set_game_score_at_index(int idx, int hs, int as_, const QString &winner)
{
	if (g_control_panel) g_control_panel->setGameScoreAtIndex(idx, hs, as_, winner);
}

QJsonObject get_settings_json()
{
	return g_control_panel ? g_control_panel->getSettingsJson() : QJsonObject();
}

void apply_settings_json(const QJsonObject &s)
{
	if (g_control_panel) g_control_panel->applySettingsJson(s);
}

QJsonObject get_rois_json()
{
	return g_control_panel ? g_control_panel->getRoisJson() : QJsonObject();
}

void set_roi_data(const QString &clock, int x, int y, int w, int h)
{
	if (g_control_panel) g_control_panel->setRoiData(clock, x, y, w, h);
}

QJsonArray get_teams_json()
{
	return g_control_panel ? g_control_panel->getTeamsJson() : QJsonArray();
}

void set_team_color_data(const QString &name, const QString &hbg, const QString &ht,
                         const QString &abg, const QString &at_)
{
	if (g_control_panel) g_control_panel->setTeamColorData(name, hbg, ht, abg, at_);
}

void sync_control_panel_ui()
{
	if (g_control_panel)
		QMetaObject::invokeMethod(g_control_panel, "syncUIFromState", Qt::QueuedConnection);
}

