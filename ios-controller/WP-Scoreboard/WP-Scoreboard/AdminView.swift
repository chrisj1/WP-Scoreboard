import SwiftUI

struct AdminView: View {
    @EnvironmentObject var conn: ScoreboardConnection
    @State private var editingTeam: TeamColor?
    @State private var editingRoiClock: String? // "shot" or "game"

    var body: some View {
        NavigationStack {
            List {
                displaySection
                clockSection
                cnnSection
                roiSection
                teamsSection
            }
            .navigationTitle("Admin")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button { refresh() } label: { Image(systemName: "arrow.clockwise") }
                }
            }
            .onAppear { refresh() }
            .sheet(item: $editingTeam) { team in
                TeamColorEditSheet(team: team) { updated in
                    conn.setTeamColor(name: updated.name,
                                      homeBg: updated.homeBg, homeText: updated.homeText,
                                      awayBg: updated.awayBg, awayText: updated.awayText)
                    editingTeam = nil
                    Task {
                        try? await Task.sleep(nanoseconds: 300_000_000)
                        conn.fetchTeams()
                    }
                }
            }
            .sheet(item: .init(get: { editingRoiClock.map { ROIEditTarget(clock: $0) } },
                               set: { editingRoiClock = $0?.clock })) { target in
                ROIEditSheet(clock: target.clock, current: target.clock == "shot" ? conn.roiData.shotClock : conn.roiData.gameClock) { roi in
                    conn.setRoi(clock: target.clock, x: roi.x, y: roi.y, width: roi.width, height: roi.height)
                    editingRoiClock = nil
                }
            }
        }
    }

    private func refresh() {
        conn.fetchSettings()
        conn.fetchRois()
        conn.fetchTeams()
    }

    // MARK: - Display section

    var displaySection: some View {
        Section("Display") {
            Toggle("Show Game Clock", isOn: Binding(
                get: { conn.obsSettings.showGameClock },
                set: { conn.applySettings(["show_game_clock": $0]); var s = conn.obsSettings; s.showGameClock = $0; conn.obsSettings = s }
            ))
            Toggle("Show Shot Clock", isOn: Binding(
                get: { conn.obsSettings.showShotClock },
                set: { conn.applySettings(["show_shot_clock": $0]); var s = conn.obsSettings; s.showShotClock = $0; conn.obsSettings = s }
            ))
        }
    }

    // MARK: - Clock section

    var clockSection: some View {
        Section("Clock Defaults") {
            HStack {
                Text("Quarter Length")
                Spacer()
                Stepper(value: Binding(
                    get: { conn.obsSettings.defaultQuarterMinutes },
                    set: { v in
                        var s = conn.obsSettings; s.defaultQuarterMinutes = v; conn.obsSettings = s
                        conn.applySettings(["default_quarter_minutes": v])
                    }), in: 0...99) {
                    Text("\(conn.obsSettings.defaultQuarterMinutes)m \(conn.obsSettings.defaultQuarterSeconds)s")
                        .monospacedDigit()
                }
            }
            HStack {
                Text("Quarter Seconds")
                Spacer()
                Stepper(value: Binding(
                    get: { conn.obsSettings.defaultQuarterSeconds },
                    set: { v in
                        var s = conn.obsSettings; s.defaultQuarterSeconds = v; conn.obsSettings = s
                        conn.applySettings(["default_quarter_seconds": v])
                    }), in: 0...59) {
                    Text("\(conn.obsSettings.defaultQuarterSeconds)s").monospacedDigit()
                }
            }
        }
    }

    // MARK: - CNN section

    @ViewBuilder
    var cnnSection: some View {
        if conn.obsSettings.cnnAvailable {
            Section("CNN Clock Detection") {
                Picker("Sync Mode", selection: Binding(
                    get: { conn.obsSettings.clockSyncMode },
                    set: { v in
                        var s = conn.obsSettings; s.clockSyncMode = v; conn.obsSettings = s
                        conn.applySettings(["clock_sync_mode": v])
                    })) {
                    Text("Event-based").tag(0)
                    Text("Rate-based").tag(1)
                }
                HStack {
                    Text("Smoothing Frames")
                    Spacer()
                    Stepper(value: Binding(
                        get: { conn.obsSettings.smoothingFrames },
                        set: { v in
                            var s = conn.obsSettings; s.smoothingFrames = v; conn.obsSettings = s
                            conn.applySettings(["smoothing_frames": v])
                        }), in: 1...10) {
                        Text("\(conn.obsSettings.smoothingFrames)").monospacedDigit()
                    }
                }
                if !conn.obsSettings.shotClockModelPath.isEmpty {
                    LabeledContent("Shot Model") {
                        Text(URL(fileURLWithPath: conn.obsSettings.shotClockModelPath).lastPathComponent)
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                if !conn.obsSettings.gameClockModelPath.isEmpty {
                    LabeledContent("Game Model") {
                        Text(URL(fileURLWithPath: conn.obsSettings.gameClockModelPath).lastPathComponent)
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    // MARK: - ROI section

    @ViewBuilder
    var roiSection: some View {
        if conn.obsSettings.cnnAvailable {
            Section("Regions of Interest") {
                ROIRow(label: "Shot Clock", roi: conn.roiData.shotClock,
                       source: conn.roiData.shotClockSource) { editingRoiClock = "shot" }
                ROIRow(label: "Game Clock", roi: conn.roiData.gameClock,
                       source: conn.roiData.gameClockSource) { editingRoiClock = "game" }
            }
        }
    }

    // MARK: - Teams section

    var teamsSection: some View {
        Section("Team Colors") {
            if conn.teamColors.isEmpty {
                Text("No teams loaded").foregroundStyle(.secondary).font(.caption)
            } else {
                ForEach(conn.teamColors) { team in
                    Button { editingTeam = team } label: {
                        TeamColorRow(team: team)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}

// MARK: - ROI Row

struct ROIRow: View {
    let label: String
    let roi: ROIRect
    let source: String
    let onEdit: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(label).font(.headline)
                if roi.isEmpty {
                    Text("Not configured").font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("\(roi.x),\(roi.y)  \(roi.width)×\(roi.height)")
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                if !source.isEmpty {
                    Text(source).font(.caption2).foregroundStyle(.tertiary)
                }
            }
            Spacer()
            Button("Edit", action: onEdit).buttonStyle(.bordered)
        }
    }
}

// MARK: - Team Color Row

struct TeamColorRow: View {
    let team: TeamColor

    var body: some View {
        HStack(spacing: 12) {
            Text(team.name).font(.headline).frame(maxWidth: .infinity, alignment: .leading)
            ColorSwatch(bg: team.homeBgColor, text: team.homeTextColor, label: "H")
            ColorSwatch(bg: team.awayBgColor, text: team.awayTextColor, label: "A")
            Image(systemName: "chevron.right").foregroundStyle(.secondary).font(.caption)
        }
    }
}

struct ColorSwatch: View {
    let bg: Color
    let text: Color
    let label: String

    var body: some View {
        Text(label)
            .font(.caption.bold())
            .foregroundStyle(text)
            .frame(width: 28, height: 28)
            .background(bg)
            .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

// MARK: - Team Color Edit Sheet

struct TeamColorEditSheet: View {
    let team: TeamColor
    let onSave: (TeamColor) -> Void

    @State private var homeBg: Color
    @State private var homeText: Color
    @State private var awayBg: Color
    @State private var awayText: Color

    init(team: TeamColor, onSave: @escaping (TeamColor) -> Void) {
        self.team = team
        self.onSave = onSave
        _homeBg   = State(initialValue: team.homeBgColor)
        _homeText = State(initialValue: team.homeTextColor)
        _awayBg   = State(initialValue: team.awayBgColor)
        _awayText = State(initialValue: team.awayTextColor)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Home") {
                    ColorPicker("Background", selection: $homeBg, supportsOpacity: false)
                    ColorPicker("Text", selection: $homeText, supportsOpacity: false)
                    HStack {
                        Text("Preview")
                        Spacer()
                        Text(team.name)
                            .font(.headline)
                            .foregroundStyle(homeText)
                            .padding(.horizontal, 12).padding(.vertical, 6)
                            .background(homeBg)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }
                Section("Away") {
                    ColorPicker("Background", selection: $awayBg, supportsOpacity: false)
                    ColorPicker("Text", selection: $awayText, supportsOpacity: false)
                    HStack {
                        Text("Preview")
                        Spacer()
                        Text(team.name)
                            .font(.headline)
                            .foregroundStyle(awayText)
                            .padding(.horizontal, 12).padding(.vertical, 6)
                            .background(awayBg)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
            .navigationTitle(team.name)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { onSave(team) } // dismiss without changes
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        var updated = team
                        updated.homeBg   = homeBg.toHex()
                        updated.homeText = homeText.toHex()
                        updated.awayBg   = awayBg.toHex()
                        updated.awayText = awayText.toHex()
                        onSave(updated)
                    }
                    .fontWeight(.bold)
                }
            }
        }
    }
}

// MARK: - ROI Edit Sheet

struct ROIEditTarget: Identifiable {
    let clock: String
    var id: String { clock }
}

struct ROIEditSheet: View {
    let clock: String
    let current: ROIRect
    let onSave: (ROIRect) -> Void

    @State private var x: Int
    @State private var y: Int
    @State private var width: Int
    @State private var height: Int

    init(clock: String, current: ROIRect, onSave: @escaping (ROIRect) -> Void) {
        self.clock = clock
        self.current = current
        self.onSave = onSave
        _x      = State(initialValue: current.x)
        _y      = State(initialValue: current.y)
        _width  = State(initialValue: max(current.width,  1))
        _height = State(initialValue: max(current.height, 1))
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Position (pixels from top-left of source)") {
                    LabeledStepper(label: "X", value: $x, range: 0...9999)
                    LabeledStepper(label: "Y", value: $y, range: 0...9999)
                }
                Section("Size") {
                    LabeledStepper(label: "Width",  value: $width,  range: 1...9999)
                    LabeledStepper(label: "Height", value: $height, range: 1...9999)
                }
                Section {
                    Text("\(clock == "shot" ? "Shot" : "Game") clock region: (\(x), \(y)) \(width)×\(height) px")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("\(clock == "shot" ? "Shot" : "Game") Clock ROI")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { onSave(current) }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        onSave(ROIRect(x: x, y: y, width: width, height: height))
                    }
                    .fontWeight(.bold)
                }
            }
        }
    }
}

struct LabeledStepper: View {
    let label: String
    @Binding var value: Int
    let range: ClosedRange<Int>

    var body: some View {
        HStack {
            Text(label).frame(width: 60, alignment: .leading)
            Spacer()
            Stepper(value: $value, in: range, step: 10) {
                Text("\(value)").monospacedDigit().frame(width: 60, alignment: .trailing)
            }
        }
    }
}
