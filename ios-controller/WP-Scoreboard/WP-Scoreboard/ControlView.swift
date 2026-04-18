import SwiftUI

struct ControlView: View {
    @EnvironmentObject var conn: ScoreboardConnection

    var body: some View {
        TabView {
            ScoreTab()
                .tabItem { Label("Score", systemImage: "number.circle.fill") }

            ClocksTab()
                .tabItem { Label("Clocks", systemImage: "timer") }

            ExtrasTab()
                .tabItem { Label("Extras", systemImage: "ellipsis.circle.fill") }

            GameTab()
                .tabItem { Label("Game", systemImage: "calendar") }

            ScheduleView()
                .tabItem { Label("Schedule", systemImage: "list.bullet.rectangle") }

            AdminView()
                .tabItem { Label("Admin", systemImage: "gearshape.fill") }

            OBSView()
                .tabItem { Label("OBS", systemImage: "tv") }

            ChatView()
                .tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }
        }
        .navigationBarHidden(true)
    }
}

// MARK: - Score Tab

struct ScoreTab: View {
    @EnvironmentObject var conn: ScoreboardConnection

    var body: some View {
        VStack(spacing: 0) {
            // Live score header
            HStack {
                Spacer()
                VStack {
                    Text(conn.state.homeTeam).font(.caption).foregroundColor(.secondary)
                    Text("\(conn.state.homeScore)").font(.system(size: 64, weight: .bold))
                }
                Spacer()
                Text("–").font(.largeTitle).foregroundColor(.secondary)
                Spacer()
                VStack {
                    Text(conn.state.awayTeam).font(.caption).foregroundColor(.secondary)
                    Text("\(conn.state.awayScore)").font(.system(size: 64, weight: .bold))
                }
                Spacer()
            }
            .padding()
            .background(Color(.systemGray6))

            // Period + clock strip
            HStack {
                Text(conn.state.periodLabel).font(.headline)
                Spacer()
                Text(conn.state.gameClock).font(.headline.monospacedDigit()
)
                Spacer()
                Text("Shot: \(String(format: "%02d", conn.state.shotClock))")
                    .font(.headline.monospacedDigit())
                    .foregroundColor(conn.state.shotClock <= 5 ? .red : .primary)
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(Color(.systemGray5))

            ScrollView {
                VStack(spacing: 16) {
                    // Home score controls
                    TeamScoreSection(
                        teamName: conn.state.homeTeam,
                        score: conn.state.homeScore,
                        color: conn.state.homeColor,
                        onUp:   { conn.homeScoreUp() },
                        onDown: { conn.homeScoreDown() }
                    )

                    // Away score controls
                    TeamScoreSection(
                        teamName: conn.state.awayTeam,
                        score: conn.state.awayScore,
                        color: conn.state.awayColor,
                        onUp:   { conn.awayScoreUp() },
                        onDown: { conn.awayScoreDown() }
                    )
                }
                .padding()
            }
        }
    }
}

struct TeamScoreSection: View {
    let teamName: String
    let score: Int
    let color: Color
    let onUp: () -> Void
    let onDown: () -> Void

    var body: some View {
        VStack(spacing: 8) {
            Text(teamName)
                .font(.headline)
                .foregroundColor(color)

            HStack(spacing: 20) {
                ControlButton("−1", color: color.opacity(0.6), action: onDown)
                Text("\(score)")
                    .font(.system(size: 48, weight: .bold))
                    .frame(width: 80)
                ControlButton("+1", color: color, action: onUp)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(16)
    }
}

// MARK: - Clocks Tab

struct ClocksTab: View {
    @EnvironmentObject var conn: ScoreboardConnection

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Period controls
                SectionCard(title: "Period") {
                    HStack(spacing: 12) {
                        ControlButton("◀ Prev", color: .gray) { conn.prevPeriod() }
                        VStack {
                            Text(conn.state.periodLabel).font(.title2.bold())
                            Text("Period").font(.caption).foregroundColor(.secondary)
                        }.frame(maxWidth: .infinity)
                        ControlButton("Next ▶", color: .blue) { conn.nextPeriod() }
                    }
                    ControlButton("Set Final", color: .orange, wide: true) { conn.setFinal() }
                }

                // Shot clock
                SectionCard(title: "Shot Clock") {
                    HStack {
                        Text(String(format: "%02d", conn.state.shotClock))
                            .font(.system(size: 56, weight: .bold).monospacedDigit())
                            .foregroundColor(conn.state.shotClock <= 5 ? .red : .primary)
                        Spacer()
                        VStack(spacing: 10) {
                            ControlButton("Reset 30", color: .blue) { conn.resetShotClock() }
                            ControlButton("Set 35",   color: .teal) { conn.shotClock35() }
                        }
                    }
                }

                // Game clock (read-only display)
                SectionCard(title: "Game Clock") {
                    Text(conn.state.gameClock)
                        .font(.system(size: 56, weight: .bold).monospacedDigit())
                        .frame(maxWidth: .infinity)
                }
            }
            .padding()
        }
    }
}

// MARK: - Extras Tab

struct ExtrasTab: View {
    @EnvironmentObject var conn: ScoreboardConnection

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Timeouts
                SectionCard(title: "Timeouts") {
                    TimeoutRow(
                        team: conn.state.homeTeam,
                        remaining: conn.state.homeTimeouts,
                        color: conn.state.homeColor,
                        onUse: { conn.useHomeTimeout() },
                        onRestore: { conn.restoreHomeTimeout() }
                    )
                    Divider()
                    TimeoutRow(
                        team: conn.state.awayTeam,
                        remaining: conn.state.awayTimeouts,
                        color: conn.state.awayColor,
                        onUse: { conn.useAwayTimeout() },
                        onRestore: { conn.restoreAwayTimeout() }
                    )
                }

                // Exclusions
                SectionCard(title: "Exclusions") {
                    ExclusionRow(
                        team: conn.state.homeTeam,
                        count: conn.state.homeExclusions,
                        color: conn.state.homeColor,
                        onAdd: { conn.addHomeExclusion() },
                        onClear: { conn.clearHomeExclusions() }
                    )
                    Divider()
                    ExclusionRow(
                        team: conn.state.awayTeam,
                        count: conn.state.awayExclusions,
                        color: conn.state.awayColor,
                        onAdd: { conn.addAwayExclusion() },
                        onClear: { conn.clearAwayExclusions() }
                    )
                }

                // Man-up
                SectionCard(title: "Man-Up") {
                    HStack(spacing: 12) {
                        ManupButton(
                            team: conn.state.homeTeam,
                            active: conn.state.homeManup,
                            color: conn.state.homeColor,
                            action: { conn.toggleHomeManup() }
                        )
                        ManupButton(
                            team: conn.state.awayTeam,
                            active: conn.state.awayManup,
                            color: conn.state.awayColor,
                            action: { conn.toggleAwayManup() }
                        )
                    }
                }
            }
            .padding()
        }
    }
}

struct TimeoutRow: View {
    let team: String; let remaining: Int; let color: Color
    let onUse: () -> Void; let onRestore: () -> Void

    var body: some View {
        HStack {
            Text(team).font(.headline).foregroundColor(color).frame(width: 80, alignment: .leading)
            Text(String(repeating: "●", count: 3 - remaining) + String(repeating: "○", count: remaining))
                .font(.title3)
            Spacer()
            Button("Use", action: onUse)
                .buttonStyle(.borderedProminent).tint(remaining > 0 ? color : .gray)
                .disabled(remaining == 0)
            Button("+", action: onRestore)
                .buttonStyle(.bordered)
                .disabled(remaining >= 3)
        }
    }
}

struct ExclusionRow: View {
    let team: String; let count: Int; let color: Color
    let onAdd: () -> Void; let onClear: () -> Void

    var body: some View {
        HStack {
            Text(team).font(.headline).foregroundColor(color).frame(width: 80, alignment: .leading)
            Text("\(count)").font(.title2.bold())
            Spacer()
            Button("+EXCL", action: onAdd).buttonStyle(.borderedProminent).tint(color)
            Button("CLR",   action: onClear).buttonStyle(.bordered).disabled(count == 0)
        }
    }
}

struct ManupButton: View {
    let team: String; let active: Bool; let color: Color; let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack {
                Text("▲").font(.title)
                Text(active ? "MAN UP" : "man up").font(.caption.bold())
                Text(team).font(.caption2)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(active ? Color.yellow.opacity(0.3) : Color(.systemGray6))
            .foregroundColor(active ? .yellow : .secondary)
            .cornerRadius(12)
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(active ? Color.yellow : Color.clear, lineWidth: 2))
        }
    }
}

// MARK: - Game Tab

struct GameTab: View {
    @EnvironmentObject var conn: ScoreboardConnection
    @State private var showResetConfirm = false

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                SectionCard(title: "Schedule") {
                    HStack(spacing: 12) {
                        ControlButton("◀ Prev Game", color: .gray) { conn.prevGame() }
                        ControlButton("Next Game ▶", color: .blue) { conn.nextGame() }
                    }
                }

                SectionCard(title: "Instant Replay") {
                    ControlButton("⏪  Replay", color: .purple, wide: true) {
                        conn.triggerReplay()
                    }
                }

                SectionCard(title: "Current Game") {
                    HStack {
                        VStack(alignment: .leading) {
                            Text(conn.state.homeTeam).font(.headline)
                            Text("Home").font(.caption).foregroundColor(.secondary)
                        }
                        Spacer()
                        Text("vs").foregroundColor(.secondary)
                        Spacer()
                        VStack(alignment: .trailing) {
                            Text(conn.state.awayTeam).font(.headline)
                            Text("Away").font(.caption).foregroundColor(.secondary)
                        }
                    }
                }

                SectionCard(title: "Danger Zone") {
                    Button {
                        showResetConfirm = true
                    } label: {
                        Text("Reset Entire Game")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.red.opacity(0.15))
                            .foregroundColor(.red)
                            .cornerRadius(10)
                            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.red.opacity(0.5), lineWidth: 1))
                    }
                    .confirmationDialog("Reset the entire game?", isPresented: $showResetConfirm, titleVisibility: .visible) {
                        Button("Reset", role: .destructive) { conn.resetGame() }
                        Button("Cancel", role: .cancel) {}
                    } message: {
                        Text("This will zero all scores, timeouts, and exclusions.")
                    }
                }
            }
            .padding()
        }
    }
}

// MARK: - Shared components

struct ControlButton: View {
    let label: String
    let color: Color
    let wide: Bool
    let action: () -> Void

    init(_ label: String, color: Color, wide: Bool = false, action: @escaping () -> Void) {
        self.label = label; self.color = color; self.wide = wide; self.action = action
    }

    var body: some View {
        Button(action: action) {
            Text(label)
                .font(.headline)
                .frame(minWidth: 90, maxWidth: wide ? .infinity : nil)
                .padding(.vertical, 14)
                .padding(.horizontal, 16)
                .background(color)
                .foregroundColor(.white)
                .cornerRadius(12)
        }
    }
}

struct SectionCard<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.caption.uppercaseSmallCaps())
                .foregroundColor(.secondary)
            content
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(16)
    }
}
