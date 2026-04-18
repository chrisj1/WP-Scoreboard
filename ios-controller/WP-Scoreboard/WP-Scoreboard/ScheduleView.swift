import SwiftUI

struct ScheduleView: View {
    @EnvironmentObject var conn: ScoreboardConnection
    @State private var editingGame: ScheduleGame?

    var body: some View {
        NavigationStack {
            Group {
                if conn.scheduleGames.isEmpty {
                    ContentUnavailableView("No Schedule", systemImage: "calendar.badge.exclamationmark",
                                          description: Text("Tap Refresh or ensure a schedule is loaded in OBS."))
                } else {
                    List(conn.scheduleGames) { game in
                        ScheduleGameRow(game: game) { editingGame = game }
                    }
                }
            }
            .navigationTitle("Schedule")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button { conn.fetchSchedule() } label: { Image(systemName: "arrow.clockwise") }
                }
            }
            .onAppear { conn.fetchSchedule() }
            .sheet(item: $editingGame) { game in
                ScoreEditSheet(game: game) { editingGame = nil }
                    .environmentObject(conn)
            }
        }
    }
}

struct ScheduleGameRow: View {
    let game: ScheduleGame
    let onEdit: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("\(game.home) vs \(game.away)")
                    .font(.headline)
                Text(game.startTime)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if !game.winner.isEmpty {
                    Text("W: \(game.winner)")
                        .font(.caption2)
                        .foregroundStyle(.green)
                }
            }
            Spacer()
            if game.isPlayed {
                Text("\(game.homeScore ?? 0) – \(game.awayScore ?? 0)")
                    .font(.title3.bold().monospacedDigit())
            } else {
                Text("TBD")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(Color(.systemGray5))
                    .clipShape(Capsule())
            }
            Button(action: onEdit) {
                Image(systemName: "pencil.circle.fill")
                    .font(.title2)
                    .foregroundStyle(.blue)
            }
            .buttonStyle(.plain)
        }
        .padding(.vertical, 2)
    }
}

struct ScoreEditSheet: View {
    @EnvironmentObject var conn: ScoreboardConnection
    let game: ScheduleGame
    let onDone: () -> Void

    @State private var homeScore: Int
    @State private var awayScore: Int
    @State private var winner: String

    init(game: ScheduleGame, onDone: @escaping () -> Void) {
        self.game = game
        self.onDone = onDone
        _homeScore = State(initialValue: game.homeScore ?? 0)
        _awayScore = State(initialValue: game.awayScore ?? 0)
        _winner    = State(initialValue: game.winner)
    }

    private var computedWinner: String {
        if homeScore > awayScore { return game.home }
        if awayScore > homeScore { return game.away }
        return winner
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    HStack {
                        Text(game.home)
                            .font(.headline)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Stepper(value: $homeScore, in: 0...99) {
                            Text("\(homeScore)").font(.title2.bold().monospacedDigit()).frame(width: 40)
                        }
                    }
                    HStack {
                        Text(game.away)
                            .font(.headline)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Stepper(value: $awayScore, in: 0...99) {
                            Text("\(awayScore)").font(.title2.bold().monospacedDigit()).frame(width: 40)
                        }
                    }
                } header: { Text("Score") }

                if homeScore == awayScore {
                    Section("Winner (tied — choose manually)") {
                        Picker("Winner", selection: $winner) {
                            Text(game.home).tag(game.home)
                            Text(game.away).tag(game.away)
                            Text("Tie").tag("")
                        }
                        .pickerStyle(.segmented)
                    }
                } else {
                    Section {
                        Label(computedWinner, systemImage: "trophy.fill")
                            .foregroundStyle(.yellow)
                    } header: { Text("Winner") }
                }

                Section {
                    Text(game.startTime).foregroundStyle(.secondary)
                } header: { Text("Time") }
            }
            .navigationTitle("Edit Score")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { onDone() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        conn.setGameScore(index: game.id, homeScore: homeScore, awayScore: awayScore,
                                          winner: computedWinner, homeTeam: game.home, awayTeam: game.away)
                        onDone()
                        Task {
                            try? await Task.sleep(nanoseconds: 500_000_000)
                            conn.fetchSchedule()
                        }
                    }
                    .fontWeight(.bold)
                }
            }
        }
    }
}
