import Foundation
import Combine

@MainActor
class ScoreboardConnection: ObservableObject {
    @Published var state = ScoreboardState()
    @Published var isConnected = false
    @Published var host = "" {
        didSet { UserDefaults.standard.set(host, forKey: "lastHost") }
    }

    @Published var scheduleGames: [ScheduleGame] = []
    @Published var obsSettings = ScoreboardSettings()
    @Published var roiData = ROIData()
    @Published var teamColors: [TeamColor] = []

    private var task: URLSessionWebSocketTask?
    private var pollTimer: Timer?
    private var reconnectTimer: Timer?

    init() {
        host = UserDefaults.standard.string(forKey: "lastHost") ?? ""
    }

    // MARK: - Connection

    func connect(to host: String) {
        self.host = host
        disconnect()
        guard let url = URL(string: "ws://\(host):8766") else { return }
        task = URLSession.shared.webSocketTask(with: url)
        task?.resume()
        isConnected = true
        receive()
        send(["type": "get_state"])
        startPolling()
    }

    func disconnect() {
        pollTimer?.invalidate(); pollTimer = nil
        reconnectTimer?.invalidate(); reconnectTimer = nil
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        isConnected = false
    }

    // MARK: - Polling

    private func startPolling() {
        pollTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.send(["type": "get_state"]) }
        }
    }

    // MARK: - Send / receive

    func send(_ payload: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let str  = String(data: data, encoding: .utf8) else { return }
        task?.send(.string(str)) { [weak self] error in
            if let error {
                print("WS send error: \(error)")
                Task { @MainActor in self?.handleDisconnect() }
            }
        }
    }

    private func receive() {
        task?.receive { [weak self] result in
            Task { @MainActor in
                switch result {
                case .success(let msg):
                    if case .string(let str) = msg { self?.handle(str) }
                    self?.receive()
                case .failure(let err):
                    print("WS receive error: \(err)")
                    self?.handleDisconnect()
                }
            }
        }
    }

    private func handle(_ raw: String) {
        guard let data = raw.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }

        switch json["type"] as? String {
        case "state":
            if let payload = json["data"] as? [String: Any] { state.apply(payload) }
        case "schedule":
            if let payload = json["data"] as? [String: Any] { applySchedule(payload) }
        case "settings":
            if let payload = json["data"] as? [String: Any] { applySettingsData(payload) }
        case "rois":
            if let payload = json["data"] as? [String: Any] { applyRois(payload) }
        case "teams":
            if let arr = json["data"] as? [[String: Any]] { applyTeams(arr) }
        default:
            break
        }
    }

    private func handleDisconnect() {
        guard isConnected else { return }
        isConnected = false
        pollTimer?.invalidate(); pollTimer = nil
        reconnectTimer = Timer.scheduledTimer(withTimeInterval: 3, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self, !self.host.isEmpty else { return }
                self.connect(to: self.host)
            }
        }
    }

    // MARK: - Remote data parsers

    private func applySchedule(_ data: [String: Any]) {
        guard let games = data["games"] as? [[String: Any]] else { return }
        scheduleGames = games.compactMap { g in
            guard let index = g["index"] as? Int,
                  let home  = g["home"]  as? String,
                  let away  = g["away"]  as? String else { return nil }
            return ScheduleGame(
                id: index,
                startTime: g["start_time"] as? String ?? "",
                home: home,
                away: away,
                homeScore: g["home_score"] as? Int,
                awayScore: g["away_score"] as? Int,
                winner: g["winner"] as? String ?? ""
            )
        }
    }

    private func applySettingsData(_ data: [String: Any]) {
        var s = ScoreboardSettings()
        s.showGameClock         = data["show_game_clock"]         as? Bool ?? true
        s.showShotClock         = data["show_shot_clock"]         as? Bool ?? true
        s.defaultQuarterMinutes = data["default_quarter_minutes"] as? Int  ?? 8
        s.defaultQuarterSeconds = data["default_quarter_seconds"] as? Int  ?? 0
        s.smoothingFrames       = data["smoothing_frames"]        as? Int  ?? 3
        s.shotClockModelPath    = data["shot_clock_model_path"]   as? String ?? ""
        s.gameClockModelPath    = data["game_clock_model_path"]   as? String ?? ""
        s.clockSyncMode         = data["clock_sync_mode"]         as? Int  ?? 0
        s.cnnAvailable          = data["cnn_available"]           as? Bool ?? false
        s.configDir             = data["config_dir"]              as? String ?? ""
        obsSettings = s
    }

    private func applyRois(_ data: [String: Any]) {
        var r = ROIData()
        if let shot = data["shot_clock"] as? [String: Any] {
            r.shotClock = ROIRect(x: shot["x"] as? Int ?? 0, y: shot["y"] as? Int ?? 0,
                                  width: shot["width"] as? Int ?? 0, height: shot["height"] as? Int ?? 0)
        }
        if let game = data["game_clock"] as? [String: Any] {
            r.gameClock = ROIRect(x: game["x"] as? Int ?? 0, y: game["y"] as? Int ?? 0,
                                  width: game["width"] as? Int ?? 0, height: game["height"] as? Int ?? 0)
        }
        r.shotClockSource = data["shot_clock_source"] as? String ?? ""
        r.gameClockSource = data["game_clock_source"] as? String ?? ""
        roiData = r
    }

    private func applyTeams(_ arr: [[String: Any]]) {
        teamColors = arr.compactMap { t in
            guard let name = t["name"] as? String else { return nil }
            return TeamColor(
                name: name,
                homeBg:   t["home_bg"]   as? String ?? "#000000",
                homeText: t["home_text"] as? String ?? "#FFFFFF",
                awayBg:   t["away_bg"]   as? String ?? "#000000",
                awayText: t["away_text"] as? String ?? "#FFFFFF"
            )
        }
    }

    // MARK: - Actions

    func update(_ changes: [String: Any]) {
        var payload = changes
        payload["home_score"]          = state.homeScore
        payload["away_score"]          = state.awayScore
        payload["period"]              = state.period
        payload["period_text"]         = state.periodText
        payload["game_clock_minutes"]  = state.gameClockMinutes
        payload["game_clock_seconds"]  = state.gameClockSeconds
        payload["shot_clock"]          = state.shotClock
        payload["home_team"]           = state.homeTeam
        payload["away_team"]           = state.awayTeam
        payload["home_exclusions"]     = state.homeExclusions
        payload["away_exclusions"]     = state.awayExclusions
        payload["home_timeouts"]       = state.homeTimeouts
        payload["away_timeouts"]       = state.awayTimeouts
        payload["home_manup"]          = state.homeManup
        payload["away_manup"]          = state.awayManup
        for (k, v) in changes { payload[k] = v }
        send(["type": "update", "data": payload])
    }

    func homeScoreUp()   { update(["home_score": max(0, state.homeScore + 1)]) }
    func homeScoreDown() { update(["home_score": max(0, state.homeScore - 1)]) }
    func awayScoreUp()   { update(["away_score": max(0, state.awayScore + 1)]) }
    func awayScoreDown() { update(["away_score": max(0, state.awayScore - 1)]) }

    func nextPeriod() {
        let seq: [(Int, String)] = [(1,""),(2,""),(3,""),(4,""),(5,"OT"),(5,"Final")]
        let cur = (state.period, state.periodText)
        let idx = seq.firstIndex { $0 == cur } ?? 0
        let nxt = seq[min(idx + 1, seq.count - 1)]
        update(["period": nxt.0, "period_text": nxt.1,
                "shot_clock": 30, "game_clock_minutes": 8, "game_clock_seconds": 0])
    }
    func prevPeriod() {
        let seq: [(Int, String)] = [(1,""),(2,""),(3,""),(4,""),(5,"OT"),(5,"Final")]
        let cur = (state.period, state.periodText)
        let idx = seq.firstIndex { $0 == cur } ?? 1
        let prv = seq[max(idx - 1, 0)]
        update(["period": prv.0, "period_text": prv.1])
    }
    func setFinal() { update(["period": 5, "period_text": "Final"]) }

    func resetShotClock() { update(["shot_clock": 30]) }
    func shotClock35()    { update(["shot_clock": 35]) }

    func useHomeTimeout()     { if state.homeTimeouts > 0 { update(["home_timeouts": state.homeTimeouts - 1]) } }
    func restoreHomeTimeout() { update(["home_timeouts": min(3, state.homeTimeouts + 1)]) }
    func useAwayTimeout()     { if state.awayTimeouts > 0 { update(["away_timeouts": state.awayTimeouts - 1]) } }
    func restoreAwayTimeout() { update(["away_timeouts": min(3, state.awayTimeouts + 1)]) }

    func addHomeExclusion()    { update(["home_exclusions": state.homeExclusions + 1]) }
    func clearHomeExclusions() { update(["home_exclusions": 0]) }
    func addAwayExclusion()    { update(["away_exclusions": state.awayExclusions + 1]) }
    func clearAwayExclusions() { update(["away_exclusions": 0]) }

    func toggleHomeManup() { update(["home_manup": !state.homeManup]) }
    func toggleAwayManup() { update(["away_manup": !state.awayManup]) }

    func nextGame() { send(["type": "next_game"]) }
    func prevGame() { send(["type": "prev_game"]) }
    func triggerReplay() { send(["type": "trigger_replay"]) }

    func resetGame() {
        update([
            "home_score": 0, "away_score": 0,
            "period": 1, "period_text": "",
            "game_clock_minutes": 8, "game_clock_seconds": 0,
            "shot_clock": 30,
            "home_exclusions": 0, "away_exclusions": 0,
            "home_timeouts": 3, "away_timeouts": 3,
            "home_manup": false, "away_manup": false,
        ])
    }

    // MARK: - Schedule

    func fetchSchedule() { send(["type": "get_schedule"]) }

    func setGameScore(index: Int, homeScore: Int, awayScore: Int,
                      winner: String, homeTeam: String, awayTeam: String) {
        send(["type": "set_game_score", "data": [
            "index": index, "home_score": homeScore, "away_score": awayScore,
            "winner": winner, "home_team": homeTeam, "away_team": awayTeam
        ]])
    }

    // MARK: - Settings

    func fetchSettings() { send(["type": "get_settings"]) }

    func applySettings(_ changes: [String: Any]) {
        send(["type": "set_settings", "data": changes])
    }

    // MARK: - ROI

    func fetchRois() { send(["type": "get_rois"]) }

    func setRoi(clock: String, x: Int, y: Int, width: Int, height: Int) {
        send(["type": "set_roi", "data": [
            "clock": clock, "x": x, "y": y, "width": width, "height": height
        ]])
    }

    // MARK: - Teams

    func fetchTeams() { send(["type": "get_teams"]) }

    func setTeamColor(name: String, homeBg: String, homeText: String,
                      awayBg: String, awayText: String) {
        send(["type": "set_team_color", "data": [
            "name": name, "home_bg": homeBg, "home_text": homeText,
            "away_bg": awayBg, "away_text": awayText
        ]])
    }
}
