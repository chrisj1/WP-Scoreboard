import SwiftUI

struct ScoreboardState {
    var homeScore: Int = 0
    var awayScore: Int = 0
    var period: Int = 1
    var periodText: String = ""
    var gameClockMinutes: Int = 8
    var gameClockSeconds: Int = 0
    var shotClock: Int = 30
    var homeTeam: String = "HOME"
    var awayTeam: String = "AWAY"
    var homeExclusions: Int = 0
    var awayExclusions: Int = 0
    var homeTimeouts: Int = 3
    var awayTimeouts: Int = 3
    var homeManup: Bool = false
    var awayManup: Bool = false
    var homeColorARGB: Int = 0xFF0080FF
    var awayColorARGB: Int = 0xFFFF8000

    var homeColor: Color { Color(argb: homeColorARGB) }
    var awayColor: Color { Color(argb: awayColorARGB) }

    var gameClock: String {
        String(format: "%d:%02d", gameClockMinutes, gameClockSeconds)
    }

    var periodLabel: String {
        periodText.isEmpty ? "Q\(period)" : periodText
    }

    mutating func apply(_ data: [String: Any]) {
        if let v = data["home_score"]        as? Int    { homeScore = v }
        if let v = data["away_score"]        as? Int    { awayScore = v }
        if let v = data["period"]            as? Int    { period = v }
        if let v = data["period_text"]       as? String { periodText = v }
        if let v = data["game_clock_minutes"]as? Int    { gameClockMinutes = v }
        if let v = data["game_clock_seconds"]as? Int    { gameClockSeconds = v }
        if let v = data["shot_clock"]        as? Int    { shotClock = v }
        if let v = data["home_team"]         as? String { homeTeam = v }
        if let v = data["away_team"]         as? String { awayTeam = v }
        if let v = data["home_exclusions"]   as? Int    { homeExclusions = v }
        if let v = data["away_exclusions"]   as? Int    { awayExclusions = v }
        if let v = data["home_timeouts"]     as? Int    { homeTimeouts = v }
        if let v = data["away_timeouts"]     as? Int    { awayTimeouts = v }
        if let v = data["home_manup"]        as? Bool   { homeManup = v }
        if let v = data["away_manup"]        as? Bool   { awayManup = v }
        if let v = data["home_color"]        as? Int    { homeColorARGB = v }
        if let v = data["away_color"]        as? Int    { awayColorARGB = v }
    }
}

// MARK: - Schedule

struct ScheduleGame: Identifiable {
    let id: Int
    var startTime: String
    var home: String
    var away: String
    var homeScore: Int?
    var awayScore: Int?
    var winner: String

    var isPlayed: Bool { homeScore != nil && awayScore != nil && !winner.isEmpty }
}

// MARK: - Team Colors

struct TeamColor: Identifiable {
    var name: String
    var homeBg: String
    var homeText: String
    var awayBg: String
    var awayText: String

    var id: String { name }
    var homeBgColor:   Color { Color(hex: homeBg) }
    var homeTextColor: Color { Color(hex: homeText) }
    var awayBgColor:   Color { Color(hex: awayBg) }
    var awayTextColor: Color { Color(hex: awayText) }
}

// MARK: - Settings

struct ScoreboardSettings {
    var showGameClock: Bool = true
    var showShotClock: Bool = true
    var defaultQuarterMinutes: Int = 8
    var defaultQuarterSeconds: Int = 0
    var smoothingFrames: Int = 3
    var shotClockModelPath: String = ""
    var gameClockModelPath: String = ""
    var clockSyncMode: Int = 0
    var cnnAvailable: Bool = false
    var configDir: String = ""
}

// MARK: - ROI

struct ROIRect {
    var x: Int = 0
    var y: Int = 0
    var width: Int = 0
    var height: Int = 0
    var isEmpty: Bool { width == 0 || height == 0 }
}

struct ROIData {
    var shotClock = ROIRect()
    var gameClock = ROIRect()
    var shotClockSource: String = ""
    var gameClockSource: String = ""
}

// MARK: - Color helpers

extension Color {
    init(argb: Int) {
        let r = Double((argb >> 16) & 0xFF) / 255
        let g = Double((argb >> 8)  & 0xFF) / 255
        let b = Double( argb        & 0xFF) / 255
        self.init(red: r, green: g, blue: b)
    }

    init(hex: String) {
        let cleaned = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var value: UInt64 = 0
        Scanner(string: cleaned).scanHexInt64(&value)
        let r = Double((value >> 16) & 0xFF) / 255
        let g = Double((value >> 8)  & 0xFF) / 255
        let b = Double( value        & 0xFF) / 255
        self.init(red: r, green: g, blue: b)
    }

    func toHex() -> String {
        let ui = UIColor(self)
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        ui.getRed(&r, green: &g, blue: &b, alpha: &a)
        return String(format: "#%02X%02X%02X", Int(r * 255), Int(g * 255), Int(b * 255))
    }
}
