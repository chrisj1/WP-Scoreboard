import SwiftUI

@main
struct WPScoreboardApp: App {
    @StateObject private var connection = ScoreboardConnection()
    @StateObject private var obs = OBSConnection()
    @StateObject private var chat = YouTubeChat()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(connection)
                .environmentObject(obs)
                .environmentObject(chat)
                .preferredColorScheme(.dark)
        }
    }
}
