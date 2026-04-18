import SwiftUI

struct RootView: View {
    @EnvironmentObject var conn: ScoreboardConnection
    @State private var hostInput = ""

    var body: some View {
        if conn.isConnected {
            ControlView()
                .toolbar {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button("Disconnect") { conn.disconnect() }
                            .foregroundColor(.red)
                    }
                }
        } else {
            ConnectView()
        }
    }
}
