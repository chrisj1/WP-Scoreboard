import SwiftUI

struct ConnectView: View {
    @EnvironmentObject var conn: ScoreboardConnection
    @State private var hostInput = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            Image(systemName: "sportscourt.fill")
                .font(.system(size: 64))
                .foregroundColor(.blue)

            Text("WP Scoreboard")
                .font(.largeTitle.bold())

            VStack(spacing: 12) {
                Text("OBS computer IP address")
                    .font(.caption)
                    .foregroundColor(.secondary)

                TextField("192.168.1.x", text: $hostInput)
                    .textFieldStyle(.roundedBorder)
                    .keyboardType(.decimalPad)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    .focused($focused)
                    .padding(.horizontal, 40)

                Button {
                    let h = hostInput.isEmpty ? conn.host : hostInput
                    guard !h.isEmpty else { return }
                    focused = false
                    conn.connect(to: h)
                } label: {
                    Text("Connect")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                        .padding(.horizontal, 40)
                }
            }

            if !conn.host.isEmpty {
                Button("Reconnect to \(conn.host)") {
                    focused = false
                    conn.connect(to: conn.host)
                }
                .foregroundColor(.secondary)
            }

            Spacer()
        }
        .onAppear { hostInput = conn.host }
    }
}
