import SwiftUI

struct OBSView: View {
    @EnvironmentObject var obs: OBSConnection
    @State private var showConnect = false
    @State private var showStopStreamConfirm = false

    var body: some View {
        NavigationStack {
            if obs.isConnected {
                connectedView
            } else {
                ContentUnavailableView {
                    Label("Not Connected to OBS", systemImage: "tv.slash")
                } description: {
                    Text("Connect to OBS WebSocket (port 4455) to control scenes and streaming.")
                } actions: {
                    Button("Connect…") { showConnect = true }
                        .buttonStyle(.borderedProminent)
                }
                .navigationTitle("OBS")
                .sheet(isPresented: $showConnect) {
                    OBSConnectSheet().environmentObject(obs)
                }
            }
        }
    }

    // MARK: - Connected view

    var connectedView: some View {
        List {
            streamSection
            recordSection
            scenesSection
        }
        .navigationTitle("OBS")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Menu {
                    Button("Refresh") {
                        obs.fetchScenes()
                        obs.fetchStreamStatus()
                        obs.fetchRecordStatus()
                    }
                    Button("Disconnect", role: .destructive) { obs.disconnect() }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .confirmationDialog("Stop streaming?", isPresented: $showStopStreamConfirm, titleVisibility: .visible) {
            Button("Stop Stream", role: .destructive) { obs.stopStream() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will end the live stream for all viewers.")
        }
    }

    // MARK: - Stream section

    var streamSection: some View {
        Section {
            HStack(spacing: 12) {
                Circle()
                    .fill(obs.isStreaming ? Color.red : Color.gray)
                    .frame(width: 10, height: 10)
                    .overlay {
                        if obs.isStreaming {
                            Circle().fill(Color.red).scaleEffect(1.5).opacity(0.3)
                                .animation(.easeInOut(duration: 1).repeatForever(), value: obs.isStreaming)
                        }
                    }
                Text(obs.isStreaming ? "LIVE" : "Offline")
                    .font(.headline)
                    .foregroundStyle(obs.isStreaming ? .red : .secondary)
                Spacer()
                if obs.isStreaming {
                    Button("End Stream") { showStopStreamConfirm = true }
                        .buttonStyle(.borderedProminent).tint(.red)
                } else {
                    Button("Go Live") { obs.startStream() }
                        .buttonStyle(.borderedProminent).tint(.green)
                }
            }
        } header: { Text("Stream") }
    }

    // MARK: - Recording section

    var recordSection: some View {
        Section {
            HStack(spacing: 12) {
                Image(systemName: obs.isRecording
                      ? (obs.isRecordingPaused ? "pause.circle.fill" : "record.circle.fill")
                      : "stop.circle")
                    .foregroundStyle(obs.isRecording
                                     ? (obs.isRecordingPaused ? .orange : .red)
                                     : .secondary)
                    .font(.title3)
                Text(obs.isRecording
                     ? (obs.isRecordingPaused ? "Paused" : "Recording")
                     : "Not recording")
                    .font(.headline)
                    .foregroundStyle(obs.isRecording ? .primary : .secondary)
                Spacer()
                if obs.isRecording {
                    Button(obs.isRecordingPaused ? "Resume" : "Pause") {
                        obs.isRecordingPaused ? obs.resumeRecord() : obs.pauseRecord()
                    }
                    .buttonStyle(.bordered)
                    Button("Stop") { obs.stopRecord() }
                        .buttonStyle(.borderedProminent).tint(.red)
                } else {
                    Button("Record") { obs.startRecord() }
                        .buttonStyle(.bordered).tint(.red)
                }
            }
        } header: { Text("Recording") }
    }

    // MARK: - Scenes section

    var scenesSection: some View {
        Section {
            if obs.scenes.isEmpty {
                Text("No scenes found").foregroundStyle(.secondary)
            } else {
                ForEach(obs.scenes, id: \.self) { scene in
                    Button {
                        obs.switchScene(scene)
                    } label: {
                        HStack {
                            Image(systemName: scene == obs.currentScene
                                  ? "checkmark.circle.fill" : "circle")
                                .foregroundStyle(scene == obs.currentScene ? .blue : .secondary)
                            Text(scene)
                                .foregroundStyle(.primary)
                                .fontWeight(scene == obs.currentScene ? .semibold : .regular)
                            Spacer()
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
        } header: { Text("Scenes") }
    }
}

// MARK: - Connect sheet

struct OBSConnectSheet: View {
    @EnvironmentObject var obs: OBSConnection
    @Environment(\.dismiss) var dismiss
    @State private var host = ""
    @State private var password = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("OBS Computer IP") {
                    TextField("192.168.x.x", text: $host)
                        .keyboardType(.numbersAndPunctuation)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                }
                Section("WebSocket Password") {
                    SecureField("Password (leave blank if none)", text: $password)
                }
                Section {
                    Text("Enable in OBS → Tools → WebSocket Server Settings. Default port is 4455.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Connect to OBS")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Connect") {
                        obs.connect(host: host, password: password)
                        dismiss()
                    }
                    .fontWeight(.bold)
                    .disabled(host.isEmpty)
                }
            }
            .onAppear { host = obs.host; password = obs.password }
        }
    }
}
