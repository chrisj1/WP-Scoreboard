import SwiftUI

struct ChatView: View {
    @EnvironmentObject var chat: YouTubeChat
    @State private var showConnect = false
    @State private var messageText = ""
    @FocusState private var inputFocused: Bool

    var body: some View {
        NavigationStack {
            if chat.isConnected {
                connectedView
            } else {
                ContentUnavailableView {
                    Label("Chat Not Connected", systemImage: "bubble.left.and.bubble.right")
                } description: {
                    Text("Connect to a YouTube live stream to view and send messages.")
                } actions: {
                    Button("Connect…") { showConnect = true }
                        .buttonStyle(.borderedProminent)
                }
                .navigationTitle("Chat")
                .sheet(isPresented: $showConnect) {
                    ChatConnectSheet().environmentObject(chat)
                }
            }
        }
    }

    // MARK: - Connected view

    var connectedView: some View {
        VStack(spacing: 0) {
            messageList
            if !chat.accessToken.isEmpty {
                inputBar
            }
        }
        .navigationTitle(chat.channelName.isEmpty ? "YouTube Chat" : chat.channelName)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Menu {
                    Button("Settings") { showConnect = true }
                    Button("Clear Messages", role: .destructive) { chat.messages.removeAll() }
                    Button("Disconnect", role: .destructive) { chat.disconnect() }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .sheet(isPresented: $showConnect) {
            ChatConnectSheet().environmentObject(chat)
        }
    }

    var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    ForEach(chat.messages) { msg in
                        ChatMessageRow(message: msg)
                            .id(msg.id)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
            }
            .onChange(of: chat.messages.count) { _, _ in
                if let last = chat.messages.last {
                    withAnimation(.linear(duration: 0.1)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Send a message…", text: $messageText)
                .textFieldStyle(.roundedBorder)
                .focused($inputFocused)
                .onSubmit { sendMessage() }
            Button(action: sendMessage) {
                Image(systemName: "paperplane.fill")
            }
            .buttonStyle(.borderedProminent)
            .disabled(messageText.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(.systemBackground))
        .overlay(alignment: .top) { Divider() }
    }

    private func sendMessage() {
        let text = messageText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return }
        Task { await chat.sendMessage(text) }
        messageText = ""
    }
}

// MARK: - Message row

struct ChatMessageRow: View {
    let message: ChatMessage

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
            Text(message.user + ":")
                .font(.caption.bold())
                .foregroundStyle(message.isMine ? Color.blue : Color.orange)
                .fixedSize()
            Text(message.text)
                .font(.caption)
                .foregroundStyle(.primary)
                .textSelection(.enabled)
        }
    }
}

// MARK: - Connect sheet

struct ChatConnectSheet: View {
    @EnvironmentObject var chat: YouTubeChat
    @Environment(\.dismiss) var dismiss
    @State private var apiKey = ""
    @State private var videoId = ""
    @State private var accessToken = ""
    @State private var isConnecting = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("API Key", text: $apiKey)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                } header: {
                    Text("YouTube Data API Key")
                } footer: {
                    Text("Required for reading chat. Create one in Google Cloud Console.")
                }

                Section {
                    TextField("Video or broadcast ID", text: $videoId)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                } header: {
                    Text("Live Stream")
                } footer: {
                    Text("The video ID from your YouTube live stream URL.")
                }

                Section {
                    SecureField("OAuth access token (for sending)", text: $accessToken)
                } header: {
                    Text("Send Messages (optional)")
                } footer: {
                    Text("Leave blank for read-only. Requires a valid YouTube OAuth token.")
                }

                if chat.isConnected {
                    Section {
                        Button("Disconnect", role: .destructive) {
                            chat.disconnect(); dismiss()
                        }
                    }
                }
            }
            .navigationTitle("YouTube Chat")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button(isConnecting ? "Connecting…" : "Connect") {
                        if !accessToken.isEmpty { chat.saveAccessToken(accessToken) }
                        isConnecting = true
                        Task {
                            await chat.connect(apiKey: apiKey, videoId: videoId)
                            isConnecting = false
                            dismiss()
                        }
                    }
                    .fontWeight(.bold)
                    .disabled(apiKey.isEmpty || videoId.isEmpty || isConnecting)
                }
            }
            .onAppear {
                apiKey = chat.apiKey
                videoId = chat.videoId
                accessToken = chat.accessToken
            }
        }
    }
}
