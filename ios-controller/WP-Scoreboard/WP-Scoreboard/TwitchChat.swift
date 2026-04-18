import Foundation
import Combine
import SwiftUI

struct ChatMessage: Identifiable {
    let id = UUID()
    let user: String
    let text: String
    let avatarURL: String?
    let isMine: Bool
}

@MainActor
class YouTubeChat: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var isConnected = false
    @Published var apiKey = ""
    @Published var videoId = ""        // YouTube video/broadcast ID
    @Published var liveChatId = ""
    @Published var accessToken = ""   // OAuth — needed to post messages
    @Published var channelName = ""

    private var pollTimer: Timer?
    private var nextPageToken = ""
    private let baseURL = "https://www.googleapis.com/youtube/v3"

    init() {
        apiKey      = UserDefaults.standard.string(forKey: "ytApiKey")      ?? ""
        videoId     = UserDefaults.standard.string(forKey: "ytVideoId")     ?? ""
        accessToken = UserDefaults.standard.string(forKey: "ytAccessToken") ?? ""
    }

    // MARK: - Connect

    func connect(apiKey: String, videoId: String) async {
        self.apiKey  = apiKey
        self.videoId = videoId.trimmingCharacters(in: .whitespaces)
        UserDefaults.standard.set(apiKey,  forKey: "ytApiKey")
        UserDefaults.standard.set(videoId, forKey: "ytVideoId")

        stopPolling()
        messages = []
        nextPageToken = ""

        do {
            try await fetchLiveChatId()
            isConnected = true
            startPolling()
        } catch {
            isConnected = false
            print("YouTube connect error: \(error)")
        }
    }

    func disconnect() {
        stopPolling()
        isConnected = false
        liveChatId = ""
        messages = []
        nextPageToken = ""
    }

    // MARK: - Fetch live chat ID

    private func fetchLiveChatId() async throws {
        guard !apiKey.isEmpty, !videoId.isEmpty else {
            throw URLError(.badURL)
        }
        var comps = URLComponents(string: "\(baseURL)/videos")!
        comps.queryItems = [
            .init(name: "id",   value: videoId),
            .init(name: "part", value: "liveStreamingDetails,snippet"),
            .init(name: "key",  value: apiKey),
        ]
        let (data, _) = try await URLSession.shared.data(from: comps.url!)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let items = json?["items"] as? [[String: Any]]
        let details = items?.first?["liveStreamingDetails"] as? [String: Any]
        let snippet = items?.first?["snippet"] as? [String: Any]
        channelName = snippet?["channelTitle"] as? String ?? ""
        guard let id = details?["activeLiveChatId"] as? String else {
            throw URLError(.resourceUnavailable)
        }
        liveChatId = id
    }

    // MARK: - Polling

    private func startPolling() {
        pollTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in
            Task { await self?.pollMessages() }
        }
        Task { await pollMessages() }
    }

    private func stopPolling() {
        pollTimer?.invalidate(); pollTimer = nil
    }

    private func pollMessages() async {
        guard !liveChatId.isEmpty, !apiKey.isEmpty else { return }
        var comps = URLComponents(string: "\(baseURL)/liveChatMessages")!
        var items: [URLQueryItem] = [
            .init(name: "liveChatId", value: liveChatId),
            .init(name: "part",       value: "snippet,authorDetails"),
            .init(name: "key",        value: apiKey),
            .init(name: "maxResults", value: "200"),
        ]
        if !nextPageToken.isEmpty { items.append(.init(name: "pageToken", value: nextPageToken)) }
        comps.queryItems = items

        do {
            let (data, _) = try await URLSession.shared.data(from: comps.url!)
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            nextPageToken = json?["nextPageToken"] as? String ?? nextPageToken
            let msgs = (json?["items"] as? [[String: Any]]) ?? []
            let newMessages = msgs.compactMap { parseMessage($0) }
            if !newMessages.isEmpty {
                messages.append(contentsOf: newMessages)
                if messages.count > 300 { messages = Array(messages.suffix(200)) }
            }
        } catch {
            print("Poll error: \(error)")
        }
    }

    private func parseMessage(_ item: [String: Any]) -> ChatMessage? {
        let snippet      = item["snippet"]       as? [String: Any]
        let author       = item["authorDetails"] as? [String: Any]
        let details      = snippet?["textMessageDetails"] as? [String: Any]
        guard let text   = details?["messageText"] as? String,
              let name   = author?["displayName"]  as? String else { return nil }
        let avatar = author?["profileImageUrl"] as? String
        return ChatMessage(user: name, text: text, avatarURL: avatar, isMine: false)
    }

    // MARK: - Send message (requires OAuth access token)

    func sendMessage(_ text: String) async {
        guard isConnected, !accessToken.isEmpty, !liveChatId.isEmpty else { return }
        guard let url = URL(string: "\(baseURL)/liveChatMessages?part=snippet") else { return }

        let body: [String: Any] = ["snippet": [
            "liveChatId": liveChatId,
            "type": "textMessageEvent",
            "textMessageDetails": ["messageText": text]
        ]]

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)

        do {
            let (data, resp) = try await URLSession.shared.data(for: req)
            if let http = resp as? HTTPURLResponse, http.statusCode == 200 {
                if let msg = parseMessage((try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]) {
                    messages.append(ChatMessage(user: msg.user, text: text, avatarURL: nil, isMine: true))
                } else {
                    // Show optimistically
                    messages.append(ChatMessage(user: "You", text: text, avatarURL: nil, isMine: true))
                }
            }
        } catch {
            print("Send error: \(error)")
        }
    }

    func saveAccessToken(_ token: String) {
        accessToken = token
        UserDefaults.standard.set(token, forKey: "ytAccessToken")
    }
}
