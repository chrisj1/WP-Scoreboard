import Foundation
import Combine
import CryptoKit

@MainActor
class OBSConnection: ObservableObject {
    @Published var isConnected = false
    @Published var scenes: [String] = []
    @Published var currentScene = ""
    @Published var isStreaming = false
    @Published var isRecording = false
    @Published var isRecordingPaused = false
    @Published var host = ""
    @Published var password = ""

    private var task: URLSessionWebSocketTask?
    private var pendingRequests: [String: ([String: Any]) -> Void] = [:]
    private var reconnectTimer: Timer?

    init() {
        host     = UserDefaults.standard.string(forKey: "obsHost")     ?? ""
        password = UserDefaults.standard.string(forKey: "obsPassword") ?? ""
    }

    // MARK: - Connection

    func connect(host: String, password: String) {
        self.host     = host
        self.password = password
        UserDefaults.standard.set(host,     forKey: "obsHost")
        UserDefaults.standard.set(password, forKey: "obsPassword")
        reconnectTimer?.invalidate()
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        isConnected = false
        guard let url = URL(string: "ws://\(host):4455") else { return }
        task = URLSession.shared.webSocketTask(with: url)
        task?.resume()
        receive()
    }

    func disconnect() {
        reconnectTimer?.invalidate(); reconnectTimer = nil
        task?.cancel(with: .goingAway, reason: nil); task = nil
        isConnected = false
        scenes = []; currentScene = ""
    }

    // MARK: - Send / receive

    private func send(_ payload: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let str = String(data: data, encoding: .utf8) else { return }
        task?.send(.string(str)) { _ in }
    }

    private func receive() {
        task?.receive { [weak self] result in
            Task { @MainActor in
                switch result {
                case .success(let msg):
                    if case .string(let str) = msg { self?.handle(str) }
                    self?.receive()
                case .failure:
                    self?.handleDisconnect()
                }
            }
        }
    }

    // MARK: - Protocol handling

    private func handle(_ raw: String) {
        guard let data = raw.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let op   = json["op"] as? Int,
              let d    = json["d"]  as? [String: Any] else { return }

        switch op {
        case 0: identify(hello: d)            // Hello
        case 2:                                // Identified
            isConnected = true
            fetchScenes()
            fetchStreamStatus()
            fetchRecordStatus()
        case 4: handleEvent(d)                 // Event
        case 6:                                // RequestResponse
            if let id = d["requestId"] as? String,
               let h  = pendingRequests.removeValue(forKey: id) {
                h(d["responseData"] as? [String: Any] ?? [:])
            }
        default: break
        }
    }

    private func identify(hello: [String: Any]) {
        // Subscribe to Scenes (1<<2=4) + Outputs (1<<6=64)
        var body: [String: Any] = ["rpcVersion": 1, "eventSubscriptions": 68]

        if let auth      = hello["authentication"] as? [String: Any],
           let challenge = auth["challenge"] as? String,
           let salt      = auth["salt"]      as? String,
           !password.isEmpty {
            body["authentication"] = computeAuth(password: password, salt: salt, challenge: challenge)
        }
        send(["op": 1, "d": body])
    }

    private func computeAuth(password: String, salt: String, challenge: String) -> String {
        let step1  = Data(SHA256.hash(data: Data((password + salt).utf8))).base64EncodedString()
        let step2  = Data(SHA256.hash(data: Data((step1 + challenge).utf8)))
        return step2.base64EncodedString()
    }

    private func request(_ type: String, data: [String: Any] = [:],
                         completion: @escaping ([String: Any]) -> Void = { _ in }) {
        let id = UUID().uuidString
        pendingRequests[id] = completion
        send(["op": 5, "d": ["requestType": type, "requestId": id, "requestData": data]])
    }

    private func handleEvent(_ d: [String: Any]) {
        guard let eventType = d["eventType"] as? String else { return }
        let eventData = d["eventData"] as? [String: Any] ?? [:]

        switch eventType {
        case "CurrentProgramSceneChanged":
            currentScene = eventData["sceneName"] as? String ?? currentScene
        case "SceneListChanged":
            fetchScenes()
        case "StreamStateChanged":
            isStreaming = eventData["outputActive"] as? Bool ?? isStreaming
        case "RecordStateChanged":
            isRecording       = eventData["outputActive"] as? Bool ?? isRecording
            isRecordingPaused = eventData["outputPaused"] as? Bool ?? isRecordingPaused
        default:
            break
        }
    }

    private func handleDisconnect() {
        guard isConnected || task != nil else { return }
        isConnected = false
        task = nil
        reconnectTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self, !self.host.isEmpty else { return }
                self.connect(host: self.host, password: self.password)
            }
        }
    }

    // MARK: - Scenes

    func fetchScenes() {
        request("GetSceneList") { [weak self] data in
            guard let self else { return }
            if let list = data["scenes"] as? [[String: Any]] {
                self.scenes = list.compactMap { $0["sceneName"] as? String }.reversed()
            }
            self.currentScene = data["currentProgramSceneName"] as? String ?? self.currentScene
        }
    }

    func switchScene(_ name: String) {
        request("SetCurrentProgramScene", data: ["sceneName": name])
        currentScene = name // optimistic
    }

    // MARK: - Stream

    func fetchStreamStatus() {
        request("GetStreamStatus") { [weak self] data in
            self?.isStreaming = data["outputActive"] as? Bool ?? false
        }
    }

    func startStream() { request("StartStream") { [weak self] _ in self?.isStreaming = true  } }
    func stopStream()  { request("StopStream")  { [weak self] _ in self?.isStreaming = false } }

    // MARK: - Recording

    func fetchRecordStatus() {
        request("GetRecordStatus") { [weak self] data in
            self?.isRecording       = data["outputActive"] as? Bool ?? false
            self?.isRecordingPaused = data["outputPaused"] as? Bool ?? false
        }
    }

    func startRecord()  { request("StartRecord")  { [weak self] _ in self?.isRecording = true;  self?.isRecordingPaused = false } }
    func stopRecord()   { request("StopRecord")   { [weak self] _ in self?.isRecording = false; self?.isRecordingPaused = false } }
    func pauseRecord()  { request("PauseRecord")  { [weak self] _ in self?.isRecordingPaused = true  } }
    func resumeRecord() { request("ResumeRecord") { [weak self] _ in self?.isRecordingPaused = false } }
}
