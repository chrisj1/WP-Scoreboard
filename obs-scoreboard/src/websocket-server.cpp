#include <QWebSocketServer>
#include <QWebSocket>
#include <QJsonDocument>
#include <QJsonObject>
#include <obs-module.h>

// Forward declaration from scoreboard-source.cpp
void update_scoreboard_data(obs_data_t *data);

class WebSocketServer : public QObject {
	Q_OBJECT

private:
	QWebSocketServer *server;
	QList<QWebSocket *> clients;

public:
	WebSocketServer(quint16 port = 8765, QObject *parent = nullptr) : QObject(parent) {
		server = new QWebSocketServer(QStringLiteral("Water Polo Scoreboard WebSocket Server"),
		                              QWebSocketServer::NonSecureMode, this);
		
		if (server->listen(QHostAddress::Any, port)) {
			blog(LOG_INFO, "WebSocket server listening on port %d", port);
			connect(server, &QWebSocketServer::newConnection, this, &WebSocketServer::onNewConnection);
		} else {
			blog(LOG_ERROR, "Failed to start WebSocket server on port %d", port);
		}
	}
	
	~WebSocketServer() {
		server->close();
		qDeleteAll(clients);
	}

private slots:
	void onNewConnection() {
		QWebSocket *socket = server->nextPendingConnection();
		blog(LOG_INFO, "New WebSocket client connected: %s", 
		     socket->peerAddress().toString().toUtf8().constData());
		
		connect(socket, &QWebSocket::textMessageReceived, this, &WebSocketServer::processTextMessage);
		connect(socket, &QWebSocket::disconnected, this, &WebSocketServer::socketDisconnected);
		
		clients << socket;
		
		// Send welcome message
		QJsonObject welcome;
		welcome["type"] = "connected";
		welcome["message"] = "Water Polo Scoreboard WebSocket Server";
		welcome["version"] = "1.0.0";
		
		QJsonDocument doc(welcome);
		socket->sendTextMessage(doc.toJson(QJsonDocument::Compact));
	}
	
	void processTextMessage(QString message) {
		QWebSocket *client = qobject_cast<QWebSocket *>(sender());
		
		blog(LOG_INFO, "WebSocket message received: %s", message.toUtf8().constData());
		
		// Parse JSON
		QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8());
		if (!doc.isObject()) {
			sendError(client, "Invalid JSON format");
			return;
		}
		
		QJsonObject json = doc.object();
		QString type = json["type"].toString();
		
		if (type == "update") {
			// Update scoreboard with provided data
			QJsonObject data = json["data"].toObject();
			
			obs_data_t *obs_data = obs_data_create();
			
			if (data.contains("home_score"))
				obs_data_set_int(obs_data, "home_score", data["home_score"].toInt());
			if (data.contains("away_score"))
				obs_data_set_int(obs_data, "away_score", data["away_score"].toInt());
			if (data.contains("shot_clock"))
				obs_data_set_int(obs_data, "shot_clock", data["shot_clock"].toInt());
			if (data.contains("game_clock")) {
				QString gameClock = data["game_clock"].toString();
				QStringList parts = gameClock.split(":");
				if (parts.size() == 2) {
					obs_data_set_int(obs_data, "game_minutes", parts[0].toInt());
					obs_data_set_int(obs_data, "game_seconds", parts[1].toInt());
				}
			}
			if (data.contains("game_minutes"))
				obs_data_set_int(obs_data, "game_minutes", data["game_minutes"].toInt());
			if (data.contains("game_seconds"))
				obs_data_set_int(obs_data, "game_seconds", data["game_seconds"].toInt());
			if (data.contains("period"))
				obs_data_set_int(obs_data, "period", data["period"].toInt());
			if (data.contains("home_exclusions"))
				obs_data_set_int(obs_data, "home_exclusions", data["home_exclusions"].toInt());
			if (data.contains("away_exclusions"))
				obs_data_set_int(obs_data, "away_exclusions", data["away_exclusions"].toInt());
			if (data.contains("home_timeouts"))
				obs_data_set_int(obs_data, "home_timeouts", data["home_timeouts"].toInt());
			if (data.contains("away_timeouts"))
				obs_data_set_int(obs_data, "away_timeouts", data["away_timeouts"].toInt());
			if (data.contains("home_team"))
				obs_data_set_string(obs_data, "home_team", data["home_team"].toString().toUtf8().constData());
			if (data.contains("away_team"))
				obs_data_set_string(obs_data, "away_team", data["away_team"].toString().toUtf8().constData());
			
			update_scoreboard_data(obs_data);
			obs_data_release(obs_data);
			
			// Send confirmation
			QJsonObject response;
			response["type"] = "success";
			response["message"] = "Scoreboard updated";
			QJsonDocument responseDoc(response);
			client->sendTextMessage(responseDoc.toJson(QJsonDocument::Compact));
			
		} else if (type == "ping") {
			// Respond to ping
			QJsonObject response;
			response["type"] = "pong";
			QJsonDocument responseDoc(response);
			client->sendTextMessage(responseDoc.toJson(QJsonDocument::Compact));
			
		} else {
			sendError(client, "Unknown message type: " + type);
		}
	}
	
	void socketDisconnected() {
		QWebSocket *client = qobject_cast<QWebSocket *>(sender());
		blog(LOG_INFO, "WebSocket client disconnected");
		
		if (client) {
			clients.removeAll(client);
			client->deleteLater();
		}
	}
	
	void sendError(QWebSocket *client, QString errorMessage) {
		QJsonObject error;
		error["type"] = "error";
		error["message"] = errorMessage;
		QJsonDocument doc(error);
		client->sendTextMessage(doc.toJson(QJsonDocument::Compact));
	}
};

#include "websocket-server.moc"

// Global WebSocket server instance
static WebSocketServer *g_websocket_server = nullptr;

void init_websocket_server()
{
	if (!g_websocket_server) {
		g_websocket_server = new WebSocketServer(8765);
		blog(LOG_INFO, "WebSocket server initialized on port 8765");
	}
}

void shutdown_websocket_server()
{
	if (g_websocket_server) {
		delete g_websocket_server;
		g_websocket_server = nullptr;
		blog(LOG_INFO, "WebSocket server shutdown");
	}
}
