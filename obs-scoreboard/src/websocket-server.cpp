// websocket-server.cpp
// Custom WebSocket server built on QTcpServer + QTcpSocket so the plugin can
// use OBS's bundled Qt (which does NOT include QtWebSockets).
// Implements RFC 6455 subset: text frames, ping/pong, close.

#include <QTcpServer>
#include <QTcpSocket>
#include <QHostAddress>
#include <QCryptographicHash>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <obs-module.h>
#include <obs-frontend-api.h>

// Forward declarations
void trigger_replay_snapshot();
void update_scoreboard_data(obs_data_t *data);
void select_next_game();
void select_prev_game();
QJsonObject get_schedule_json();
void set_game_score_at_index(int idx, int hs, int as_, const QString &winner);
QJsonObject get_settings_json();
void apply_settings_json(const QJsonObject &s);
QJsonObject get_rois_json();
void set_roi_data(const QString &clock, int x, int y, int w, int h);
QJsonArray get_teams_json();
void set_team_color_data(const QString &name, const QString &hbg, const QString &ht,
                         const QString &abg, const QString &at_);
#include "scoreboard-source.h"

// ── WebSocket upgrade / framing helpers ──────────────────────────────────────

static const QByteArray WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

// Build a server→client text frame (unmasked, per RFC 6455 §5.1)
static QByteArray makeTextFrame(const QByteArray &payload)
{
	QByteArray f;
	f.reserve(10 + payload.size());
	f.append(char(0x81));            // FIN + opcode 1 (text)
	qint64 len = payload.size();
	if (len <= 125) {
		f.append(char(len));
	} else if (len <= 65535) {
		f.append(char(126));
		f.append(char((len >> 8) & 0xFF));
		f.append(char( len       & 0xFF));
	} else {
		f.append(char(127));
		for (int i = 7; i >= 0; --i)
			f.append(char((len >> (i * 8)) & 0xFF));
	}
	f.append(payload);
	return f;
}

// ── Per-client connection ────────────────────────────────────────────────────

class WsClient : public QObject {
	Q_OBJECT
public:
	QTcpSocket *sock;

	explicit WsClient(QTcpSocket *s, QObject *parent = nullptr)
		: QObject(parent), sock(s)
	{
		connect(sock, &QTcpSocket::readyRead,   this, &WsClient::onData);
		connect(sock, &QTcpSocket::disconnected, this, &WsClient::onDisconnected);
	}

	QString peerAddress() const { return sock->peerAddress().toString(); }

	void sendText(const QString &text)
	{
		if (!upgraded) return;
		sock->write(makeTextFrame(text.toUtf8()));
	}

signals:
	void textReceived(const QString &text);
	void disconnected();

private slots:
	void onData()
	{
		buf.append(sock->readAll());
		if (!upgraded)
			doHandshake();
		else
			parseFrames();
	}

	void onDisconnected() { emit disconnected(); }

private:
	bool      upgraded = false;
	QByteArray buf;

	void doHandshake()
	{
		int end = buf.indexOf("\r\n\r\n");
		if (end < 0) return;

		QByteArray headers = buf.left(end);
		buf = buf.mid(end + 4);

		QByteArray key;
		for (const QByteArray &raw : headers.split('\n')) {
			QByteArray line = raw.trimmed();
			if (line.startsWith("Sec-WebSocket-Key:")) {
				key = line.mid(18).trimmed();
				break;
			}
		}
		if (key.isEmpty()) { sock->close(); return; }

		QByteArray accept = QCryptographicHash::hash(
			key + WS_GUID, QCryptographicHash::Sha1).toBase64();

		sock->write(
			"HTTP/1.1 101 Switching Protocols\r\n"
			"Upgrade: websocket\r\n"
			"Connection: Upgrade\r\n"
			"Sec-WebSocket-Accept: " + accept + "\r\n\r\n");

		upgraded = true;
		if (!buf.isEmpty()) parseFrames();
	}

	void parseFrames()
	{
		while (buf.size() >= 2) {
			quint8 byte0   = (quint8)buf[0];
			quint8 byte1   = (quint8)buf[1];
			quint8 opcode  = byte0 & 0x0F;
			bool   masked  = (byte1 & 0x80) != 0;
			qint64 payLen  = byte1 & 0x7F;
			int    hdrLen  = 2;

			if (payLen == 126) {
				if (buf.size() < 4) return;
				payLen = ((quint8)buf[2] << 8) | (quint8)buf[3];
				hdrLen = 4;
			} else if (payLen == 127) {
				if (buf.size() < 10) return;
				payLen = 0;
				for (int i = 0; i < 8; ++i)
					payLen = (payLen << 8) | (quint8)buf[2 + i];
				hdrLen = 10;
			}

			int    maskLen  = masked ? 4 : 0;
			qint64 totalLen = hdrLen + maskLen + payLen;
			if ((qint64)buf.size() < totalLen) return;

			QByteArray maskKey = masked ? buf.mid(hdrLen, 4) : QByteArray();
			QByteArray payload = buf.mid(hdrLen + maskLen, (int)payLen);

			if (masked)
				for (int i = 0; i < payload.size(); ++i)
					payload[i] = payload[i] ^ maskKey[i % 4];

			buf = buf.mid((int)totalLen);

			switch (opcode) {
			case 0x1: // text
				emit textReceived(QString::fromUtf8(payload));
				break;
			case 0x8: // close
				sock->close();
				return;
			case 0x9: { // ping → pong
				QByteArray pong;
				pong.append(char(0x8A));
				pong.append(char(payload.size()));
				pong.append(payload);
				sock->write(pong);
				break;
			}
			default: break;
			}
		}
	}
};

// ── Server ───────────────────────────────────────────────────────────────────

class WebSocketServer : public QObject {
	Q_OBJECT
public:
	explicit WebSocketServer(quint16 port = 8766, QObject *parent = nullptr)
		: QObject(parent)
	{
		server = new QTcpServer(this);
		connect(server, &QTcpServer::newConnection,
		        this,   &WebSocketServer::onNewConnection);

		if (server->listen(QHostAddress::Any, port))
			blog(LOG_INFO, "WebSocket server listening on port %d", port);
		else
			blog(LOG_ERROR, "Failed to start WebSocket server on port %d", port);
	}

	~WebSocketServer()
	{
		server->close();
		for (WsClient *c : clients) c->sock->close();
		qDeleteAll(clients);
	}

	void broadcast(const QString &text)
	{
		for (WsClient *c : clients) c->sendText(text);
	}

private slots:
	void onNewConnection()
	{
		QTcpSocket *sock = server->nextPendingConnection();
		blog(LOG_INFO, "New WebSocket client: %s",
		     sock->peerAddress().toString().toUtf8().constData());

		WsClient *client = new WsClient(sock, this);
		clients << client;

		connect(client, &WsClient::textReceived,
		        this,   &WebSocketServer::processMessage);
		connect(client, &WsClient::disconnected,
		        this,   &WebSocketServer::onDisconnected);

		// Send welcome after upgrade completes (done lazily on first text msg)
		// We store in a map so we can reply to the right client
		pendingWelcome[client] = true;
	}

	void processMessage(const QString &message)
	{
		WsClient *client = qobject_cast<WsClient *>(sender());
		if (!client) return;

		// Send welcome on first message from a newly upgraded client
		if (pendingWelcome.take(client)) {
			QJsonObject w;
			w["type"]    = "connected";
			w["message"] = "Water Polo Scoreboard WebSocket Server";
			w["version"] = "1.0.0";
			client->sendText(QJsonDocument(w).toJson(QJsonDocument::Compact));
		}

		blog(LOG_INFO, "WebSocket message: %s", message.toUtf8().constData());

		QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8());
		if (!doc.isObject()) { sendError(client, "Invalid JSON format"); return; }

		QJsonObject json = doc.object();
		QString type = json["type"].toString();

		// ── Handlers ───────────────────────────────────────────────────────
		if (type == "update") {
			QJsonObject data = json["data"].toObject();
			obs_data_t *obs_data = obs_data_create();

			if (data.contains("home_score"))
				obs_data_set_int(obs_data, "home_score", data["home_score"].toInt());
			if (data.contains("away_score"))
				obs_data_set_int(obs_data, "away_score", data["away_score"].toInt());
			if (data.contains("shot_clock"))
				obs_data_set_int(obs_data, "shot_clock", data["shot_clock"].toInt());
			if (data.contains("game_clock")) {
				QStringList parts = data["game_clock"].toString().split(":");
				if (parts.size() == 2) {
					obs_data_set_int(obs_data, "game_clock_minutes", parts[0].toInt());
					obs_data_set_int(obs_data, "game_clock_seconds", parts[1].toInt());
				}
			}
			if (data.contains("game_clock_minutes"))
				obs_data_set_int(obs_data, "game_clock_minutes", data["game_clock_minutes"].toInt());
			if (data.contains("game_clock_seconds"))
				obs_data_set_int(obs_data, "game_clock_seconds", data["game_clock_seconds"].toInt());
			if (data.contains("period"))
				obs_data_set_int(obs_data, "period", data["period"].toInt());
			if (data.contains("period_text"))
				obs_data_set_string(obs_data, "period_text",
				                    data["period_text"].toString().toUtf8().constData());
			if (data.contains("home_exclusions"))
				obs_data_set_int(obs_data, "home_exclusions", data["home_exclusions"].toInt());
			if (data.contains("away_exclusions"))
				obs_data_set_int(obs_data, "away_exclusions", data["away_exclusions"].toInt());
			if (data.contains("home_timeouts"))
				obs_data_set_int(obs_data, "home_timeouts", data["home_timeouts"].toInt());
			if (data.contains("away_timeouts"))
				obs_data_set_int(obs_data, "away_timeouts", data["away_timeouts"].toInt());
			if (data.contains("home_team"))
				obs_data_set_string(obs_data, "home_team",
				                    data["home_team"].toString().toUtf8().constData());
			if (data.contains("away_team"))
				obs_data_set_string(obs_data, "away_team",
				                    data["away_team"].toString().toUtf8().constData());
			if (data.contains("home_manup"))
				obs_data_set_bool(obs_data, "home_manup", data["home_manup"].toBool());
			if (data.contains("away_manup"))
				obs_data_set_bool(obs_data, "away_manup", data["away_manup"].toBool());

			update_scoreboard_data(obs_data);
			obs_data_release(obs_data);
			reply(client, "success", "Scoreboard updated");

		} else if (type == "get_state") {
			struct scoreboard_source *sb = get_global_scoreboard();
			if (sb) {
				QJsonObject state;
				state["home_score"]         = sb->home_score;
				state["away_score"]         = sb->away_score;
				state["period"]             = sb->period;
				state["period_text"]        = QString::fromStdString(sb->period_text);
				state["game_clock_minutes"] = sb->game_clock_minutes;
				state["game_clock_seconds"] = sb->game_clock_seconds;
				state["shot_clock"]         = sb->shot_clock;
				state["home_team"]          = QString::fromStdString(sb->home_team);
				state["away_team"]          = QString::fromStdString(sb->away_team);
				state["home_exclusions"]    = sb->home_exclusions;
				state["away_exclusions"]    = sb->away_exclusions;
				state["home_timeouts"]      = sb->home_timeouts;
				state["away_timeouts"]      = sb->away_timeouts;
				state["home_manup"]         = sb->home_manup;
				state["away_manup"]         = sb->away_manup;
				state["home_wins"]          = sb->home_wins;
				state["home_losses"]        = sb->home_losses;
				state["home_ties"]          = sb->home_ties;
				state["away_wins"]          = sb->away_wins;
				state["away_losses"]        = sb->away_losses;
				state["away_ties"]          = sb->away_ties;
				state["home_color"]         = (qint64)sb->home_color;
				state["away_color"]         = (qint64)sb->away_color;

				QJsonObject resp;
				resp["type"] = "state";
				resp["data"] = state;
				client->sendText(QJsonDocument(resp).toJson(QJsonDocument::Compact));
			} else {
				sendError(client, "Scoreboard not initialized");
			}

		} else if (type == "next_game") {
			select_next_game();
			reply(client, "success", "Advanced to next game");

		} else if (type == "prev_game") {
			select_prev_game();
			reply(client, "success", "Went to previous game");

		} else if (type == "get_schedule") {
			QJsonObject resp;
			resp["type"] = "schedule";
			resp["data"] = get_schedule_json();
			client->sendText(QJsonDocument(resp).toJson(QJsonDocument::Compact));

		} else if (type == "set_game_score") {
			QJsonObject data = json["data"].toObject();
			int idx     = data["index"].toInt();
			int hs      = data["home_score"].toInt();
			int as_     = data["away_score"].toInt();
			QString winner = data["winner"].toString();
			if (winner.isEmpty()) {
				if (hs > as_)      winner = data["home_team"].toString();
				else if (as_ > hs) winner = data["away_team"].toString();
			}
			set_game_score_at_index(idx, hs, as_, winner);
			reply(client, "success", "Game score saved");

		} else if (type == "get_settings") {
			QJsonObject resp;
			resp["type"] = "settings";
			resp["data"] = get_settings_json();
			client->sendText(QJsonDocument(resp).toJson(QJsonDocument::Compact));

		} else if (type == "set_settings") {
			apply_settings_json(json["data"].toObject());
			reply(client, "success", "Settings applied");

		} else if (type == "get_rois") {
			QJsonObject resp;
			resp["type"] = "rois";
			resp["data"] = get_rois_json();
			client->sendText(QJsonDocument(resp).toJson(QJsonDocument::Compact));

		} else if (type == "set_roi") {
			QJsonObject data = json["data"].toObject();
			set_roi_data(data["clock"].toString(),
			             data["x"].toInt(), data["y"].toInt(),
			             data["width"].toInt(), data["height"].toInt());
			reply(client, "success", "ROI updated");

		} else if (type == "get_teams") {
			QJsonObject resp;
			resp["type"] = "teams";
			resp["data"] = QJsonValue(get_teams_json());
			client->sendText(QJsonDocument(resp).toJson(QJsonDocument::Compact));

		} else if (type == "set_team_color") {
			QJsonObject data = json["data"].toObject();
			set_team_color_data(data["name"].toString(),
			                    data["home_bg"].toString(), data["home_text"].toString(),
			                    data["away_bg"].toString(), data["away_text"].toString());
			reply(client, "success", "Team color updated");

		} else if (type == "trigger_replay") {
			trigger_replay_snapshot();
			reply(client, "success", "Replay snapshot captured");

		} else if (type == "ping") {
			QJsonObject resp;
			resp["type"] = "pong";
			client->sendText(QJsonDocument(resp).toJson(QJsonDocument::Compact));

		} else {
			sendError(client, "Unknown message type: " + type);
		}
	}

	void onDisconnected()
	{
		WsClient *client = qobject_cast<WsClient *>(sender());
		blog(LOG_INFO, "WebSocket client disconnected");
		if (client) {
			clients.removeAll(client);
			pendingWelcome.remove(client);
			client->sock->deleteLater();
			client->deleteLater();
		}
	}

private:
	QTcpServer        *server;
	QList<WsClient *>  clients;
	QHash<WsClient *, bool> pendingWelcome;

	void reply(WsClient *client, const QString &type, const QString &msg)
	{
		QJsonObject r;
		r["type"]    = type;
		r["message"] = msg;
		client->sendText(QJsonDocument(r).toJson(QJsonDocument::Compact));
	}

	void sendError(WsClient *client, const QString &msg)
	{
		QJsonObject e;
		e["type"]    = "error";
		e["message"] = msg;
		client->sendText(QJsonDocument(e).toJson(QJsonDocument::Compact));
	}
};

#include "websocket-server.moc"

// ── Module API ───────────────────────────────────────────────────────────────

static WebSocketServer *g_websocket_server = nullptr;

void init_websocket_server()
{
	if (!g_websocket_server) {
		g_websocket_server = new WebSocketServer(8766);
		blog(LOG_INFO, "WebSocket server initialized on port 8766");
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
