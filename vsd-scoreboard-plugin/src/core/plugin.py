import json
import threading
import websocket
from src.core.logger import Logger
from src.core.timer import Timer
from src.core.action_factory import ActionFactory


class Plugin:
    def __init__(self, port: int, plugin_uuid: str, register_event: str, info: dict):
        self.port = port
        self.uuid = plugin_uuid
        self.register_event = register_event
        self.info = info
        self.timer = Timer()
        self._actions: dict = {}   # context -> Action instance
        self._ws = None
        self._lock = threading.Lock()
        self.get_global_settings_dict = {}

        ActionFactory.scan_and_register()

        self._ws = websocket.WebSocketApp(
            f"ws://127.0.0.1:{port}",
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        t = threading.Thread(target=self._ws.run_forever, daemon=True)
        t.start()

    # ── Public API ─────────────────────────────────────────────────

    def send(self, data: str):
        try:
            if self._ws:
                self._ws.send(data)
        except Exception as e:
            Logger.error(f"Plugin.send error: {e}")

    def set_global_settings(self, payload: dict):
        self.send(json.dumps({
            "event": "setGlobalSettings",
            "context": self.uuid,
            "payload": payload,
        }))

    def get_global_settings(self):
        self.send(json.dumps({
            "event": "getGlobalSettings",
            "context": self.uuid,
        }))

    def get_action(self, context: str):
        return self._actions.get(context)

    def get_actions(self, action_uuid: str) -> list:
        return [a for a in self._actions.values() if a.action == action_uuid]

    def stop(self):
        self.timer.stop()
        if self._ws:
            self._ws.close()

    # ── WebSocket handlers ─────────────────────────────────────────

    def _on_open(self, ws):
        Logger.info("Connected to VSDinside")
        self.send(json.dumps({
            "event": self.register_event,
            "uuid": self.uuid,
        }))

    def _on_message(self, ws, raw: str):
        try:
            msg = json.loads(raw)
        except Exception:
            Logger.error(f"Malformed message: {raw}")
            return

        event = msg.get("event", "")
        context = msg.get("context", "")
        action_uuid = msg.get("action", "")
        payload = msg.get("payload", {})

        # Events that create/destroy action instances
        if event == "willAppear":
            settings = payload.get("settings", {})
            action = ActionFactory.create(action_uuid, context, settings, self)
            if action:
                with self._lock:
                    self._actions[context] = action
                action.on_will_appear()

        elif event == "willDisappear":
            action = self._actions.get(context)
            if action:
                action.on_will_disappear()
                with self._lock:
                    self._actions.pop(context, None)

        # Route all other events to existing action instance
        else:
            action = self._actions.get(context)
            self._dispatch(event, action, payload, msg)

    def _dispatch(self, event: str, action, payload: dict, msg: dict):
        # Global settings are broadcast to all actions
        if event == "didReceiveGlobalSettings":
            self.get_global_settings_dict = payload.get("settings", {})
            for a in list(self._actions.values()):
                a.on_did_receive_global_settings(self.get_global_settings_dict)
            return

        if action is None:
            return

        handlers = {
            "keyDown":                       lambda: action.on_key_down(),
            "keyUp":                         lambda: action.on_key_up(),
            "dialDown":                      lambda: action.on_dial_down(),
            "dialUp":                        lambda: action.on_dial_up(),
            "dialRotate":                    lambda: action.on_dial_rotate(payload.get("ticks", 0)),
            "didReceiveSettings":            lambda: action.on_did_receive_settings(payload.get("settings", {})),
            "sendToPlugin":                  lambda: action.on_send_to_property_inspector(payload),
            "propertyInspectorDidAppear":    lambda: action.on_property_inspector_did_appear(),
            "propertyInspectorDidDisappear": lambda: action.on_property_inspector_did_disappear(),
            "deviceDidConnect":              lambda: action.on_device_did_connect(msg.get("deviceInfo", {})),
            "deviceDidDisconnect":           lambda: action.on_device_did_disconnect(msg.get("device", "")),
            "applicationDidLaunch":          lambda: action.on_application_did_launch(payload.get("application", "")),
            "applicationDidTerminate":       lambda: action.on_application_did_terminate(payload.get("application", "")),
            "systemDidWakeUp":               lambda: action.on_system_did_wake_up(),
        }
        handler = handlers.get(event)
        if handler:
            try:
                handler()
            except Exception as e:
                Logger.error(f"Event handler error [{event}]: {e}")
        else:
            Logger.debug(f"Unhandled event: {event}")

    def _on_error(self, ws, error):
        Logger.error(f"VSDinside WebSocket error: {error}")

    def _on_close(self, ws, code, msg):
        Logger.info(f"VSDinside WebSocket closed (code={code})")
