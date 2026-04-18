import json
from src.core.logger import Logger


class Action:
    """Base class for all VSDinside plugin actions."""

    def __init__(self, action: str, context: str, settings: dict, plugin):
        self.action = action        # UUID of this action type
        self.context = context      # Unique instance context
        self.settings = settings or {}
        self.plugin = plugin

    # ── Display helpers ────────────────────────────────────────────

    def set_title(self, text: str, target: int = 0):
        self._send({
            "event": "setTitle",
            "context": self.context,
            "payload": {"title": text, "target": target},
        })

    def set_image(self, image: str, target: int = 0):
        """image can be a data:image/png;base64,... string or a file path."""
        self._send({
            "event": "setImage",
            "context": self.context,
            "payload": {"image": image, "target": target},
        })

    def set_state(self, state: int):
        self._send({
            "event": "setState",
            "context": self.context,
            "payload": {"state": state},
        })

    def show_ok(self):
        # showOk tells StreamDock to flash a checkmark then RESTORE the
        # pre-press image, which overwrites our live score badge update.
        # So we suppress it — the badge value changing is the feedback.
        pass

    def show_alert(self):
        self._send({"event": "showAlert", "context": self.context})

    def set_settings(self, settings: dict):
        self.settings.update(settings)
        self._send({
            "event": "setSettings",
            "context": self.context,
            "payload": settings,
        })

    def send_to_property_inspector(self, data: dict):
        self._send({
            "event": "sendToPropertyInspector",
            "action": self.action,
            "context": self.context,
            "payload": data,
        })

    def log_message(self, message: str):
        self._send({"event": "logMessage", "payload": {"message": message}})

    def open_url(self, url: str):
        self._send({"event": "openUrl", "payload": {"url": url}})

    def _send(self, payload: dict):
        try:
            self.plugin.send(json.dumps(payload))
        except Exception as e:
            Logger.error(f"Action._send error: {e}")

    # ── Lifecycle / event handlers (override in subclass) ──────────

    def on_will_appear(self):               pass
    def on_will_disappear(self):            pass
    def on_key_down(self):                  pass
    def on_key_up(self):                    pass
    def on_dial_down(self):                 pass
    def on_dial_up(self):                   pass
    def on_dial_rotate(self, ticks: int):   pass
    def on_did_receive_settings(self, settings: dict):        pass
    def on_did_receive_global_settings(self, settings: dict): pass
    def on_send_to_property_inspector(self, data: dict):      pass
    def on_property_inspector_did_appear(self):               pass
    def on_property_inspector_did_disappear(self):            pass
    def on_device_did_connect(self, device: dict):            pass
    def on_device_did_disconnect(self, device: dict):         pass
    def on_application_did_launch(self, app: str):            pass
    def on_application_did_terminate(self, app: str):         pass
    def on_system_did_wake_up(self):                          pass
