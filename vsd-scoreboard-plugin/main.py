"""
VSDinside plugin entry point.

VSDinside launches this with:
  -port <port> -pluginUUID <uuid> -registerEvent <event> -info <json>
"""

import argparse
import json
import signal
import sys
import time

from src.core.plugin import Plugin
from src.core.logger import Logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-port",          type=int,  required=True)
    parser.add_argument("-pluginUUID",    type=str,  required=True)
    parser.add_argument("-registerEvent", type=str,  required=True)
    parser.add_argument("-info",          type=str,  required=True)
    args = parser.parse_args()

    try:
        info = json.loads(args.info)
    except Exception:
        info = {}

    Logger.info(f"WP Scoreboard plugin starting (port={args.port})")
    time.sleep(1)   # give VSD Craft time to finish setup before we connect

    plugin = Plugin(
        port=args.port,
        plugin_uuid=args.pluginUUID,
        register_event=args.registerEvent,
        info=info,
    )

    # Keep main thread alive; daemon threads do the real work
    def _shutdown(sig, frame):
        Logger.info("Shutdown signal received")
        plugin.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
