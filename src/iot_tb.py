from __future__ import annotations

import json
import os
import threading

import paho.mqtt.client as mqtt


def send_to_thingsboard(server: str, port: int, token: str, telemetry: dict) -> None:
    try:
        client = mqtt.Client(
            client_id=f"ac_{os.getpid()}_{int(threading.get_ident())}",
            protocol=mqtt.MQTTv311,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
    except TypeError:
        client = mqtt.Client(
            client_id=f"ac_{os.getpid()}_{int(threading.get_ident())}",
            protocol=mqtt.MQTTv311,
        )

    client.username_pw_set(token)
    client.connect(server, int(port), keepalive=30)
    client.loop_start()

    payload = json.dumps(telemetry, ensure_ascii=False)
    info = client.publish("v1/devices/me/telemetry", payload=payload, qos=1)
    info.wait_for_publish(timeout=5)

    if not info.is_published():
        client.loop_stop()
        client.disconnect()
        raise RuntimeError("MQTT publish timeout (not published).")

    client.loop_stop()
    client.disconnect()
