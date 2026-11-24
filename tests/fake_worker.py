import json, sys, tempfile, wave
print("ruido de librería que no es JSON")
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req.get("cmd") == "quit":
        break
    if req["text"] == "falla":
        print(json.dumps({"error": "boom"}), flush=True)
        continue
    if req["text"] == "stream":
        for i in range(3):
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); f.close()
            with wave.open(f.name, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b"\0\0" * 1000)
            print(json.dumps({"ok": True, "path": f.name, "last": False}), flush=True)
        print(json.dumps({"ok": True, "last": True}), flush=True)
        continue
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); f.close()
    with wave.open(f.name, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b"\0\0" * 2400)
    print(json.dumps({"ok": True, "path": f.name}), flush=True)
