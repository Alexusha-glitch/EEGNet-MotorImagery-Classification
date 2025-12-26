import socket, json, time
import numpy as np
import pandas as pd
from BCITesting import EEGNet
import torch

IP = "127.0.0.1" # localhost, refers to this own computer
PORT = 12345
N_CH = 16

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((IP, PORT))
sock.settimeout(2.0)

samples = []
timestamps = []

print(f"Listening for OpenBCI GUI UDP on {IP}:{PORT} ... (Ctrl+C to stop)")

try:
    while True:
        packet, _ = sock.recvfrom(1_000_000)
        msg = json.loads(packet.decode("utf-8"))

        if msg.get("type") != "timeSeriesFilt":
            continue

        data = msg["data"]
        X = np.array(data, dtype=np.float32).T

        X = X[:, :N_CH]

        samples.append(X)
        timestamps.append(time.time())
        print(X)

except KeyboardInterrupt:
    pass
finally:
    sock.close()

X_all = np.vstack(samples)
df = pd.DataFrame(X_all, columns=[f"ch{i+1}" for i in range(N_CH)])

ckpt = torch.load("eegnet_leftright.pt", map_location="cpu")

model = EEGNet(
    n_channels=ckpt["n_channels"],
    n_classes=2,
    samples=ckpt["samples"]
)
model.load_state_dict(ckpt["model_state"])
model.eval()

