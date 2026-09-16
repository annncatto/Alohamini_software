"""Check PC development dependencies without downloading models or opening robot devices."""

import tempfile
from pathlib import Path


def main() -> None:
    import av
    import cv2
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import torchvision
    from torchcodec.decoders import VideoDecoder

    import alohamini

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this learning environment")
    device = torch.device("cuda")
    # Exercise convolution, backpropagation and an optimizer, not just GPU enumeration.
    network = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU(), torch.nn.Flatten())
    network.to(device)
    optimizer = torch.optim.AdamW(network.parameters())
    loss = network(torch.randn(2, 3, 32, 32, device=device)).square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite CUDA test result")
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device=device)
    kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5)
    if kept.tolist() != [0]:
        raise RuntimeError("Torchvision CUDA operator test failed")

    with tempfile.TemporaryDirectory(prefix="alohamini-learning-check-") as directory:
        root = Path(directory)
        torch.save(network.state_dict(), root / "weights.pt")
        network.load_state_dict(
            torch.load(root / "weights.pt", weights_only=True, map_location=device)
        )
        image = np.full((64, 64, 3), 127, dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", image)
        if not ok or cv2.imdecode(jpeg, cv2.IMREAD_COLOR).shape != image.shape:
            raise RuntimeError("JPEG round trip failed")
        table = pa.table({"timestamp": [0.0], "observation.state": [[0.1, 0.2]]})
        pq.write_table(table, root / "frame.parquet")
        if not pq.read_table(root / "frame.parquet").equals(table):
            raise RuntimeError("Parquet round trip failed")
        with av.open(str(root / "video.mp4"), mode="w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width = stream.height = 64
            stream.pix_fmt = "yuv420p"
            for _ in range(3):
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        decoder = VideoDecoder(str(root / "video.mp4"), device="cpu")
        if len(decoder) != 3 or tuple(decoder[0].shape) != (3, 64, 64):
            raise RuntimeError("Video round trip failed")
        del decoder

    print(f"AlohaMini: {alohamini.__file__}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        "PASS: CUDA forward/backward/optimizer, local checkpoint, torchvision, JPEG, Parquet, video"
    )
    print("Robot connections, dataset semantics and full policy training were not exercised.")


if __name__ == "__main__":
    main()
