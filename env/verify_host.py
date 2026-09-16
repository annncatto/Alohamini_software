"""Check Host dependencies without opening serial ports, cameras or network ports."""


def main() -> None:
    import cv2
    import numpy as np
    import serial
    import zmq
    from scservo_sdk.protocol_packet_handler import protocol_packet_handler

    import alohamini

    with serial.Serial() as port:
        if port.is_open:
            raise RuntimeError("Unexpected open serial device")
    if protocol_packet_handler().getProtocolVersion() != 1.0:
        raise RuntimeError("Unexpected servo protocol")
    if not cv2.videoio_registry.hasBackend(cv2.CAP_V4L2):
        raise RuntimeError("OpenCV V4L2 backend unavailable")
    frame = np.full((480, 640, 3), 127, dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("JPEG encoder unavailable")
    with zmq.Context() as context:
        with context.socket(zmq.PAIR) as sender, context.socket(zmq.PAIR) as receiver:
            for socket in (sender, receiver):
                socket.linger = 0
                socket.sndtimeo = socket.rcvtimeo = 1000
            sender.bind("inproc://alohamini-host-check")
            receiver.connect("inproc://alohamini-host-check")
            sender.send_multipart([b'{"sequence":1}', jpeg.tobytes()])
            state, image = receiver.recv_multipart()
    decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if state != b'{"sequence":1}' or decoded is None or decoded.shape != frame.shape:
        raise RuntimeError("Image/transport round trip failed")
    print(f"AlohaMini: {alohamini.__file__}")
    print(f"NumPy: {np.__version__}; OpenCV: {cv2.__version__}; ZMQ: {zmq.__version__}")
    print("PASS: serial/SDK, V4L2 backend, JPEG encode/decode, ZMQ inproc")
    print("Device connections, permissions and real-time performance were not exercised.")


if __name__ == "__main__":
    main()
