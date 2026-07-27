from core.session_recording import CLSessionRecorder, SessionRecordingConfig


class _FakeRecording:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class _FakeStream:
    def __init__(self):
        self.points = []

    def append(self, timestamp, data):
        self.points.append((timestamp, data))


class _FakeNeurons:
    def __init__(self):
        self.stream = _FakeStream()
        self.recording = _FakeRecording()
        self.record_kwargs = None

    def create_data_stream(self, name, attributes=None):
        assert name == "senxe_control"
        assert attributes["protocol_id"] == "test"
        return self.stream

    def record(self, **kwargs):
        self.record_kwargs = kwargs
        return self.recording

    def timestamp(self):
        return 123


def test_session_recorder_synchronizes_json_telemetry():
    neurons = _FakeNeurons()
    recorder = CLSessionRecorder(
        neurons,
        SessionRecordingConfig(enabled=True, include_raw_samples=False),
        attributes={"protocol_id": "test"},
    )

    recorder.start()
    recorder.append({"step": 2, "phase": "frozen"})
    recorder.stop()

    assert neurons.record_kwargs["include_spikes"]
    assert neurons.record_kwargs["include_stims"]
    assert not neurons.record_kwargs["include_raw_samples"]
    assert neurons.recording.stopped
    assert neurons.stream.points == [
        (123, '{"phase":"frozen","step":2}')
    ]
