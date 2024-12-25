from flask import Flask, jsonify
import pyaudio
import wave
import threading
import os

TEMP_FOLDER = './temp'
TRANSCRIPTION_TEMP_AUDIO_FILE = TEMP_FOLDER + "/transcription_temp_audio.wav"

app = Flask(__name__)

sampling_rate = 44100
channels = 1
chunk = 1024
audio_format = pyaudio.paInt16
recording = False
frames = []

p = pyaudio.PyAudio()


def record_audio():
    global frames, recording
    stream = p.open(
        format=audio_format,
        channels=channels,
        rate=sampling_rate,
        input=True,
        frames_per_buffer=chunk,
    )

    while recording:
        frames.append(stream.read(chunk))

    stream.stop_stream()
    stream.close()


@app.route("/start_recording", methods=["GET"])
def start_recording():
    global recording, frames
    frames.clear()
    recording = True
    threading.Thread(target=record_audio).start()
    return jsonify({"message": "Recording started"}), 200


@app.route("/stop_recording", methods=["GET"])
def stop_recording():
    global recording, frames
    recording = False

    # Ensure the directory for the transcription temp audio file exists
    os.makedirs(os.path.dirname(TEMP_FOLDER), exist_ok=True)

    with wave.open(TRANSCRIPTION_TEMP_AUDIO_FILE, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(audio_format))
        wf.setframerate(sampling_rate)
        wf.writeframes(b"".join(frames))

    frames.clear()
    return jsonify({"message": "Recording stopped"}), 200


if __name__ == "__main__":
    app.run(debug=False)
