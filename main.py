import soundcard as sc
import numpy as np
import wave
import threading
import time
from collections import deque
import os
import audio_to_text
import warnings
import llm
from soundcard import SoundcardRuntimeWarning\

# ---- IgnoreWarning ----
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)
# ---- Configuration ----
SAMPLERATE = 44100        # samples per second
CHANNELS = 2              # stereo
SEGMENT_DURATION = 10     # seconds per segment
OVERLAP = 4               # seconds overlap between segments

SEGMENT_SAMPLES = int(SAMPLERATE * SEGMENT_DURATION)
STEP_SAMPLES = int(SAMPLERATE * (SEGMENT_DURATION - OVERLAP))
BUFFER_MAXLEN = int(SAMPLERATE * SEGMENT_DURATION * 5)

audio_buffer = deque(maxlen=BUFFER_MAXLEN)

def continuous_recording(buffer: deque, mic):
    """Continuously record audio blocks and append them to the ring buffer."""
    block_size = 1024  # number of samples per block
    with mic.recorder(samplerate=SAMPLERATE, channels=CHANNELS, blocksize=block_size) as rec:
        while True:
            block = rec.record(block_size)
            buffer.extend(block)

def segment_processor(buffer):
    """Extract overlapping segments from the buffer and process them."""
    segment_counter = 0
    
    while True:
        if len(buffer) >= SEGMENT_SAMPLES:
            print(len(buffer))
            segment_data = np.array(list(buffer))[-SEGMENT_SAMPLES:]
            filename = os.path.join("audio_segments", f"segment_{segment_counter:03d}.wav")
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLERATE)
                wf.writeframes((segment_data * 32767).astype(np.int16).tobytes())
            #print(f"[Segment {segment_counter}] Saved {filename}")
            transcriptions = audio_to_text.audio_to_text(filename)
            #print(f"[Segment {segment_counter}] Transcriptions: {transcriptions}")
            responseOfAi = llm.process_text(transcriptions["en-US"])
            print(responseOfAi)
            segment_counter += 1
            time.sleep(SEGMENT_DURATION - OVERLAP)
        else:
            time.sleep(0.1)

def main():
    os.makedirs("audio_segments", exist_ok=True)
    
    # Instead of default_speaker(), get a microphone that supports loopback.
    mics = sc.all_microphones(include_loopback=True)
    if not mics:
        raise RuntimeError("No loopback microphones found! Ensure your system supports loopback or install a virtual audio cable.")
    # Choose one (for example, the first one)
    mic = mics[4]
    print(f"Using loopback microphone: {mic.name}")
    
    rec_thread = threading.Thread(target=continuous_recording, args=(audio_buffer, mic), daemon=True)
    rec_thread.start()
    
    proc_thread = threading.Thread(target=segment_processor, args=(audio_buffer,), daemon=True)
    proc_thread.start()
    
    print("Continuous recording started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Recording stopped.")

if __name__ == '__main__':
    main()
