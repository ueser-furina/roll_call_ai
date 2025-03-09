import soundcard as sc
import numpy as np
import wave

def record_audio_segment(duration, filename, samplerate=44100, channels=2, mic=None):
    """
    使用 soundcard 庫錄製系統輸出音訊（loopback），並存成 WAV 檔。
    
    參數:
      duration: 錄製時長（秒）
      filename: 輸出檔案名稱
      samplerate: 採樣率（預設 44100）
      channels: 聲道數（預設 2，即立體聲）
      mic: 要使用的 microphone 物件，必須支援 loopback。
           若為 None，則會自動從 all_microphones(include_loopback=True)
           取得第一個支援 loopback 的麥克風。
    """
    if mic is None:
        mics = sc.all_microphones(include_loopback=True)
        if not mics:
            raise RuntimeError("找不到支援 loopback 的麥克風！請確認系統設定或安裝虛擬音訊裝置。")
        mic = mics[4]
    print(f"使用麥克風 (loopback): {mic.name}")

    # 錄製指定時長的音訊
    with mic.recorder(samplerate=samplerate, channels=channels) as rec:
        data = rec.record(int(samplerate * duration))
    
    # 儲存為 WAV 檔（16-bit PCM 格式）
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 2 bytes = 16-bit
        wf.setframerate(samplerate)
        wf.writeframes((data * 32767).astype(np.int16).tobytes())
    #print(f"錄製完成，存檔：{filename}")
