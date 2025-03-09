import whisper
import torch

def audio_to_text(audio_file):
    """
    使用 OpenAI 的 Whisper 模型將音訊檔（WAV 格式）轉換為文字。
    此函式支援兩種語言：
      - 中文 (zh-TW)：如果音訊為中文，直接回傳原文。
      - 英文 (en-US)：若音訊為中文，則回傳英文翻譯；若原始語言為英文，則回傳英文原文。

    Parameters:
        audio_file (str): 音訊檔案的路徑。

    Returns:
        dict: 一個字典，包含兩個鍵：
              - 'zh-TW': 中文辨識結果，
              - 'en-US': 英文辨識或翻譯結果。

              
    """
    # 檢查 GPU 是否可用，並設定運行裝置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 載入 Whisper 模型，根據需求可選擇 "base", "small", "medium", "large" 等模型
    model = whisper.load_model("base", device=device)

    # 進行自動語言辨識與轉錄
    result = model.transcribe(audio_file)
    detected_language = result.get("language", "")
    transcription = result.get("text", "").strip()

    transcriptions = {"zh-TW": "", "en-US": ""}

    if detected_language.startswith("zh"):
        # 如果偵測到音訊為中文，原文直接為中文
        transcriptions["zh-TW"] = transcription
        # 使用翻譯模式獲取英文翻譯（Whisper 的翻譯功能會將中文翻譯成英文）
        translation_result = model.transcribe(audio_file, task="translate")
        transcriptions["en-US"] = translation_result.get("text", "").strip()
    else:
        # 假設偵測到的是英文，則直接使用原文作為英文結果
        transcriptions["en-US"] = transcription
        # 若有需要，也可以利用翻譯功能將英文翻譯成中文，但通常不必要
        # translation_result = model.transcribe(audio_file, task="translate")
        # transcriptions["zh-TW"] = translation_result.get("text", "").strip()

    print(f"Detected language: {detected_language}")
    print(f"Transcription (zh-TW): {transcriptions['zh-TW']}")
    print(f"Transcription (en-US): {transcriptions['en-US']}")

    return transcriptions
