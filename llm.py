import os
import json
import google.generativeai as genai

# Configure the Gemini API with your API key.
api_key = "AIzaSyDfxEglMsSYa3X1w5MyCeCOc5UX4IkM9zI"
if not api_key:
    raise RuntimeError("請設定環境變數 GOOGLE_API_KEY 以使用 Gemini API。")
genai.configure(api_key=api_key)

def process_text(text: str) -> dict:
    """
    接受處理好的文字，生成大綱並判斷是否為點名時機。
    - 若音訊為中文，直接回傳原文作為中文結果，並同時回傳英文翻譯；
    - 若音訊為英文，直接回傳英文原文。

    請求以 JSON 格式回應，格式如下：
      {
         "outline": ["大綱項目 1", "大綱項目 2", ...],
         "roll_call": "YES"   // 如果判斷為點名時機則回覆 "YES"，否則回覆 "NO"
      }

    Parameters:
        text (str): 處理好的文字（例如從音訊轉錄而來）

    Returns:
        dict: 包含 "outline" 與 "roll_call" 的結果。
    """
    prompt = f"""以下是一份上課的，並判斷這段文字是否包含點名時機。
請以 json 格式回應，格式如下：
{{
  "roll_call": "YES"  // 如果是點名時機則回覆 "YES"，否則回覆 "NO"
}}

文字：
{text}
"""
    try:
        # 使用 Gemini API 的 generate_content 方法進行生成
        # The model name is an example; please adjust it to the actual model available.
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    except Exception as e:
        print("Error calling Gemini API:", e)
        return {"outline": [text], "roll_call": "NO"}
    
    result_text = response.text[7:-4]
    try:
        result = json.loads(result_text)
    except Exception as e:
        print("Error parsing Gemini API response as JSON:", e)
        # Fallback: return original text as a single outline item and assume not a roll call.
        result = {"outline": [text], "roll_call": "NO"}

    
    return result

# For testing purposes:
if __name__ == "__main__":
    sample_text = (
        "同學們，剛才我們討論了科技倫理的多個面向，包括隱私權、人工智慧的倫理問題、自動化對就業的影響，以及科技產品的設計倫理。這些議題對我們未來的學習和職業生涯都有深遠的影響。我原本打算不逐一點名，但為了確保每位同學都能參與討論並深入思考，現在我決定還是點名提問，但我想一下快要下課了，因此還是不點名了。"

        "小華，你能分享一下對人工智慧偏見問題的看法嗎？接著，麗華，請談談自動化對就業影響的觀點。最後，志強，請你討論一下科技產品設計中的倫理考量。希望大家踴躍發言，讓我們的討論更深入、更有意義。"
    )
    result = process_text(sample_text)
    print("Gemini API Result:")
    print(result)
