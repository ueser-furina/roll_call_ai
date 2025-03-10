# 代理線上課程 Agent 專題-報告1

**作者：** 張旭孝111550146  
**學系：** 資工3B  
**日期：** 2025/2/18

---

## 1. 專題背景與動機

隨著線上課程日益普及，為了提升線上課程的效率與便利性，本專題計劃開發一個「代理線上課程的 Agent」，代理上課、實現***課堂內容摘要***功能，從而避免無法上課時的問題。

---

## 2. 專題目標

- 主要功能
   - **課堂內容摘要：** 擷取線上課程音訊，通過 ASR 模組轉換成文字，並利用大語言模型 API 自動生成課堂重點大綱。

- 附要功能
   - **自動點名：** 利用語音或視覺訊號自動識別點名要求，並回應點名操作。

---

## 3. 系統架構與技術路線

- **音訊擷取：**  
  - 直接從電腦系統音訊截取。  
  - 利用虛擬音頻驅動（如 VB-Audio Virtual Cable 或 Soundflower）捕捉線上課程的音訊流。

- **語音轉文字 (ASR)：**  
  - 選用成熟的語音辨識 API（例如 Google Cloud Speech-to-Text、Microsoft Azure Speech）將數位音訊轉換為文字，確保轉錄準確性與低延遲。

- **課堂大綱摘要生成：**  
  - 利用大語言模型 API（如 OpenAI GPT 系列）進行摘要生成。  
  - 透過分段輸入與提示工程，將課堂文字內容轉化為精簡且具重點的大綱。

- **自動點名與平台互動：**  
  - 利用自動化工具（如 Selenium 或 Puppeteer）模擬用戶操作，應對平台彈窗、按鍵點擊等要求。  
  - 結合關鍵字偵測與圖像處理技術，應對多種點名驗證方式。
---

## 4. 可能遇到的問題

- **平台介面多樣性：**  
  - **挑戰：** 不同線上課程平台介面與互動邏輯各異，代理操作可能面對困難。  

- **音訊數據處理：**  
  - **挑戰：需保證從系統音訊獲取數據的連續性與正確格式。  

- **摘要生成的精準性與 API 調用限制：**  
  - **挑戰：** 大語言模型 API 可能存在調用延遲與費用考量，且摘要品質需符合預期。  

- **自動點名的準確性：**  
  - **挑戰：** 系統需準確識別點名信號並及時回應，避免誤觸發或漏觸發。  

---

## 5. 下一步計劃

- **需求細化：** 與教授討論並進一步確認專題各功能模組的具體需求。  
- **原型開發：** 優先開發核心模組（數位音訊擷取、ASR 轉錄及摘要生成），進行初步測試與效果驗證。  
- **平台適配：** 分析目標線上課程平台，進行自動化操作模組的定制開發與測試。  

---

**附件：** 本報告作為初始報告提交，供教授審閱，期待進一步討論與指導。
