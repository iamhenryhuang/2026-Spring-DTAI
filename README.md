# 深度學習農作物病蟲害影像辨識
> **基於深度學習 ResNet-18 的即時影像辨識解決方案**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![ResNet18](https://img.shields.io/badge/Model-ResNet--18-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-97.2%25-brightgreen)

本專案利用 ResNet 遷移學習技術，實現高精度的自動化農作物病害檢測。透過兩階段訓練策略（凍結特徵提取層與全層微調），有效提升了模型在複雜病斑上的辨識能力。

## 訓練成果
本模型在驗證集上表現極其優異，尤其在第二階段 Fine-tuning 後取得了顯著突破：
- 初始驗證準確度：約 94% (僅訓練全連接層)
- 最終驗證準確度 (Fine-tuning)：**97.2%**
- 分類表現：模型對於 15 種常見作物病害（如蘋果黑星病、番茄晚疫病等）展現了極強的泛化能力，F1-score 穩定於高位。

## 技術規格
### 1. 訓練策略：兩階段遷移學習
為了達到最佳收斂效果，專案採用了以下步驟：
* 第一階段 (Transfer Learning)：凍結 ResNet-18 的卷積層，僅訓練輸出的 Linear Head。
* 第二階段 (Fine-tuning)：解凍模型參數，以極小的學習率進行全局微調，使模型能更精準地捕捉植物葉片的細微病變特徵，最終將 Accuracy 推升至 97% 以上。

### 2. 數據處理與模型架構
- 模型基礎：ResNet-18 (ImageNet Pre-trained)。
- 影像尺寸：輸入解析度調整為 `299x299`。
- 優化器配置：使用 Adam 優化器，搭配學習率衰減策略 (StepLR) 以確保細節微調時的穩定性。
- 數據集規模：[PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)，包含超過 20,000 張標註影像。

## 核心功能
- 自動化分類：精準辨識包含黑斑病、銹病、白粉病等多種常見作物病害。
- 信心度回傳：輸出疾病類別並附帶信心程度，同時列出 Top-3 候選結果。
- 高效推理：ResNet-18 的輕量化特性，使其在邊緣設備端也能維持快速反應。
- AI 照護建議：辨識完成後自動呼叫 Groq API（llama-3.3-70b-versatile），產生針對病害的中文照護建議。
- 三欄式介面：影像上傳、診斷結果、AI 建議並排顯示於同一面板，一目瞭然。

---

## 專案結構

```
2026-Spring-DTAI/
├── app.py                   # Flask 後端推論伺服器
├── requirements.txt         # Python 依賴套件清單
├── Dockerfile               # 容器化設定
├── docker-compose.yml
├── .dockerignore
├── templates/
│   └── index.html           # 前端網頁介面
└── train/
    ├── Train.ipynb          # 模型訓練流程
    └── plantvillage.ipynb   # 資料集探索
```

---

## 環境建置

### 1. 建立虛擬環境

> 需要 Python 3.8 以上版本。

```bash
# 建立虛擬環境
python -m venv .venv

# 啟動（Windows bash / Git Bash）
source .venv/Scripts/activate

# 啟動（macOS / Linux）
source .venv/bin/activate
```

### 2. 安裝依賴套件

在虛擬環境啟動後執行：

```bash
pip install -r requirements.txt
```

### 3. 設定 Groq API Key（選用）

若要在辨識完成後產生照護建議，請先設定 `GROQ_API_KEY`。API key 可至 [console.groq.com](https://console.groq.com) 免費申請，會由 Flask 後端讀取，不會放在前端程式碼中。

```powershell
$env:GROQ_API_KEY="你的 Groq API Key"
```

可選擇指定模型，未設定時預設使用 `llama-3.3-70b-versatile`：

```powershell
$env:GROQ_MODEL="llama-3.3-70b-versatile"
```

---

## 啟動 Web 介面

本專案建議透過 Docker 啟動，容器首次執行時會自動從 HuggingFace 下載模型權重，無需手動準備。

```bash
docker compose up --build
```

啟動成功後，開啟瀏覽器前往 **http://localhost:5000** 即可使用。

> 詳細 Docker 指令請參考下方 **Docker 容器化部署** 章節。

---

## Docker 容器化部署

**前置條件：** 安裝 [Docker Desktop](https://www.docker.com/products/docker-desktop/)，並在專案根目錄建立 `.env`：

```
GROQ_API_KEY=你的key
```

| 動作 | 指令 |
|------|------|
| 首次啟動（自動 build，約 5–10 分鐘） | `docker compose up --build` |
| 日常啟動 | `docker compose up` |
| 背景執行 | `docker compose up -d` |
| 停止 | `Ctrl+C` 或 `docker compose down` |
| 完整清除（含 image，釋放約 1.1 GB） | `docker compose down --rmi all` |

啟動後開啟瀏覽器前往 **http://localhost:5000**。

---

### 可辨識的 15 種病害類別

| # | 類別 |
|---|------|
| 1 | Pepper Bell - Bacterial Spot（甜椒細菌性斑點病） |
| 2 | Pepper Bell - Healthy（甜椒健康） |
| 3 | Potato - Early Blight（馬鈴薯早疫病） |
| 4 | Potato - Late Blight（馬鈴薯晚疫病） |
| 5 | Potato - Healthy（馬鈴薯健康） |
| 6 | Tomato - Bacterial Spot（番茄細菌性斑點病） |
| 7 | Tomato - Early Blight（番茄早疫病） |
| 8 | Tomato - Late Blight（番茄晚疫病） |
| 9 | Tomato - Leaf Mold（番茄葉黴病） |
| 10 | Tomato - Septoria Leaf Spot（番茄斑枯病） |
| 11 | Tomato - Spider Mites（番茄二斑葉蟎） |
| 12 | Tomato - Target Spot（番茄靶斑病） |
| 13 | Tomato - Yellow Leaf Curl Virus（番茄黃化曲葉病毒） |
| 14 | Tomato - Mosaic Virus（番茄花葉病毒） |
| 15 | Tomato - Healthy（番茄健康） |

---

## 組員

| 系級 | 姓名 |
|------|------|
| 統計四 | 林承佑 |
| 統計四 | 曾博鴻 |
| 資訊三 | 黃柏淵 |
| 資訊三 | 陳立衡 |
| 資訊三 | 羅士恆 |
