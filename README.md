# 農作物病害自動化診斷系統 (Plant Disease Classification)
> **基於深度學習 ResNet-18 的即時影像辨識解決方案**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![ResNet18](https://img.shields.io/badge/Model-ResNet--18-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-97.2%25-brightgreen)

本專案利用殘差網路 (ResNet) 遷移學習技術，實現高精度的自動化農作物病害檢測。透過兩階段訓練策略（凍結特徵提取層與全層微調），有效提升了模型在複雜病斑上的辨識能力。

## 訓練成果 (Results)
本模型在驗證集上表現極其優異，尤其在第二階段 Fine-tuning 後取得了顯著突破：
- 初始驗證準確度：約 94% (僅訓練全連接層)
- 最終驗證準確度 (Fine-tuning)：**97.2%**
- 分類表現：模型對於 15 種常見作物病害（如蘋果黑星病、番茄晚疫病等）展現了極強的泛化能力，F1-score 穩定於高位。

## 技術規格 (Technical Specifications)
### 1. 訓練策略：兩階段遷移學習
為了達到最佳收斂效果，專案採用了以下步驟：
* 第一階段 (Transfer Learning)：凍結 ResNet-18 的卷積層，僅訓練輸出的 Linear Head。
* 第二階段 (Fine-tuning)：解凍模型參數，以極小的學習率進行全局微調，使模型能更精準地捕捉植物葉片的細微病變特徵，最終將 Accuracy 推升至 97% 以上。

### 2. 數據處理與模型架構
- 模型基礎：ResNet-18 (ImageNet Pre-trained)。
- 影像尺寸：輸入解析度調整為 `299x299`。
- 優化器配置：使用 Adam 優化器，搭配學習率衰減策略 (StepLR) 以確保細節微調時的穩定性。
- 數據集規模：PlantVillage Dataset，包含超過 20,000 張標註影像。

## 核心功能 (Core Features)
- 自動化分類：精準辨識包含黑斑病、銹病、白粉病等多種常見作物病害。
- 信心度回傳：輸出疾病類別並附帶信心程度。
- 高效推理：ResNet-18 的輕量化特性，使其在邊緣設備端也能維持快速反應。

---

## 專案結構 (Project Structure)

```
2026-Spring-DTAI/
├── app.py                              # Flask 後端推論伺服器
├── requirements.txt                    # Python 依賴套件清單
├── plant_disease_resnet18_finetuned.pth  # 訓練好的模型權重
├── templates/
│   └── index.html                      # 前端網頁介面
├── Final_Project.ipynb                 # 訓練流程 Notebook
└── .venv/                              # Python 虛擬環境 (本地，不入版控)
```

---

## 環境建置 (Setup)

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

> **GPU 加速（選用）**：若機器有 NVIDIA GPU，可至 [pytorch.org](https://pytorch.org/get-started/locally/) 安裝對應 CUDA 版本的 PyTorch，推論速度可大幅提升。

---

## 啟動 Web 診斷介面 (Run the Web UI)

確認已啟動虛擬環境，且專案根目錄下有 `plant_disease_resnet18_finetuned.pth` 權重檔。

```bash
python app.py
```

啟動成功後，終端機會顯示：

```
 * Running on http://127.0.0.1:5000
```

開啟瀏覽器前往 **http://127.0.0.1:5000** 即可使用。

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
