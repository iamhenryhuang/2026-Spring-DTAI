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
