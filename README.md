# 農作物病蟲害自動化診斷系統
> **基於深度學習的即時影像辨識解決方案**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-orange)
![DIP](https://img.shields.io/badge/Field-Computer%20Vision-green)

本專案旨在解決農民在辨識作物病害時對專家的依賴痛點，透過開發輕量化的影像辨識模型，實現手機端的即時檢測，降低農業損害。

## 專案概述 (Overview)
在傳統農業中，病害診斷往往需要具備多年經驗的專家到場，這導致了診斷延遲與防治成本增加。本專案利用 **MobileNetV2** 架構，在確保高準確度的前提下，優化模型大小以利於移動端部署。

## 核心功能 (Core Features)
- **自動化分類**：精準辨識包含黑斑病、銹病、白粉病等多種常見作物病害。
- **信心度回傳**：輸出疾病類別並附帶 **信心程度 (Confidence Score)**，提供可靠的決策依據。
- **邊緣運算優化**：專為手機場景設計，確保在自然光環境下也能快速反應。

## 技術規格 (Technical Specifications)
### 1. 數據集與模型
- **資料來源**：[PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) (包含 14 種作物、38 個類別)。
- **模型架構**：MobileNetV2 (以 ImageNet 權重進行遷移學習)。
- **數據分割**：70% 訓練集 / 30% 測試集。

### 2. 影像預處理
- **對比度增強**：應用 CLAHE (Contrast Limited Adaptive Histogram Equalization) 強化病斑特徵。
- **資料增強 (Data Augmentation)**：
  - 隨機旋轉、縮放與亮度調整。
  - 模擬田間自然光源與多角度拍攝。

## 預期效益
- **去中心化**：農民可即時自主診斷，無需等待專家。
- **精準防治**：減少用藥浪費，抓住最佳防治時機，降低經濟損害。
