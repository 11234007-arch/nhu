# 如何在Google Colab GPU上訓練你的神經網路模型
摘要:主要目的是**建立並訓練一個用於 MNIST 手寫數字分類的簡單全連接神經網路模型**。程式先載入必要的套件，下載 MNIST 資料集並將其轉為張量，切分為訓練集與驗證集，再建立 DataLoader 方便批次處理，並將資料與模型搬到可用的運算裝置（CPU 或 GPU）。模型 Mnistmodel 包含一個隱藏層和輸出層，並定義了訓練步驟、驗證步驟與計算準確率的方法。接著，程式先在驗證集上評估模型初始表現，再以不同學習率分兩階段進行訓練，每個 epoch 後計算驗證集的損失與準確率，並將結果累計到 history 中，從而觀察模型性能隨訓練的變化。整體而言，程式完整示範了**資料準備、模型建立、運算裝置管理、訓練與驗證**的深度學習流程。
## 步驟1:點選執行階段中的變更執行階段類型並將執行時類型設定為 GPU

<img width="545" height="777" alt="螢幕擷取畫面 2025-10-23 010120" src="https://github.com/user-attachments/assets/1f486cc4-e2f3-408b-b1d7-43d66fa11ebe" />
<img width="795" height="719" alt="螢幕擷取畫面 2025-10-23 010130" src="https://github.com/user-attachments/assets/659367c9-a709-4777-8a4f-a0d14d359f36" />

## 步驟2:載入 PyTorch、Torchvision 與其他相關套件，準備處理 MNIST 資料集、建立神經網路以及用 Matplotlib 畫圖。

<img width="863" height="470" alt="螢幕擷取畫面 2025-10-23 003317" src="https://github.com/user-attachments/assets/75cb5456-d78f-483e-91bd-729144d93557" />

## 步驟3:下載 MNIST 手寫數字資料集，將圖像轉成張量，並把資料集切分成訓練集和驗證集。

<img width="996" height="290" alt="螢幕擷取畫面 2025-10-23 003359" src="https://github.com/user-attachments/assets/8494ae75-417f-4c08-9990-29a5141b9330" />

## 步驟4:定義一個函式，用來計算模型輸出與標籤之間的分類準確率。
### 在後面的class Mnistmodel(nn.Module)跟ddef validation_step(self,batch)有使用，但沒有先定義，所以先在第四步驟先定義

<img width="970" height="177" alt="螢幕擷取畫面 2025-10-23 004615" src="https://github.com/user-attachments/assets/7bd0f3dd-eeff-4a22-8db4-7a6cc79850db" />

## 步驟5:建立訓練集和驗證集的 DataLoader，用指定的批次大小、打亂資料並加速資料讀取。

<img width="1182" height="129" alt="螢幕擷取畫面 2025-10-23 003406" src="https://github.com/user-attachments/assets/677000a2-01e7-4ca1-ae56-df301865a1c6" />

## 步驟6:定義一個簡單的全連接神經網路（Mnistmodel），包含前向傳播、訓練步驟、驗證步驟，以及計算整個驗證集的損失和準確率。

<img width="969" height="744" alt="螢幕擷取畫面 2025-10-23 003420" src="https://github.com/user-attachments/assets/03da25e4-a8f4-41d6-bafc-386546490222" />
<img width="1359" height="694" alt="螢幕擷取畫面 2025-10-23 004641" src="https://github.com/user-attachments/assets/f02294fa-d4af-48e7-ba01-4369d7302c3e" />

## 步驟7:檢查目前系統是否有可用的 GPU（CUDA）來加速 PyTorch 運算。
### True 為「有可用 GPU」，False 為「無可用 GPU」

<img width="531" height="132" alt="螢幕擷取畫面 2025-10-23 003445" src="https://github.com/user-attachments/assets/241786c2-c051-4e26-9861-712bc88f9119" />

## 步驟8:選擇可用的運算裝置：如果有 GPU 就用 CUDA，否則使用 CPU，並把裝置存到 device 變數。

<img width="827" height="280" alt="螢幕擷取畫面 2025-10-23 003454" src="https://github.com/user-attachments/assets/9b5f5f82-d971-459a-a4fb-3c8f5ff7c362" />

## 步驟9:將張量或張量列表遞迴搬移到指定的運算裝置（CPU 或 GPU）上。

<img width="844" height="182" alt="螢幕擷取畫面 2025-10-23 003520" src="https://github.com/user-attachments/assets/395cd037-0dac-4006-8e20-bb0c94b58463" />

## 步驟10:建立一個封裝 DataLoader 的類別，讓每個批次資料自動搬到指定裝置（CPU/GPU），並用它來包裝訓練集與驗證集的 DataLoader。

<img width="890" height="471" alt="螢幕擷取畫面 2025-10-23 003723" src="https://github.com/user-attachments/assets/dabfd68e-7e62-4a34-bfbb-996f5255c8ab" />

## 步驟11:定義訓練函式 fit 和驗證函式 evaluate，用指定的 epoch 數與學習率訓練模型，並在每個 epoch 後計算驗證集的損失與準確率。
### 要注意縮排
<img width="1086" height="520" alt="螢幕擷取畫面 2025-10-23 003850" src="https://github.com/user-attachments/assets/699a4ac2-ac6d-4d43-8117-2e18886a89d6" />

## 步驟12:建立一個輸入 784 維、隱藏層 32 個神經元、輸出 10 類的 Mnistmodel，並將模型搬到指定的運算裝置（CPU 或 GPU）。

<img width="1022" height="257" alt="螢幕擷取畫面 2025-10-23 003910" src="https://github.com/user-attachments/assets/4b8cdaac-1a56-4d95-befc-b5f54885b639" />

## 步驟13:對模型在驗證集上做一次評估，計算初始的損失與準確率，並將結果存入 history 並印出。

<img width="1189" height="158" alt="螢幕擷取畫面 2025-10-23 004700" src="https://github.com/user-attachments/assets/769dd2d6-e379-427a-bbfa-791cbe5000d6" />

## 步驟14:用學習率 0.5 訓練模型 5 個 epoch，並將每個 epoch 的驗證結果累加到 history 中。

<img width="756" height="214" alt="螢幕擷取畫面 2025-10-23 004833" src="https://github.com/user-attachments/assets/6ef219eb-17da-46da-88c6-89cc0c94f263" />

## 步驟15:以較小的學習率 0.1 再訓練模型 5 個 epoch，並把每個 epoch 的驗證結果加入 history 紀錄。

<img width="695" height="233" alt="螢幕擷取畫面 2025-10-23 004927" src="https://github.com/user-attachments/assets/36cfe244-4a31-4d3c-9468-9f47341994e3" />

