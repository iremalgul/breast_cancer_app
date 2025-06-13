# Meme Kanseri Tahmin Uygulaması

Bu proje, Wisconsin Meme Kanseri (Diagnostic) veri seti kullanılarak geliştirilmiş bir makine öğrenmesi tabanlı web uygulamasıdır. Uygulama, tümör özelliklerine dayanarak iyi huylu (benign) veya kötü huylu (malignant) tümör tahmini yapmaktadır.

## 🚀 Özellikler

- 5 farklı makine öğrenmesi modeli (Logistic Regression, Random Forest, SVM, KNN, Neural Network)
- Kullanıcı dostu web arayüzü
- Gerçek zamanlı tahmin
- Model performans karşılaştırmaları
- Görsel geri bildirimler

## 📊 Veri Seti

Wisconsin Meme Kanseri (Diagnostic) veri seti kullanılmaktadır:
- Veri seti: [Kaggle - Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- 569 örnek
- 32 özellik
- Her özellik için 3 farklı ölçüm:
  - Mean (Ortalama) değerler
  - SE (Standard Error) değerler
  - Worst (En Kötü) değerler

## 🛠️ Teknolojiler

- Python 3.x
- Streamlit
- Scikit-learn
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Seaborn

## ⚙️ Kurulum

1. Projeyi klonlayın:
```bash
git clone [proje-url]
cd breast_cancer_app
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Uygulamayı başlatın:
```bash
streamlit run app.py
```

## 📝 Kullanım

1. Uygulama başlatıldığında, tarayıcınızda otomatik olarak açılacaktır
2. Veri seti hakkında bilgileri ve model performanslarını inceleyebilirsiniz
3. Tahmin yapmak için:
   - Tümör özelliklerini girin
   - İstediğiniz modeli seçin
   - "Tahmin Yap" butonuna tıklayın
4. Sonuç ve güven oranı ekranda gösterilecektir

## 📈 Model Performansları

- Logistic Regression: %95
- Random Forest: %97
- SVM: %96
- KNN: %95
- Neural Network: %96

## 📁 Proje Yapısı

```
breast_cancer_app/
│
├── app.py              # Streamlit uygulaması
├── requirements.txt    # Gerekli kütüphaneler
├── README.md          # Proje dokümantasyonu
│
└── models/            # Eğitilmiş modeller
    ├── logreg.pkl
    ├── rf_model.pkl
    ├── svm_model.pkl
    ├── knn_model.pkl
    ├── nn_model.h5
    └── scaler.pkl
```

## ⚠️ Önemli Notlar

- Bu uygulama sadece eğitim amaçlıdır
- Gerçek tıbbi teşhis için kullanılmamalıdır
- Tüm tahminler bir doktor tarafından doğrulanmalıdır
