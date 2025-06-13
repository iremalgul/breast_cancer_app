# Meme Kanseri Tahmin Uygulaması

Bu proje, Wisconsin Meme Kanseri (Diagnostic) veri setini kullanarak meme kanseri teşhisi için makine öğrenmesi modelleri geliştiren ve bu modelleri kullanarak tahmin yapan bir web uygulamasıdır.

## Veri Seti

Uygulama, [Kaggle Wisconsin Meme Kanseri (Diagnostic) Veri Seti](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)'ni kullanmaktadır.

## Canlı Uygulama

Uygulamayı [Streamlit Cloud](https://breastcancerapp-rx3e5grzzrwkuzpx24wr9m.streamlit.app/) üzerinden test edebilirsiniz.

## Özellikler

- Veri seti hakkında detaylı bilgi
- Farklı makine öğrenmesi modellerinin performans karşılaştırması
- Kullanıcı dostu arayüz ile tahmin yapma imkanı
- Görsel grafikler ve sonuç analizleri

## Kullanılan Teknolojiler

- Python
- Streamlit
- Scikit-learn
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/iremalgul/breast_cancer_app.git
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Uygulamayı çalıştırın:
```bash
streamlit run app.py
```

## Model Performansları

Uygulama içerisinde kullanılan modellerin performans metrikleri:

- Logistic Regression: %95
- Random Forest: %97
- SVM: %96
- KNN: %95
- Neural Network: %96

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik: Açıklama'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Bir Pull Request oluşturun

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Daha fazla bilgi için `LICENSE` dosyasına bakın.
