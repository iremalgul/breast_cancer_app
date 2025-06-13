import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Sayfa yapılandırması
st.set_page_config(
    page_title="Meme Kanseri Tahmin Uygulaması",
    page_icon="🏥",
    layout="wide"
)

# Başlık
st.title("Meme Kanseri Tahmin Uygulaması")
st.markdown("---")

# Veri seti hakkında bilgi
st.header("Veri Seti Hakkında")
st.markdown("""
Bu uygulama, Wisconsin Meme Kanseri (Diagnostic) veri setini kullanmaktadır. Veri seti, meme kanseri teşhislerine ilişkin ölçümleri içermektedir.

### Veri Seti Yapısı
- **Hedef Değişken**: diagnosis (M: Malignant/Kötü huylu, B: Benign/İyi huylu)
- **Veri Seti Boyutu**: 569 örnek, 32 özellik

### Özellikler ve Ölçümler
Her özellik için 3 farklı ölçüm bulunmaktadır:

1. **Mean (Ortalama) Değerler**
   - Tümörün temel özelliklerinin ortalamasını gösterir
   - Örnek: radius_mean, texture_mean, perimeter_mean

2. **SE (Standard Error) Değerler**
   - Ölçümlerdeki standart hata değerlerini gösterir
   - Örnek: radius_se, texture_se, perimeter_se

3. **Worst (En Kötü) Değerler**
   - Tümörün en kötü/şüpheli bölgelerindeki özellikleri gösterir
   - Örnek: radius_worst, texture_worst, perimeter_worst

### Özellikler
- Yarıçap (radius)
- Doku (texture)
- Çevre (perimeter)
- Alan (area)
- Pürüzsüzlük (smoothness)
- Kompaktlık (compactness)
- İçbükeylik (concavity)
- İçbükey Noktalar (concave points)
- Simetri (symmetry)
- Fraktal Boyut (fractal dimension)
""")

# Model yükleme fonksiyonu
@st.cache_resource
def load_models():
    models = {
        'Logistic Regression': joblib.load('models/logreg.pkl'),
        'Random Forest': joblib.load('models/rf_model.pkl'),
        'SVM': joblib.load('models/svm_model.pkl'),
        'KNN': joblib.load('models/knn_model.pkl'),
        'Neural Network': load_model('models/nn_model.h5')
    }
    scaler = joblib.load('models/scaler.pkl')
    return models, scaler

# Modelleri yükle
try:
    models, scaler = load_models()
    st.success("Modeller başarıyla yüklendi!")
except Exception as e:
    st.error(f"Model yükleme hatası: {str(e)}")
    st.stop()

# Model performansları
st.header("Model Performansları")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Doğruluk Oranları")
    accuracy_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'SVM', 'KNN', 'Neural Network'],
        'Accuracy': [0.95, 0.97, 0.96, 0.95, 0.96]
    }
    accuracy_df = pd.DataFrame(accuracy_data)
    st.dataframe(accuracy_df, hide_index=True)

with col2:
    st.subheader("Performans Grafiği")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=accuracy_df, x='Model', y='Accuracy')
    plt.xticks(rotation=45)
    plt.title('Model Doğruluk Oranları')
    st.pyplot(fig)

# Tahmin bölümü
st.header("Tahmin Yap")
st.markdown("Aşağıdaki özellikleri girerek tahmin yapabilirsiniz:")

# Özellik girişi için sütunlar
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Mean Değerler")
    radius_mean = st.number_input("Yarıçap (mean)", min_value=0.0, max_value=30.0, value=14.0)
    texture_mean = st.number_input("Doku (mean)", min_value=0.0, max_value=40.0, value=19.0)
    perimeter_mean = st.number_input("Çevre (mean)", min_value=0.0, max_value=200.0, value=91.0)
    area_mean = st.number_input("Alan (mean)", min_value=0.0, max_value=2500.0, value=654.0)
    smoothness_mean = st.number_input("Pürüzsüzlük (mean)", min_value=0.0, max_value=1.0, value=0.1)
    compactness_mean = st.number_input("Kompaktlık (mean)", min_value=0.0, max_value=1.0, value=0.1)
    concavity_mean = st.number_input("İçbükeylik (mean)", min_value=0.0, max_value=1.0, value=0.1)
    concave_points_mean = st.number_input("İçbükey Noktalar (mean)", min_value=0.0, max_value=1.0, value=0.1)
    symmetry_mean = st.number_input("Simetri (mean)", min_value=0.0, max_value=1.0, value=0.2)
    fractal_dimension_mean = st.number_input("Fraktal Boyut (mean)", min_value=0.0, max_value=1.0, value=0.1)

with col2:
    st.subheader("SE Değerler")
    radius_se = st.number_input("Yarıçap (se)", min_value=0.0, max_value=5.0, value=0.4)
    texture_se = st.number_input("Doku (se)", min_value=0.0, max_value=5.0, value=0.4)
    perimeter_se = st.number_input("Çevre (se)", min_value=0.0, max_value=20.0, value=2.9)
    area_se = st.number_input("Alan (se)", min_value=0.0, max_value=500.0, value=40.0)
    smoothness_se = st.number_input("Pürüzsüzlük (se)", min_value=0.0, max_value=0.1, value=0.01)
    compactness_se = st.number_input("Kompaktlık (se)", min_value=0.0, max_value=0.1, value=0.02)
    concavity_se = st.number_input("İçbükeylik (se)", min_value=0.0, max_value=0.1, value=0.03)
    concave_points_se = st.number_input("İçbükey Noktalar (se)", min_value=0.0, max_value=0.1, value=0.01)
    symmetry_se = st.number_input("Simetri (se)", min_value=0.0, max_value=0.1, value=0.02)
    fractal_dimension_se = st.number_input("Fraktal Boyut (se)", min_value=0.0, max_value=0.1, value=0.01)

with col3:
    st.subheader("Worst Değerler")
    radius_worst = st.number_input("Yarıçap (worst)", min_value=0.0, max_value=40.0, value=16.0)
    texture_worst = st.number_input("Doku (worst)", min_value=0.0, max_value=50.0, value=25.0)
    perimeter_worst = st.number_input("Çevre (worst)", min_value=0.0, max_value=300.0, value=107.0)
    area_worst = st.number_input("Alan (worst)", min_value=0.0, max_value=5000.0, value=880.0)
    smoothness_worst = st.number_input("Pürüzsüzlük (worst)", min_value=0.0, max_value=1.0, value=0.2)
    compactness_worst = st.number_input("Kompaktlık (worst)", min_value=0.0, max_value=1.0, value=0.3)
    concavity_worst = st.number_input("İçbükeylik (worst)", min_value=0.0, max_value=1.0, value=0.4)
    concave_points_worst = st.number_input("İçbükey Noktalar (worst)", min_value=0.0, max_value=1.0, value=0.2)
    symmetry_worst = st.number_input("Simetri (worst)", min_value=0.0, max_value=1.0, value=0.3)
    fractal_dimension_worst = st.number_input("Fraktal Boyut (worst)", min_value=0.0, max_value=1.0, value=0.1)

# Model seçimi
selected_model = st.selectbox(
    "Tahmin için model seçin:",
    list(models.keys())
)

# Tahmin yapma
if st.button("Tahmin Yap"):
    # Özellikleri birleştir
    features = np.array([[
        radius_mean, texture_mean, perimeter_mean, area_mean,
        smoothness_mean, compactness_mean, concavity_mean,
        concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se,
        smoothness_se, compactness_se, concavity_se,
        concave_points_se, symmetry_se, fractal_dimension_se,
        radius_worst, texture_worst, perimeter_worst, area_worst,
        smoothness_worst, compactness_worst, concavity_worst,
        concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]])
    
    # Özellikleri ölçeklendir
    features_scaled = scaler.transform(features)
    
    # Seçilen modele göre tahmin yap
    model = models[selected_model]
    
    if selected_model == 'Neural Network':
        prediction = model.predict(features_scaled)
        probability = prediction[0][0]
    elif selected_model == 'SVM':
        # SVM için sadece sınıf tahmini yap
        prediction = model.predict(features_scaled)
        probability = 1.0 if prediction[0] == 1 else 0.0
    else:
        probability = model.predict_proba(features_scaled)[0][1]
    
    # Sonucu göster
    st.markdown("---")
    st.subheader("Tahmin Sonucu")
    
    if probability > 0.5:
        st.error(f"Kötü Huylu (Malignant) - Güven: {probability:.2%}")
    else:
        st.success(f"İyi Huylu (Benign) - Güven: {1-probability:.2%}")
    
    # Güven grafiği
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh(['Tahmin Güveni'], [probability], color='red' if probability > 0.5 else 'green')
    ax.set_xlim(0, 1)
    st.pyplot(fig)
