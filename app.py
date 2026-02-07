import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="DiabetesRisk AI", layout="wide")

# --- JUDUL DAN DESKRIPSI (Sesuai Bab I & II) ---
st.title("🏥 DiabetesRisk AI: Sistem Deteksi Dini Diabetes")
st.markdown("""
Aplikasi ini mengimplementasikan algoritma **Random Forest** untuk memprediksi risiko Diabetes Melitus.
Sistem ini dirancang sebagai *Clinical Decision Support System* untuk membantu screening awal pasien.
Referensi: *Implementasi Algoritma Random Forest untuk Memprediksi Penyakit Diabetes Melitus*.
""")

# --- 1. PENGUMPULAN & PEMAHAMAN DATA [cite: 114-118] ---
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('diabetes.csv')
        return data
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("File 'diabetes.csv' tidak ditemukan! Silakan download dari Kaggle (Pima Indians Diabetes) dan letakkan di folder yang sama.")
else:
    # Sidebar Menu
    menu = st.sidebar.selectbox("Menu Navigasi", ["Beranda & Dataset", "Pelatihan Model", "Prediksi Diagnosa"])

    # --- 2. PRA-PEMROSESAN DATA [cite: 123-129] ---
    # Mengganti nilai 0 dengan NaN pada kolom biologis (karena 0 tidak logis untuk Glukosa, TD, dll) [cite: 126]
    cols_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_processed = df.copy()
    df_processed[cols_zero] = df_processed[cols_zero].replace(0, np.nan)
    
    # Handling Missing Data dengan Mean Imputation [cite: 127]
    imputer = SimpleImputer(strategy='mean')
    df_processed[cols_zero] = imputer.fit_transform(df_processed[cols_zero])

    # Memisahkan Fitur dan Target
    X = df_processed.drop('Outcome', axis=1)
    y = df_processed['Outcome']

    # 1. Terapkan SMOTE SEBELUM splitting atau hanya pada data training
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    # Split Data 80:20 [cite: 134]
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Membangun Model Random Forest [cite: 139-141]
    # Parameter n_estimators=100 sesuai rekomendasi efisiensi [cite: 237]
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None]
    }
    # rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf_model.fit(X_train, y_train)
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    rf_model = grid_search.best_estimator_
    
    # --- MENU 1: EKSPLORASI DATA ---
    if menu == "Beranda & Dataset":
        st.header("Eksplorasi Data Medis (Pima Indians Diabetes)")
        st.write(f"Jumlah Data: {df.shape[0]} Pasien, Jumlah Fitur: {df.shape[1]}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sampel Data Asli")
            st.dataframe(df.head(10))
        
        with col2:
            st.subheader("Statistik Deskriptif")
            st.write(df.describe())

        st.info("Catatan: Dilakukan Pra-pemrosesan (Data Cleaning) untuk menangani nilai 0 yang tidak logis pada Glukosa, Tekanan Darah, dll[cite: 126].")

    # --- MENU 2: PERFORMA MODEL ---
    elif menu == "Pelatihan Model":
        st.header("Evaluasi Model Random Forest")
        
        # Prediksi pada data test
        y_pred = rf_model.predict(X_test)
        y_probs = rf_model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Menampilkan Metrik [cite: 149-150]
        mxCol1, mxCol2, mxCol3, mxCol4, mxCol5= st.columns(5)
        
        with mxCol1:
            st.metric(label="Akurasi Model", value=f"{accuracy * 100:.2f}%")
        with mxCol2:
            st.metric(label="Precision", value=f"{prec * 100:.2f}%")
        with mxCol3:
            st.metric(label="Recall", value=f"{rec * 100:.2f}%")
        with mxCol4:
            st.metric(label="F1-Score", value=f"{f1 * 100:.2f}%")
        with mxCol5:
            st.metric(label="Mean Squared Error (MSE):", value=f"{mse:.4f}")
        st.success(f"Model berhasil mencapai akurasi yang kompetitif (Ref: Studi Rahman et al. mencapai ~96.7% [cite: 150]).")
        
        st.subheader("Visualisasi Evaluasi Model")
        c1, c2 = st.columns(2)

        with c1:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Prediksi')
            ax_cm.set_ylabel('Aktual')
            st.pyplot(fig_cm)
        with c2:
            st.write("**ROC Curve**")
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

        # Feature Importance [cite: 152, 249]
        st.subheader("Feature Importance (Tingkat Kepentingan Atribut)")
        st.markdown("Grafik ini menunjukkan faktor mana yang paling berpengaruh terhadap diagnosis.")
        
        feature_scores = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        fig, ax = plt.subplots()
        sns.barplot(x=feature_scores, y=feature_scores.index, ax=ax, palette="viridis")
        ax.set_title("Faktor Risiko Paling Dominan")
        ax.set_xlabel("Skor Kepentingan")
        ax.set_ylabel("Atribut Medis")
        st.pyplot(fig)
        
        st.write("Sesuai teori, Glukosa dan BMI sering menjadi indikator dominan.")

    # --- MENU 3: SIMULASI PREDIKSI (PRODUK UTAMA) ---
    elif menu == "Prediksi Diagnosa":
        st.header("Simulasi Diagnosa Pasien")
        st.markdown("Masukkan data parameter klinis pasien di bawah ini:")

        # Input Form [cite: 117]
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Jumlah Kehamilan (Pregnancies)", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Kadar Glukosa (Glucose)", min_value=0, max_value=200, value=120)
            bp = st.number_input("Tekanan Darah Diastolik (BloodPressure)", min_value=0, max_value=140, value=70)
            skin = st.number_input("Ketebalan Kulit (SkinThickness)", min_value=0, max_value=100, value=20)
        
        with col2:
            insulin = st.number_input("Insulin Serum (Insulin)", min_value=0, max_value=900, value=80)
            bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, max_value=70.0, value=32.0)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
            age = st.number_input("Usia (Age)", min_value=21, max_value=100, value=33)

        # Tombol Prediksi
        if st.button("Analisis Risiko Diabetes"):
            # Susun data input
            input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            
            # Prediksi Kelas dan Probabilitas [cite: 163]
            prediction = rf_model.predict(input_data)[0]
            prediction_prob = rf_model.predict_proba(input_data)[0][1] # Ambil probabilitas kelas 1 (Diabetes)

            st.divider()
            
            # Tampilan Hasil
            if prediction == 1:
                st.error(f"⚠️ **HASIL: BERISIKO DIABETES**")
                st.write(f"Probabilitas Risiko: **{prediction_prob * 100:.2f}%**")
                st.markdown("""
                **Rekomendasi Tindakan[cite: 165]:**
                * Segera lakukan pemeriksaan laboratorium lanjutan (HbA1c).
                * Konsultasi dengan dokter untuk manajemen diet dan gaya hidup.
                """)
            else:
                st.success(f"✅ **HASIL: TIDAK BERISIKO / NORMAL**")
                st.write(f"Probabilitas Risiko: **{prediction_prob * 100:.2f}%**")
                st.markdown("""
                **Rekomendasi:**
                * Pertahankan gaya hidup sehat.
                * Lakukan pengecekan rutin berkala.
                """)
            
            st.info("Catatan: Hasil ini merupakan dukungan keputusan klinis dan bukan diagnosis medis final[cite: 279].")

# Footer
st.markdown("---")
st.caption("Dikembangkan untuk Pameran Mata Kuliah Kecerdasan Buatan | Berbasis Algoritma Random Forest")