import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("🚀 Program 2: Gradient Descent Interaktif - UAS Optimization Methods")

# Load data dengan error handling
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('data.csv')
        return data
    except FileNotFoundError:
        st.error("❌ File 'data.csv' tidak ditemukan di repo GitHub!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error load data: {e}")
        st.stop()

data = load_data()
st.subheader("📊 Dataset")
st.dataframe(data.head())

# Ambil kolom X dan Y otomatis (kolom pertama = X, kedua = Y)
if len(data.columns) < 2:
    st.error("❌ Data.csv harus punya minimal 2 kolom!")
    st.stop()

X = data.iloc[:, 0].values  # Kolom pertama sebagai X
y = data.iloc[:, 1].values  # Kolom kedua sebagai Y
st.info(f"📈 Menggunakan kolom '{data.columns[0]}' sebagai X dan '{data.columns[1]}' sebagai Y")

n = len(X)

# Fungsi cost dan gradient descent
def compute_cost(m, b, X, y):
    y_pred = m * X + b
    return (1 / (2 * n)) * np.sum((y - y_pred) ** 2)

@st.cache_data
def gradient_descent(X, y, lr, n_iter, m_init=0.0, b_init=0.0):
    m = m_init
    b = b_init
    cost_history = []
    
    for i in range(n_iter):
        y_pred = m * X + b
        dm = (-2 / n) * np.sum(X * (y - y_pred))
        db = (-2 / n) * np.sum(y - y_pred)
        
        m -= lr * dm
        b -= lr * db
        cost_history.append(compute_cost(m, b, X, y))
    
    return m, b, cost_history

# Sidebar parameter
st.sidebar.header("⚙️ Parameter Gradient Descent")
lr = st.sidebar.number_input("Learning Rate", 0.0001, 0.1, 0.01, 0.0001, format="%.4f")
n_iter = st.sidebar.number_input("Jumlah Iterasi", 100, 5000, 1000, 100)

x_new = st.number_input("🔮 Prediksi untuk nilai X =", value=float(X.mean()))

# Tombol jalankan
if st.button("🚀 Jalankan Gradient Descent", type="primary"):
    with st.spinner("Training model..."):
        m_final, b_final, cost_history = gradient_descent(X, y, lr, n_iter)
    
    # Hasil
    st.subheader("✅ Hasil Training")
    col1, col2, col3 = st.columns(3)
    col1.metric("Slope (m)", f"{m_final:.4f}")
    col2.metric("Intercept (b)", f"{b_final:.4f}")
    col3.metric("Final Cost", f"{cost_history[-1]:.6f}")
    
    # Prediksi
    y_pred = m_final * x_new + b_final
    st.success(f"📊 Prediksi Y untuk X={x_new:.2f} = **{y_pred:.2f}**")
    
    # Grafik 1: Scatter plot + regresi
    st.subheader("📈 Data & Garis Regresi")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(X, y, alpha=0.7, s=60, label="Data Training")
    x_line = np.linspace(min(X), max(X), 100)
    y_line = m_final * x_line + b_final
    ax1.plot(x_line, y_line, "r-", linewidth=3, label="Garis Regresi (GD)")
    ax1.scatter(x_new, y_pred, s=200, c="green", marker="*", 
                label=f"Prediksi ({x_new:.1f}, {y_pred:.1f})", zorder=5)
    ax1.set_xlabel(data.columns[0])
    ax1.set_ylabel(data.columns[1])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    
    # Grafik 2: Konvergensi cost
    st.subheader("📉 Konvergensi Cost")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(cost_history, linewidth=2)
    ax2.set_xlabel("Iterasi")
    ax2.set_ylabel("Cost (MSE)")
    ax2.set_title("Perkembangan Loss Function")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

st.markdown("---")
st.caption("🎓 UAS Optimization Methods - Gradient Descent Linear Regression")
