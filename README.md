# 💻 Laptop Price Predictor

An end-to-end machine learning project that predicts laptop prices using a **Random Forest Regression** model (~87 % R²) served via an interactive **Streamlit** web app.

![Demo screenshot](Images/image1.png)

---

## 📁 Project Structure

```
MLP_Laptop-Price-Prediction/
├── app.py                  # Streamlit web application
├── train_model.py          # Script to train & save model artefacts
├── requirements.txt        # Python dependencies
├── Procfile                # Heroku / deployment config
├── setup.sh                # Streamlit server config for deployment
│
├── data/
│   ├── laptop_data.csv     # Raw laptop dataset
│   └── cleaned_Data.csv    # Pre-processed dataset (used for training)
│
├── models/
│   ├── pipe.pkl            # Trained scikit-learn Pipeline
│   └── df.pkl              # Reference DataFrame for UI drop-downs
│
├── notebooks/
│   └── LP_Model.ipynb      # EDA, feature engineering & model training notebook
│
└── Images/
    └── image1.png
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/harmandeep2993/MLP_Laptop-Price-Pridiction.git
cd MLP_Laptop-Price-Pridiction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain the model
```bash
python train_model.py
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🔮 How It Works

| Step | Description |
|------|-------------|
| **Data** | 1,300 + laptop listings with specs (CPU, RAM, storage, display, GPU, OS, brand …) |
| **Features** | 13 engineered features: Brand, OS, Processor, Type, GPU brand, RAM, Touchscreen, Weight, HDD, SSD, IPS, Screen size, PPI |
| **Model** | `RandomForestRegressor` inside a `sklearn.Pipeline` with `OrdinalEncoder` for categorical columns |
| **Target** | `log(Price)` — predictions are exponentiated back to INR |
| **Accuracy** | ≈ 87% R² on the hold-out test set |

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Streamlit** – interactive web UI
- **scikit-learn** – preprocessing & model
- **pandas / NumPy** – data manipulation
- **Jupyter Notebook** – EDA & experimentation

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

