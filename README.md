# 🏦 Bank Marketing Effectiveness Prediction

This project predicts whether a customer will subscribe to a term deposit based on historical bank marketing campaign data. It includes complete ML pipeline development: data preprocessing, EDA, model training, hyperparameter tuning, and deployment via Streamlit.

---

## 📁 Project Folder Structure
- app.py – Streamlit app for real-time prediction

- Bank_Marketing_Effectiveness_Prediction.ipynb – Main notebook: EDA, training, and evaluation

- Bank_Marketing_Effectiveness_Prediction_Streamlit_Deployment.ipynb – Notebook for Streamlit deployment

- random_forest_pipeline.pkl – Trained Random Forest model (stored using Git LFS)

- synthetic_bank_marketing_100k.csv – Cleaned dataset used for training

- requirements.txt – Python dependencies needed to run the app

- .gitattributes – Git LFS tracking configuration file

- README.md – You’re reading it! Project overview and usage instructions



---

## 🔍 Exploratory Data Analysis (EDA)

- Class Imbalance: ~88% of clients did not subscribe
- Subscription likelihood increases with:
  - Longer call duration
  - Higher balance
  - Past campaign success
- Students and retirees showed higher conversion rates

---

## ⚙️ Feature Engineering

- Categorical features: one-hot encoded
- Numerical features: standardized using `StandardScaler`
- Features like `job`, `education`, `month`, and `contact` were encoded
- Outliers removed: rows with negative balance + subscription
- Final dataset included 36+ engineered columns

---

## 🤖 Model Comparison

| Model                | Accuracy | AUC  |
|---------------------|----------|------|
| Logistic Regression | 92.9%    | 0.91 |
| K-Nearest Neighbors | 92.9%    | 0.93 |
| Naive Bayes         | 83.0%    | 0.79 |
| SVM (RBF Kernel)    | 92.8%    | 0.92 |
| **Random Forest**   | **93.0%**| **0.94** ✅ |

---

## ✅ Final Model: Random Forest

- **Best Parameters:** `n_estimators=300`, `max_depth=13`
- **Cross-validated Accuracy:** ~93%
- **Top Features:** `duration`, `poutcome`, `balance`, `contact`, `education`
- Trained with `GridSearchCV` and exported using `joblib`

---

## 🚀 Streamlit Deployment

The final model is deployed using **Streamlit** for interactive use by marketing teams.

## App link

https://bankmarketingeffectivenessprediction-aq78zjpamhfgmb9rnchaep.streamlit.app/

### 📱 App Features:
- Input client attributes (age, job, education, etc.)
- Instant prediction of subscription outcome
- Simple, lightweight UI
- Can be deployed to Streamlit Cloud or run locally

---

## 💻 Run the App Locally

git clone https://github.com/mayu99/Bank_Marketing_Effectiveness_Prediction.git

cd Bank_Marketing_Effectiveness_Prediction

pip install -r requirements.txt

streamlit run app.py

---

## 📦 File Descriptions

| File Name                                                        | Description                                  |
|------------------------------------------------------------------|----------------------------------------------|
| `app.py`                                                         | Streamlit app UI for predictions             |
| `random_forest_pipeline.pkl`                                     | Final trained model (requires Git LFS)       |
| `synthetic_bank_marketing_100k.csv`                              | Dataset used for training                    |
| `requirements.txt`                                               | All Python libraries needed to run the app   |
| `Bank_Marketing_Effectiveness_Prediction.ipynb`                  | Main notebook with EDA, preprocessing, training |
| `Bank_Marketing_Effectiveness_Prediction_Streamlit_Deployment.ipynb` | Notebook for deployment steps           |
| `.gitattributes`                                                 | Git LFS tracking configuration               |

---

## 👨‍💻 Authors

This project was developed as part of **ML 245** coursework.

**Group 3 – ML 245:**
- Mayuresh Pandey  
- Nikhil Swami   
- Rupesh Motipawale  
- Padmanabh Rathi  
- Suyog Jadhav  

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to use, share, and adapt with credit.

