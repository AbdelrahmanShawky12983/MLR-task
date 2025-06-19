# 🏥 BRCA Cancer Prediction System

A comprehensive machine learning web application for predicting breast cancer patient outcomes using protein expression data and clinical features.

## 🌟 Features

- **Interactive Data Upload**: Upload CSV datasets and automatically preprocess them
- **Multiple ML Models**: Compare Logistic Regression, Random Forest, and XGBoost classifiers
- **Real-time Predictions**: Make predictions on new patient data with confidence scores
- **Model Insights**: Visualize feature importance, ROC curves, and confusion matrices
- **User-friendly Interface**: Clean, modern web interface built with Streamlit

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](your-streamlit-app-url)

## 🛠️ Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brca-prediction.git
cd brca-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run streamlit_app.py
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t brca-prediction .
```

2. Run the container:
```bash
docker run -p 8501:8501 brca-prediction
```

## 📊 Dataset Requirements

Your CSV dataset should include:

### Required Columns:
- `Patient_Status`: Target variable ('Alive' or 'Dead')
- `Age`: Patient age
- `Protein1`, `Protein2`, `Protein3`, `Protein4`: Protein expression levels

### Optional Columns:
- `Tumour_Stage`: Cancer stage (I, II, III)
- `Histology`: Tumor histology type
- `ER status`: Estrogen receptor status
- `PR status`: Progesterone receptor status
- `HER2 status`: HER2 receptor status
- `Date_of_Surgery`: Surgery date
- `Date_of_Last_Visit`: Last visit date

## 🎯 Usage

### 1. Data Upload & Training
- Navigate to "Data Upload & Training"
- Upload your BRCA dataset (CSV format)
- Review dataset overview and statistics
- Click "Train Models" to train and compare multiple ML models
- View performance metrics and select the best model

### 2. Make Predictions
- Go to "Make Predictions"
- Input patient data through the interactive form
- Get instant predictions with confidence scores
- View risk probability visualization

### 3. Model Insights
- Access "Model Insights" for detailed analysis
- Compare model performance with radar charts
- Explore feature importance rankings
- Analyze ROC curves and confusion matrices

## 🔬 Machine Learning Pipeline

1. **Data Preprocessing**:
   - Handle missing values
   - Date feature engineering
   - Label encoding for categorical variables
   - Standard scaling for numerical features

2. **Model Training**:
   - Logistic Regression with balanced class weights
   - Random Forest Classifier
   - XGBoost Classifier
   - Automatic hyperparameter tuning

3. **Model Evaluation**:
   - Accuracy, Precision, Recall, F1-Score
   - ROC AUC Score
   - Cross-validation
   - Feature importance analysis

## 📈 Performance Metrics

The application evaluates models using multiple metrics:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the receiver operating characteristic curve

## 🐳 Docker Support

The application includes Docker support for easy deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]
```

## 🌐 Deployment Options

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click

### Heroku
1. Create a `Procfile`:
```
web: sh setup.sh && streamlit run streamlit_app.py
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

### AWS/GCP/Azure
Deploy using container services or virtual machines with the provided Docker configuration.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Research & Citations

If you use this application in your research, please cite:

```bibtex
@software{brca_prediction_2024,
  title={BRCA Cancer Prediction System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/brca-prediction}
}
```

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn Profile]

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Machine learning powered by [scikit-learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.readthedocs.io/)
- Visualizations created with [Plotly](https://plotly.com/) and [Seaborn](https://seaborn.pydata.org/)

---

⭐ If you find this project helpful, please give it a star on GitHub!
