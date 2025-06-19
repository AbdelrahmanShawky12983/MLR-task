import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import warnings
import pickle
import io

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="BRCA Cancer Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

class BRCAPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.target_column = 'Patient_Status'
        
    def preprocess_data(self, df):
        """Preprocess the data similar to the original notebook"""
        df_processed = df.copy()
        
        # Handle target column if present
        if self.target_column in df_processed.columns:
            df_processed[self.target_column] = df_processed[self.target_column].astype(str).str.strip().str.capitalize()
            desired_mapping = {'Alive': 0, 'Dead': 1}
            df_processed[self.target_column] = df_processed[self.target_column].map(desired_mapping)
            df_processed[self.target_column] = df_processed[self.target_column].fillna(0).astype(int)
        
        # Handle date columns if present
        date_cols = ['Date_of_Surgery', 'Date_of_Last_Visit']
        present_date_cols = [col for col in date_cols if col in df_processed.columns]
        
        if len(present_date_cols) == 2:
            try:
                df_processed[present_date_cols[0]] = pd.to_datetime(df_processed[present_date_cols[0]], errors='coerce')
                df_processed[present_date_cols[1]] = pd.to_datetime(df_processed[present_date_cols[1]], errors='coerce')
                
                # Fill missing dates with median
                df_processed[present_date_cols[0]].fillna(df_processed[present_date_cols[0]].median(), inplace=True)
                df_processed[present_date_cols[1]].fillna(df_processed[present_date_cols[1]].median(), inplace=True)
                
                # Create duration feature
                df_processed['Duration_of_Follow_up_Days'] = (df_processed[present_date_cols[1]] - df_processed[present_date_cols[0]]).dt.days
                df_processed.drop(columns=present_date_cols, inplace=True)
            except:
                pass
        
        # Handle missing values
        for col in df_processed.columns:
            if df_processed[col].isnull().any():
                if df_processed[col].dtype == 'object':
                    df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown', inplace=True)
                else:
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
        
        # Drop ID columns
        id_cols = ['Patient_ID', 'ID']
        for col in id_cols:
            if col in df_processed.columns:
                df_processed.drop(col, axis=1, inplace=True)
        
        # Encode categorical variables
        categorical_cols = df_processed.select_dtypes(include='object').columns.tolist()
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                # Handle unseen categories
                df_processed[col] = df_processed[col].apply(
                    lambda x: self.label_encoders[col].transform([x])[0] 
                    if x in self.label_encoders[col].classes_ else 0
                )
        
        return df_processed
    
    def train_model(self, df):
        """Train the model"""
        df_processed = self.preprocess_data(df)
        
        if self.target_column not in df_processed.columns:
            st.error(f"Target column '{self.target_column}' not found!")
            return False
        
        X = df_processed.drop(self.target_column, axis=1)
        y = df_processed[self.target_column]
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
        
        best_score = 0
        best_model_name = None
        results = {}
        
        for name, model in models.items():
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC AUC': roc_auc,
                'Model': model,
                'Predictions': y_pred,
                'Probabilities': y_pred_proba
            }
            
            if roc_auc > best_score:
                best_score = roc_auc
                best_model_name = name
        
        self.model = results[best_model_name]['Model']
        
        return results, X_test, y_test, best_model_name
    
    def predict(self, input_data):
        """Make predictions on new data"""
        if self.model is None or self.scaler is None:
            return None
        
        # Ensure input has all required features
        for feature in self.feature_names:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        input_data = input_data[self.feature_names]
        
        if isinstance(self.model, LogisticRegression):
            input_scaled = self.scaler.transform(input_data)
            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0]
        else:
            prediction = self.model.predict(input_data)[0]
            probability = self.model.predict_proba(input_data)[0]
        
        return prediction, probability

def main():
    st.markdown('<h1 class="main-header">🏥 BRCA Cancer Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Data Upload & Training", "Make Predictions", "Model Insights"])
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = BRCAPredictor()
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    if page == "Data Upload & Training":
        st.header("📊 Data Upload & Model Training")
        
        uploaded_file = st.file_uploader("Upload your BRCA dataset (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Dataset uploaded successfully! Shape: {df.shape}")
                
                # Display data info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Dataset Overview")
                    st.write(df.head())
                
                with col2:
                    st.subheader("Data Info")
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    info_str = buffer.getvalue()
                    st.text(info_str)
                
                # Check for target column
                if st.session_state.predictor.target_column in df.columns:
                    st.success(f"✅ Target column '{st.session_state.predictor.target_column}' found!")
                    
                    if st.button("🚀 Train Models", type="primary"):
                        with st.spinner("Training models... This may take a few minutes."):
                            results, X_test, y_test, best_model_name = st.session_state.predictor.train_model(df)
                            st.session_state.results = results
                            st.session_state.X_test = X_test
                            st.session_state.y_test = y_test
                            st.session_state.best_model_name = best_model_name
                            st.session_state.model_trained = True
                        
                        st.success(f"🎉 Models trained successfully! Best model: {best_model_name}")
                        
                        # Display results
                        st.subheader("Model Performance Comparison")
                        
                        metrics_df = pd.DataFrame({
                            model: {
                                'Accuracy': results[model]['Accuracy'],
                                'Precision': results[model]['Precision'],
                                'Recall': results[model]['Recall'],
                                'F1 Score': results[model]['F1 Score'],
                                'ROC AUC': results[model]['ROC AUC']
                            }
                            for model in results.keys()
                        }).T
                        
                        st.dataframe(metrics_df.style.highlight_max(axis=0))
                        
                        # Visualize results
                        fig = px.bar(
                            x=metrics_df.index,
                            y=metrics_df['ROC AUC'],
                            title="Model Performance (ROC AUC)",
                            color=metrics_df['ROC AUC'],
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                else:
                    st.error(f"❌ Target column '{st.session_state.predictor.target_column}' not found in the dataset!")
                    st.info("Please ensure your dataset contains a 'Patient_Status' column with 'Alive' or 'Dead' values.")
                    
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
    
    elif page == "Make Predictions":
        st.header("🔮 Make Predictions")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model first using the 'Data Upload & Training' page.")
            return
        
        st.subheader("Input Patient Data")
        
        # Create input form based on common BRCA features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=20, max_value=100, value=50)
            protein1 = st.number_input("Protein1", value=0.0, format="%.4f")
            protein2 = st.number_input("Protein2", value=0.0, format="%.4f")
            
        with col2:
            protein3 = st.number_input("Protein3", value=0.0, format="%.4f")
            protein4 = st.number_input("Protein4", value=0.0, format="%.4f")
            tumour_stage = st.selectbox("Tumour Stage", ["I", "II", "III"])
            
        with col3:
            histology = st.selectbox("Histology", ["Infiltrating Ductal Carcinoma", "Infiltrating Lobular Carcinoma", "Mucinous Carcinoma"])
            er_status = st.selectbox("ER Status", ["Positive", "Negative"])
            pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
            her2_status = st.selectbox("HER2 Status", ["Positive", "Negative"])
        
        if st.button("🎯 Make Prediction", type="primary"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'Age': [age],
                'Protein1': [protein1],
                'Protein2': [protein2],
                'Protein3': [protein3],
                'Protein4': [protein4],
                'Tumour_Stage': [tumour_stage],
                'Histology': [histology],
                'ER status': [er_status],
                'PR status': [pr_status],
                'HER2 status': [her2_status]
            })
            
            try:
                # Make prediction
                prediction, probability = st.session_state.predictor.predict(input_data)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 0:
                        st.success("🟢 **Prediction: ALIVE**")
                        st.markdown(f"**Confidence: {probability[0]:.2%}**")
                    else:
                        st.error("🔴 **Prediction: DEAD**")
                        st.markdown(f"**Confidence: {probability[1]:.2%}**")
                
                with col2:
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probability[1] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Risk Probability (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgreen"},
                                {'range': [25, 50], 'color': "yellow"},
                                {'range': [50, 75], 'color': "orange"},
                                {'range': [75, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    elif page == "Model Insights":
        st.header("📈 Model Insights")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model first using the 'Data Upload & Training' page.")
            return
        
        results = st.session_state.results
        best_model_name = st.session_state.best_model_name
        
        # Model comparison
        st.subheader("🏆 Model Performance Comparison")
        
        metrics_df = pd.DataFrame({
            model: {
                'Accuracy': results[model]['Accuracy'],
                'Precision': results[model]['Precision'],
                'Recall': results[model]['Recall'],
                'F1 Score': results[model]['F1 Score'],
                'ROC AUC': results[model]['ROC AUC']
            }
            for model in results.keys()
        }).T
        
        # Radar chart
        fig = go.Figure()
        
        for model in metrics_df.index:
            fig.add_trace(go.Scatterpolar(
                r=metrics_df.loc[model].values,
                theta=metrics_df.columns,
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        best_model = results[best_model_name]['Model']
        
        if hasattr(best_model, 'feature_importances_'):
            st.subheader(f"🎯 Feature Importance - {best_model_name}")
            
            importances = best_model.feature_importances_
            feature_names = st.session_state.predictor.feature_names
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df.head(10), 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Top 10 Most Important Features"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve
        st.subheader(f"📊 ROC Curve - {best_model_name}")
        
        y_test = st.session_state.y_test
        y_pred_proba = results[best_model_name]['Probabilities']
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = results[best_model_name]['ROC AUC']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700, height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix
        st.subheader(f"🎭 Confusion Matrix - {best_model_name}")
        
        y_pred = results[best_model_name]['Predictions']
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual"),
            x=['Alive', 'Dead'],
            y=['Alive', 'Dead']
        )
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
