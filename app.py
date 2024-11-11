import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
import warnings
from src.data_preparation import prepare_data  # Import the prepare_data function

warnings.filterwarnings("ignore")

# Load the pre-trained Random Forest model
model_path = 'model/yield_model_rf.pkl'
rf_classifier = joblib.load(model_path)  # Load the model

# Set page title and layout
st.set_page_config(page_title="Agricultural Yield Prediction", layout="wide")
st.title("🌾 Agricultural Yield Prediction")
st.markdown("This app predicts agricultural yield based on input features using a pre-trained Random Forest model.")

# File uploader for testing data
st.subheader("Upload Your Testing Data CSV")
uploaded_test_file = st.file_uploader("Choose a CSV file for testing", type="csv")

# Load datasets after upload
if uploaded_test_file is not None:
    test = pd.read_csv(uploaded_test_file)

    # Sample the first 200 rows of the test dataset
    # test_sample = test.sample(n=min(200, len(test)), random_state=42)  # Ensure no more than 200 rows are sampled

    # Prepare data
    xtest, ytest = prepare_data(test)  # Ensure this function processes the data correctly

    # Make predictions
    if st.button("Predict"):
        ypred = rf_classifier.predict(xtest)

        # If ground truth is available, calculate R² score
        if 'Yield_kg_per_hectare' in test.columns:
            rf_score = r2_score(ytest, ypred)
            st.success("Predictions Complete!")
            st.write(f"Random Forest R² Score: {rf_score:.4f}")

            # Display Predictions
            predictions_df = pd.DataFrame({
                'Actual Yield': ytest,
                'Predicted Yield': ypred
            })

            # 1. Line Chart
            st.subheader("Line Chart: Actual vs. Predicted Yield")
            line_fig = go.Figure()
            line_fig.add_trace(go.Scatter(
                x=predictions_df.index,
                y=predictions_df['Actual Yield'],
                mode='lines+markers',
                name='Actual Yield',
                line=dict(color='blue'),
                marker=dict(size=6)
            ))
            line_fig.add_trace(go.Scatter(
                x=predictions_df.index,
                y=predictions_df['Predicted Yield'],
                mode='lines+markers',
                name='Predicted Yield',
                line=dict(color='orange'),
                marker=dict(size=6)
            ))
            line_fig.update_layout(title='Line Chart: Comparison of Actual vs. Predicted Yield',
                                   xaxis_title='Index',
                                   yaxis_title='Yield (kg per hectare)',
                                   template='plotly_dark')
            st.plotly_chart(line_fig)

            # 3. Scatter Plot
            st.subheader("Scatter Plot: Actual vs. Predicted Yield")
            scatter_fig = px.scatter(predictions_df, x='Actual Yield', y='Predicted Yield',
                                      title="Scatter Plot: Actual vs. Predicted Yield",
                                      labels={'Actual Yield': 'Actual Yield', 'Predicted Yield': 'Predicted Yield'})
            scatter_fig.add_shape(type="line", x0=min(predictions_df['Actual Yield']),
                                   y0=min(predictions_df['Actual Yield']),
                                   x1=max(predictions_df['Actual Yield']),
                                   y1=max(predictions_df['Actual Yield']),
                                   line=dict(color='red', dash='dash'))
            scatter_fig.update_layout(xaxis_title='Actual Yield', yaxis_title='Predicted Yield', template='plotly_dark')
            st.plotly_chart(scatter_fig)

            # Display predictions data
            st.subheader("Predicted Values")
            st.write(predictions_df)

        else:
            st.success("Predictions Complete! (Ground truth not available for scoring)")
            st.subheader("Predicted Values")
            st.write(ypred)

# Footer
st.markdown("---")

