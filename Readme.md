# Agricultural Yield Prediction System

## Overview
This project aims to predict agricultural yield using a Random Forest model. The system utilizes various features such as soil characteristics, weather data, and crop types to forecast yield for different regions.

## Features
- **Data Preprocessing**: Cleans and prepares large datasets for model training.
- **Model Training**: Uses a Random Forest classifier to train the yield prediction model.
- **Streamlit App**: Provides an interactive interface where users can upload data files and visualize predictions.

## Requirements

The necessary Python packages are listed in `requirements.txt`. To install them, first, create a virtual environment.

```bash
pip install -r requirements.txt
Installation and Setup
Step 1: Clone the Repository
bash
Copy code
git clone <repository-url>
cd agricultural-yield-prediction
Step 2: Create and Activate a Virtual Environment
On Windows:

bash
Copy code
python -m venv venv
venv\\Scripts\\activate
On macOS/Linux:

bash
Copy code
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
Once your virtual environment is activated, install the required dependencies using:

bash
Copy code
pip install -r requirements.txt
Step 4: Deactivate the Virtual Environment
To deactivate the virtual environment at any point, use:

bash
Copy code
deactivate
Data Preparation
The raw data files should be stored in the data/raw/ directory. To process the data and generate a clean dataset, run:

bash
Copy code
python src/data_preparation.py
This will create a processed dataset in the data/processed/ directory.

Model Training
To train the Random Forest model, execute the following command:

bash
Copy code
python src/train.py
The model will be saved as a .pkl file in the model/ directory.

Running the Streamlit Application
To start the Streamlit app and use the interface for predictions, run: 

bash
Copy code
streamlit run app.py
The app will open in your default web browser, where you can upload test CSV files, predict agricultural yield, and view visualizations like bar charts and pie charts.