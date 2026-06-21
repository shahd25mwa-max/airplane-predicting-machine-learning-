# Flight Analytics & Prediction System

This project contains two Machine Learning models integrated into a Streamlit web application:
1. **Flight Delay Prediction**: Predicts expected arrival delays using Linear Regression (Accuracy: ~99%).
2. **Ticket Price Prediction**: Predicts expected ticket prices based on economic and geopolitical factors using RandomForestRegressor (Accuracy: ~99%).

## Project Structure
- `ML_Final.py`: The main Python file containing data engineering, model training, and the Streamlit interface.
- `arrival_model.pkl` & `ticket_prices.pkl`: Saved trained models.
- `requirements.txt`: List of required Python libraries.

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run the application: `python -m streamlit run ML_Final.py`