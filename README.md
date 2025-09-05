# Flask Prediction Models

This repository contains three machine learning models wrapped in a Flask API with a simple web interface:  

1. **Salary Prediction** – predicts salary based on years of experience.  
2. **Income Prediction** – predicts income based on age and experience.  
3. **Advertising Sales Prediction** – predicts sales based on TV and Radio advertising budgets.  

---

## Getting Started (Local)

### Prerequisites
- Python 3.10+  
- `pip`  
- Virtual environment recommended

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd <repo-folder>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run Flask App
python app.py

The web interface will be available at:
http://127.0.0.1:5000/

Web Interface Usage
1. Open the URL above in your browser.
2. Navigate to the desired model using the buttons.
3. Fill in the input fields and click Predict.
4. The predicted output will appear below the form.

API Usage (Postman or cURL)
1. Salary Prediction
POST http://127.0.0.1:5000/predict_salary
Body (JSON):
{
  "YearsExperience": 5.0
}
> **Note:** The value for `YearsExperience` can be any valid numbers (e.g., `5.0`).

2. Income Prediction
POST http://127.0.0.1:5000/predict_income
Body (JSON):
{
  "age": 30.3,
  "experience": 5.1
}
> **Note:** `age` and `experience` can be any valid numbers (e.g., `30.3`, `5.1`)

3. Sales Prediction
POST http://127.0.0.1:5000/predict_sales
Body (JSON):
{
  "TV": 200,
  "Radio": 50
}
> **Note:** `TV` and `Radio` can be any valid numbers (e.g., `200`, `50`)
