# Heart Disease Detection - ML Project

## Setup
1. Create virtual env:
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate

2. Install:
   pip install -r requirements.txt

3. Place dataset:
   Put `heart.csv` in `data/` (use UCI Heart dataset or the cleaned heart.csv variant).

4. Train model:
   python src/train_model.py

5. Run web app:
   streamlit run website/app.py
