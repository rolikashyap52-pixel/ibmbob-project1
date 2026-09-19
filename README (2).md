# CyberShield Analytics

## Cybersecurity Threat Analytics & Risk Prediction Dashboard

CyberShield Analytics is a Python/Streamlit project that converts cybersecurity event or network-traffic data into KPIs, trends, attack analysis, risk signals, and (when the dataset supports it) a machine-learning classification model.

### Objectives
- Clean and inspect cybersecurity data
- Identify useful KPIs and attack patterns
- Visualize trends and class/attack distributions
- Highlight data-quality and risk signals
- Build a Random Forest classification model when a suitable target exists
- Convert findings into actionable security recommendations

### Suggested dataset
A public Kaggle option is the **Multi-Type Network Attack Detection Dataset**, which contains benign and malicious network traffic and categories such as DoS, DDoS, injection and scanning. It is listed as CC0/Public Domain on its Kaggle data card.

Dataset source:
https://www.kaggle.com/datasets/zoya77/multi-type-network-attack-detection-dataset

**Important:** Do not use the exact dataset used in your internship/masterclass training sessions. Confirm that the dataset you submit is different.

### Technologies
- Python
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Streamlit

### How to run

1. Install Python 3.10+.
2. Put your CSV dataset in the project folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run:

```bash
streamlit run cybersecurity_dashboard.py
```

5. Upload the CSV through the sidebar if it is not automatically detected.

### Expected target columns
The application automatically looks for columns such as:
- `attack_detected`
- `attack_type`
- `attack_type`
- `label`
- `attack`

Rename your target column if needed.

### Project workflow

Raw Data → Cleaning → EDA → KPIs → Trends → Risk/Opportunity → ML (if appropriate) → Actionable Insights

### Deliverables
- `cybersecurity_dashboard.py`
- `requirements.txt`
- `README.md`
- `Project_Report.pdf`

### Academic note
Model metrics are generated from the actual dataset at runtime. The report does not invent accuracy or other results before the dataset is executed.
