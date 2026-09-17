# 🔍 DataLens AI – Dataset Understanding Assistant

**DataLens AI** is an intelligent, interactive web application designed to instantly profile, visualize, and explain any arbitrary tabular dataset. Built for data analysts, students, and developers, it bridges the gap between raw numbers and intuitive comprehension.

---

## 🚀 Key Features

* **Universal CSV Ingestion:** Automatically adapts to any arbitrary dataset with dynamic header and data type detection.
* **Deterministic Statistical Engine:** Computes exact metrics (row/column counts, missing values, duplicates, descriptive statistics, and correlation matrices) locally using **Pandas** and **NumPy**.
* **Interactive Visualizations:** Dynamically generates histograms, categorical bar charts, correlation heatmaps, and missing value overviews using **Plotly**.
* **AI-Powered Narrative Insights:** Utilizes the **Google Gemini API** to translate mathematical summaries into executive overviews, data pattern insights, and categorized analysis questions (Beginner, Intermediate, Advanced).
* **Clean Architecture:** Strict separation of concerns ensuring robust local computation and lightweight, safe AI interactions.

---

## 📂 Project Architecture

```text
DataLens-AI/
│
├── app.py              # Main Streamlit dashboard orchestrator
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
├── .streamlit/
│   └── secrets.toml    # Secure local API key configuration
└── utils/
    ├── __init__.py
    ├── data_analyzer.py   # Pandas & NumPy statistical processing
    ├── visualizations.py  # Plotly interactive chart generation
    └── ai_service.py      # Google Gemini API integration

```

## 🛠️ Technology Stack
- Frontend & UI: Streamlit

- Data Processing: Pandas, NumPy

- Visualization: Plotly

- AI Intelligence: Google GenAI SDK (gemini-2.5-flash)

- Deployment: Streamlit Community Cloud
