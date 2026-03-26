# 🚀 Fraudulent Job Detection & Resume Intelligence System

An end-to-end AI-powered platform designed to detect fraudulent job postings and enhance candidate-job alignment through intelligent resume analysis, scoring, and generative AI feedback.

---

## 📌 Overview

This project improves job search safety and efficiency by:
- Detecting **fraudulent job listings**
- Performing **resume analysis and matching**
- Generating **AI-powered resume feedback**

It combines **Machine Learning, NLP, and Generative AI** to build a complete intelligent system.

---

## 🔥 Features

### 🛡️ Fraud Job Detection
- Classifies job postings as **fraudulent or legitimate**
- Models used:
  - Naive Bayes
  - SGD Classifier
- Text processing using **CountVectorizer**

---

### 📄 Resume Analysis & Parsing
- Extracts key information:
  - Skills
  - Experience
  - Keywords
- Uses:
  - PyPDF2
  - NLP techniques

---

### 🎯 Resume-Job Matching
- Compares resumes with job descriptions
- Generates **similarity scores**
- Ranks candidates based on job fit

---

### 📊 Resume Scoring System
- Evaluates resumes based on:
  - Skill relevance
  - Keyword matching
  - Job requirements
- Outputs a **quantitative score**

---

### 🤖 AI Resume Feedback
- Uses **Google Gemini API (LLM)** to:
  - Generate personalized feedback
  - Suggest improvements
  - Identify missing skills

---

## 🧠 Tech Stack

- **Languages:** Python  
- **Backend:** Flask  
- **Machine Learning:** Scikit-learn, Naive Bayes, SGD Classifier  
- **NLP:** CountVectorizer  
- **Generative AI:** Google Gemini API  
- **Data Processing:** Pandas, NumPy  
- **Resume Parsing:** PyPDF2  
- **Frontend:** HTML, CSS  

---

## ⚙️ System Workflow

1. Upload resume / job description  
2. Job → Fraud Detection Model  
3. Resume → Parsing & Feature Extraction  
4. Matching Engine → Similarity Score  
5. Scoring System → Candidate Evaluation  
6. LLM → Feedback Generation  
7. Results displayed on web interface  

---

## 📈 Performance

- Efficient classification of job postings  
- Handles unstructured textual data effectively  
- Real-time resume analysis and matching  
- AI-generated feedback for improvement  

---

## 🚀 Installation & Setup
1.Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

2.Install dependencies
pip install -r requirements.txt

3.Add API Key
GEMINI_API_KEY=your_api_key_here

4.Run the application
python app.py
