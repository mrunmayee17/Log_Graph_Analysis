# Sensor Log Analysis & Failure Detection using Streamlit, cuGraph & ArangoDB

## Overview

This project provides a **Streamlit-based UI** for analyzing **sensor log data** and detecting failures using **cuGraph (GPU-accelerated graph analytics)** and **ArangoDB (graph database)**. 

It enables users to:
- 📂 Upload **CSV or GZ-compressed sensor logs**
- 📊 Detect **sensor failures** (e.g., speed, battery, brake, steering issues)
- 🔍 **Visualize sensor dependency graphs**
- 🤖 Perform **Graph-Based Question Answering (QA)** using **cuGraph, ArangoDB & LLMs (Google Gemini & Groq)**
- ⚡ Utilize **GPU acceleration with cuGraph** for efficient computations

---

## Features

✔ Upload and process **large sensor log files**  
✔ Detect **sensor failures** and display insights  
✔ Build a **sensor dependency graph** with **timestamped failures**  
✔ Visualize the **network graph** of sensors  
✔ Query failures using **Graph-Based Retrieval & LLMs**  

---

## Submission Notebook: Sensor_failure_Analysis_Submission.ipynb
## Demo Video: Sensor_Log_Analysis.mp4
## 🏗️ Project Architecture

```
Sensor Logs (CSV/GZ) → Data Processing → Failure Detection → Graph Construction (NetworkX)
  → Store in ArangoDB → Convert to cuGraph → Streamlit UI → Graph-Based Querying (LLMs & cuGraph)
```

## 📂 Installation & Setup

### Install Dependencies

#### Using Conda (Recommended)

Make sure you have Miniconda installed. Then, create a Conda environment and install the required packages:

```bash
conda create -n sensor_env python=3.10 -y
conda activate sensor_env
conda install -c rapidsai -c nvidia -c conda-forge -c defaults \
    rapids=24.02 \
    python=3.10 \
    cudatoolkit=11.8 \
    cugraph cudf cupy
```

#### Install additional dependencies

```bash
pip install streamlit pandas networkx matplotlib python-arango \
    google-generativeai langchain langchain-google-genai langchain-groq \
    arango dotenv
```

### Run the Application

```bash
streamlit run app.py
```

## 📂 File Structure

```
📂 Sensor-Log-Analyzer/
│── 📜 app.py                   # Main Streamlit UI & Graph Processing
│── 📜 requirements.txt         # Dependencies
│── 📜 .env                     # API keys & DB credentials
│── 📜 README.md                # Project Documentation
```

## How It Works

### Upload Sensor Logs
- Upload a .csv or .gz file containing sensor readings over time
- The tool automatically detects failures (e.g., speed, braking, steering)

###  Build Sensor Graph
- Construct a sensor dependency network using NetworkX
- Store the graph in ArangoDB
- Convert it to cuGraph for GPU-accelerated graph processing

### 📊 Visualize the Graph
Display an interactive graph where:
- ✅ Green nodes = Normal sensors
- ❌ Red nodes = Failing sensors

### 🤖 Ask Questions
Query failures using cuGraph, ArangoDB, and LLMs

Example queries:
- "Which sensors failed first?"
- "Find the shortest path between FL_wheel_speed and battery_level"
- "Which sensor has the highest degree centrality?"

## 🛠️ Troubleshooting

### 1️⃣ Missing cuGraph Installation?
Run:
```bash
conda install -c rapidsai -c nvidia -c conda-forge -c defaults cugraph cudf cupy
```

### 2️⃣ CUDA Issues?
Check:
```bash
nvidia-smi
nvcc --version
```

### 3️⃣ ArangoDB Connection Errors?
Ensure your .env file contains:
```
DATABASE_HOST=<ArangoDB_Host_URL>
DATABASE_NAME=sensor_graph_db
DATABASE_USERNAME=root
DATABASE_PASSWORD=<Your_Password>
```
