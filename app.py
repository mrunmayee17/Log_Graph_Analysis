import os
import re
import gzip
import streamlit as st
import pandas as pd
import networkx as nx
import nx_cugraph as nxcg
import cugraph as cg
import matplotlib.pyplot as plt
import google.generativeai as genai

from arango import ArangoClient
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs import ArangoGraph
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from io import StringIO


# ========================
# SETUP: LangChain & ArangoDB
# ========================
load_dotenv()
ARANGO_HOST = os.getenv("DATABASE_HOST")
DB_NAME = os.getenv("DATABASE_NAME")
USERNAME = os.getenv("DATABASE_USERNAME")
PASSWORD = os.getenv("DATABASE_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM models
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GOOGLE_API_KEY)
groq_llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")

# Connect to ArangoDB
client = ArangoClient(hosts=ARANGO_HOST)
db = client.db(DB_NAME, username=USERNAME, password=PASSWORD)
arango_graph = ArangoGraph(db)

# ========================
# SENSOR FAILURE DETECTION
# ========================
# Function to detect sensor failures
def detect_failures(df):
    """Analyze data and identify sensor failures"""
    failures = []

    for index, row in df.iterrows():
        failure_entry = {"utime": row["utime"], "failures": []}

        #  Wheel Speed Failures
        for wheel in ["FL_wheel_speed", "FR_wheel_speed", "RL_wheel_speed", "RR_wheel_speed"]:
            if wheel in row and (row[wheel] < 0 or row[wheel] > 150):
                failure_entry["failures"].append(wheel)

        #  Battery & Power Failures
        if "battery_level" in row and row["battery_level"] < 20:
            failure_entry["failures"].append("battery_level")

        #  Brake Failures
        if "brake" in row and row["brake"] > 0.9 and "throttle" in row and row["throttle"] > 100:
            failure_entry["failures"].append("brake_throttle_conflict")

        #  Steering Failures
        if "steering" in row and (row["steering"] > 30 or row["steering"] < -30):
            failure_entry["failures"].append("steering_out_of_bounds")

        #  Throttle Failures
        if "throttle" in row and row["throttle"] > 100:
            failure_entry["failures"].append("throttle_sensor_failure")

        # Vehicle Speed & Acceleration Failures
        if "vehicle_speed" in row and (row["vehicle_speed"] < 0 or row["vehicle_speed"] > 150):
            failure_entry["failures"].append("vehicle_speed_out_of_range")
        # Gear & Signal Failures**
        if row.get("gear_position", -1) not in range(0, 8):
            failure_entry["failures"].append("Invalid Gear Position")
        if row.get("left_signal", 0) not in [0, 1]:
            failure_entry["failures"].append("Left Signal Malfunction")
        if row.get("right_signal", 0) not in [0, 1]:
            failure_entry["failures"].append("Right Signal Malfunction")

        # Position & Orientation Failures**
        if row.get("available_distance", 0) < 0:
            failure_entry["failures"].append("Distance Sensor Issue")
        if row.get("yaw_rate", 0) > 5 or row.get("yaw_rate", 0) < -5:
            failure_entry["failures"].append("Yaw Rate Out of Range")

        if failure_entry["failures"]:
            failures.append(failure_entry)

    return failures

# ========================
# BUILD SENSOR GRAPH
# ========================
# Function to build the sensor graph
def build_sensor_graph(failures, sensor_names):
    G = nx.Graph()

    # Store timestamps per sensor
    sensor_timestamps = {}

    # Ensure timestamps are properly stored
    for _, row in df_subset.iterrows():
        timestamp = row["utime"]
        for sensor in sensor_names:
            if pd.notna(row[sensor]): 
                if sensor not in sensor_timestamps or sensor_timestamps[sensor] < timestamp:
                    sensor_timestamps[sensor] = timestamp  # Store latest available timestamp

    # Add sensor nodes with timestamps
    for sensor in sensor_names:
        timestamp = sensor_timestamps.get(sensor, 0)  
        G.add_node(sensor, status="OK", utime=timestamp)

    # Mark failed sensors and assign timestamps
    for failure in failures:
        failure_time = failure["utime"]
        for sensor in failure["failures"]:
            if sensor in G.nodes:
                G.nodes[sensor]["status"] = "failing"
                G.nodes[sensor]["utime"] = failure_time 

    # Define sensor relationships
    sensor_dependencies = [
        #  Wheel Speed Dependencies
        ("FL_wheel_speed", "FR_wheel_speed"),
        ("RL_wheel_speed", "RR_wheel_speed"),
        ("FL_wheel_speed", "RL_wheel_speed"),
        ("FR_wheel_speed", "RR_wheel_speed"),
        ("FL_wheel_speed", "vehicle_speed"),
        ("FR_wheel_speed", "vehicle_speed"),
        ("RL_wheel_speed", "vehicle_speed"),
        ("RR_wheel_speed", "vehicle_speed"),

        #  Steering Dependencies
        ("steering", "steering_sensor"),
        ("steering", "steering_speed"),
        ("steering_sensor", "steering_speed"),
        ("steering", "vehicle_speed"),
        ("steering", "yaw_rate"),
        ("steering", "throttle"),

        #  Throttle Dependencies
        ("throttle", "throttle_sensor"),
        ("throttle", "vehicle_speed"),
        ("throttle", "longitudinal_accel"),
        ("throttle_sensor", "longitudinal_accel"),

        #  Brake Dependencies
        ("brake", "brake_sensor"),
        ("brake", "brake_switch"),
        ("brake", "vehicle_speed"),
        ("brake", "throttle"),
        ("brake_sensor", "brake_switch"),
        ("brake", "longitudinal_accel"),
        ("brake", "regen"),

        #  Battery & Power Dependencies
        ("battery_level", "available_distance"),
        ("battery_level", "vehicle_speed"),
        ("battery_level", "steering"),

        #  Positioning & Acceleration
        ("longitudinal_accel", "vehicle_speed"),
        ("transversal_accel", "yaw_rate"),
        ("pos", "available_distance"),

        #  Gear & Signal Dependencies
        ("gear_position", "vehicle_speed"),
        ("left_signal", "steering"),
        ("right_signal", "steering"),
    ]


    # Add edges with timestamps if available
    for s1, s2 in sensor_dependencies:
        if s1 in G.nodes and s2 in G.nodes:
            s1_utime = G.nodes[s1].get("utime") or 0
            s2_utime = G.nodes[s2].get("utime") or 0
            edge_utime = max(s1_utime, s2_utime)  # Use the latest timestamp
            G.add_edge(s1, s2, utime=edge_utime)

    return G
# =============================================
# DATABASE SETUP
# =============================================
def setup_database():
    if not os.getenv("DATABASE_HOST") or not os.getenv("DATABASE_NAME"):
        raise ValueError("ArangoDB database environment variables are not set! Set DATABASE_HOST and DATABASE_NAME.")

    client = ArangoClient(hosts=os.getenv("DATABASE_HOST"), request_timeout=None)
    sys_db = client.db("_system", username=USERNAME, password=PASSWORD)

    if not sys_db.has_database(DB_NAME):
        sys_db.create_database(DB_NAME)

    db = client.db(DB_NAME, username=USERNAME, password=PASSWORD, verify=True)
    arango_graph = ArangoGraph(db)


    if not db.has_collection("sensors"):
        db.create_collection("sensors")

    if not db.has_collection("sensor_connections"):
        db.create_collection("sensor_connections", edge=True)

    return db, arango_graph


# =============================================
# STORE GRAPH IN ARANGO
# =============================================
def debug_arango_graph(db):
    sensors = list(db.collection("sensors").all())
    edges = list(db.collection("sensor_connections").all())

    return sensors, edges
def store_graph_in_arango(db, sensor_graph):

    db.collection("sensors").truncate()
    db.collection("sensor_connections").truncate()

    for node, data in sensor_graph.nodes(data=True):
        db.collection("sensors").insert({
            "_key": str(node),
            "name": str(node),
            "status": data.get("status", "OK"),
            "utime": data.get("utime", "N/A")  # Adding timestamp
        }, overwrite=True)

    for s1, s2 in sensor_graph.edges():
        db.collection("sensor_connections").insert({
            "_from": f"sensors/{s1}",
            "_to": f"sensors/{s2}",
            "utime": sensor_graph[s1][s2].get("utime", "N/A")  # Adding timestamp
        })
    stored_sensors = list(db.collection("sensors").all())
    failing_count = sum(1 for sensor in stored_sensors if sensor.get("status") == "failing")
    
# =============================================
# CONVERT ARANGODB GRAPH TO cuGraph
# =============================================
@st.cache_data
def cached_cugraph_conversion(sensor_graph):
    """Cache cuGraph conversion to avoid recomputation"""
    try:
        return nxcg.from_networkx(sensor_graph)
    except Exception:
        return None

def convert_arango_to_cugraph(graph_name):
    if not isinstance(graph_name, nx.Graph):
        return None

    try:
        G_cu = nxcg.from_networkx(graph_name)
        return G_cu
    except Exception as e:
        return None

# =============================================
# AQL, LLM, CU QUERIES
# =============================================
@tool
def text_to_aql_to_text(query: str):
    """Invoke ArangoGraphQAChain for AQL-based graph queries with proper error handling."""
    chain = ArangoGraphQAChain.from_llm(
        llm=groq_llm,
        graph=arango_graph,
        verbose=True,
        allow_dangerous_requests=True
    )
    corrected_aql_query = """
    WITH sensors, sensor_connections
    FOR sensor IN sensors
      FILTER sensor.status == "failing"
      SORT sensor.utime ASC
      RETURN {
        "sensor": sensor.name,
        "utime": sensor.utime
      }
    """

    try:
        result = chain.invoke(corrected_aql_query)
        return str(result["result"])
    except Exception as e:
        return f"AQL Execution Error: {str(e)}"

def text_to_nx_algorithm_to_text(query, sensor_graph):
    """Executes NetworkX graph algorithms on the existing sensor network."""

    # print("1) Generating NetworkX code")

    # Generate NetworkX Code from LLM
    text_to_nx = groq_llm.invoke(f"""
    I have a NetworkX Graph called `sensor_graph`. It has the following schema: {sensor_graph.nodes(data=True)}.

    I have the following graph analysis query: {query}.

    Generate **fully executable** Python Code that:
    1. Has no unterminated string literals.
    2. Has properly formatted multi-line strings if needed.
    3. Does not include explanation text.
    4. Uses \"\"\" for any long multi-line strings.
    5. Ends with the variable `FINAL_RESULT`.

    Only provide executable code. No markdown or explanations.
    """).content


    # Fix 1: Clean LLM Formatting Artifacts
    text_to_nx_cleaned = re.sub(r'^\s*```python\s*|\s*```$', '', text_to_nx, flags=re.MULTILINE).strip()
    if text_to_nx_cleaned.count('"""') % 2 != 0 or text_to_nx_cleaned.count("'''") % 2 != 0:
      text_to_nx_cleaned += '\n"""'  # Auto-fix missing closing quotes
    try:
        compile(text_to_nx_cleaned, '<string>', 'exec')
    except SyntaxError as e:
        # print(f"❌ Syntax Error in generated code: {e}")
        return f"❌ Syntax Error in generated code: {e}"

    # print("-" * 10)
    # print("Generated NetworkX Code:")
    # print(text_to_nx_cleaned)
    # print("-" * 10)

    # ✅ Fix 2: Check if Code Contains Unterminated String Issues
    if re.search(r'""".*"""', text_to_nx_cleaned, re.DOTALL) or re.search(r"'''.*'''", text_to_nx_cleaned, re.DOTALL):
        # print("⚠️ Warning: Multi-line string detected. Checking for syntax issues...")
        pass

    # ✅ Fix 3: Ensure sensor_graph exists before execution
    global_vars = {"sensor_graph": sensor_graph, "nx": nx}
    local_vars = {}

    try:
        exec(text_to_nx_cleaned, global_vars, local_vars)
        FINAL_RESULT = local_vars.get("FINAL_RESULT", "Execution Failed")
        # print(f"FINAL_RESULT: {FINAL_RESULT}")
        return FINAL_RESULT

    except SyntaxError as e:
        # print(f"❌ Syntax Error Detected: {e}")
        return f"❌ Syntax Error: {e}"

    except Exception as e:
        # print(f"❌ EXEC ERROR: {e}")
        return f"❌ EXEC ERROR: {e}"

# =============================================
# CUGRAPH ANALYSIS
# =============================================
def analyze_graph_with_cugraph(G_cu):

    # print(" Running cuGraph analytics...")

    try:
        # Compute Degree Centrality
        degree_centrality = nx.degree_centrality(G_cu)

        # Find the node with the highest degree centrality
        most_connected_sensor = max(degree_centrality, key=degree_centrality.get)

        # print(f"✅ Most connected sensor (highest degree centrality): {most_connected_sensor}")

        return {
            "degree_centrality": degree_centrality,
            "most_connected_sensor": most_connected_sensor
        }

    except Exception as e:
        # print(f"❌ cuGraph Error: {str(e)}")
        return {"degree_centrality": {}, "most_connected_sensor": None}

# =============================================
# Query Routing
# =============================================
@st.cache_data
def cached_llm_response(formatted_results):
    """Cache LLM response to avoid redundant API calls"""
    try:
        response = groq_llm.invoke(formatted_results)
        return response if response and "No useful answer" not in response else "⚠️ No valid response found."
    except Exception as e:
        return f"❌ Error selecting the best answer: {str(e)}"
def extract_source_target(query):
    """Extract source and target nodes from the query using regex"""
    match = re.search(r"shortest path from (\w+) to (\w+)", query, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    return None, None

def decide_best_answer(query, results):
    """
    Uses LLM to analyze results from cuGraph, NetworkX, and AQL, then selects the best response.
    """
    # print("\n🤖 Selecting the best answer from cuGraph, NetworkX, and AQL...\n")

    # Format results for LLM analysis
    formatted_results = f"""
    Query: {query}

    cuGraph Result:
    {results.get("cuGraph", "No cuGraph result.")}

    NetworkX Result:
    {results.get("NetworkX", "No NetworkX result.")}

    AQL Result:
    {results.get("AQL", "No AQL result.")}

    Select the best answer and include the best answer in your output, the answer that is most relevant, accurate, and provides the clearest response.
    If no answer is useful, select the second-best option that provides the closest possible answer.
    """

    # 🔹 Invoke LLM with retry logic to handle 429 errors
    try:
        response = cached_llm_response(formatted_results)

        if response == "⚠️ API quota exceeded. Using AQL results as fallback.":
            return results.get("AQL", "No useful answer available.")

        # print(f"✅ Best Answer Selected:\n{response}")
        return response

    except Exception as e:
        # print(f"❌ LLM Decision Error: {str(e)}")
        return results.get("AQL", "❌ Error in selecting the best answer.")

def query_graph(query, sensor_graph, G_cu):

    # print(f"🤖 Processing query: {query}")

    results = {}

    # Ensure cuGraph is available
    if G_cu is None:
        # print("⚡ Fetching cuGraph representation from ArangoDB...")
        G_cu = cached_cugraph_conversion(sensor_graph)

    if "most connected sensor" in query.lower():
        # print("🔍 Computing most connected sensor using cuGraph...")
        results["cuGraph"] = analyze_graph_with_cugraph(G_cu)


    # Always try cuGraph first
    try:
        results["cuGraph"] = analyze_graph_with_cugraph(G_cu)
    except Exception as e:
        results["cuGraph"] = f"❌ cuGraph Error: {str(e)}"


    # Check for shortest path queries
    source, target = extract_source_target(query)
    if source and target:
        # print(f"🔍 Detected shortest path query from '{source}' to '{target}'")
        if nx.has_path(G_cu, source, target):
            shortest_path = nx.shortest_path(G_cu, source=source, target=target, backend="cugraph")
            # print(f"✅ Shortest path: {shortest_path}")
            results["cuGraph"] = shortest_path
        else:
            # print(f"❌ No path found between '{source}' and '{target}'")
            results["cuGraph"] = "No path exists."


    # NetworkX Analysis
    try:
        results["NetworkX"] = text_to_nx_algorithm_to_text(query, sensor_graph)
    except Exception as e:
        results["NetworkX"] = f"❌ NetworkX Error: {str(e)}"

    # ArangoDB Query (AQL)
    try:
        results["AQL"] = text_to_aql_to_text.invoke(query)
    except Exception as e:
        results["AQL"] = f"❌ AQL Error: {str(e)}"

    # Let LLM decide best response
    final_answer = decide_best_answer(query, results) or "⚠️ No valid response found. Check query inputs."
    return final_answer

# ========================
# STREAMLIT UI
# ========================
st.title("🔍 Sensor Log Analysis & Insights")
st.sidebar.header("Upload Sensor Log File")
# st.set_option('server.maxUploadSize', 500)

uploaded_file = st.sidebar.file_uploader("Upload your sensor log CSV", type=["csv", "gz"])
if uploaded_file is not None:
    file_name = uploaded_file.name

    try:
        # Check if the file is a .gz compressed file
        if file_name.endswith(".gz"):
            with gzip.open(uploaded_file, "rt") as f:
                df = pd.read_csv(f, low_memory=False)
                df_subset = df
        else:
            df = pd.read_csv(uploaded_file, low_memory=False)
            df_subset = df

        if df.empty:
            st.error("❌ Uploaded file is empty or could not be read!")


    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

    # Show raw data
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Uploaded Sensor Data")
        st.write(df_subset.head())

    db, arango_graph = setup_database()
    failures = detect_failures(df_subset)
    sensor_names = df_subset.columns.tolist()
    sensor_graph = build_sensor_graph(failures, sensor_names)
    store_graph_in_arango(db, sensor_graph)
    G_cu = convert_arango_to_cugraph(sensor_graph)

    if st.sidebar.checkbox("Show Sensor Graph"):
        st.subheader("📡 Sensor Network Graph")
        plot_options = {"node_size": 50, "with_labels": True, "width": 0.5}
    
        # Compute node positions using spring layout
        pos = nx.spring_layout(sensor_graph, iterations=15, seed=1721)

        # Create the figure and axis
        color_map = ["red" if sensor_graph.nodes[node].get("status") == "failing" else "green" for node in sensor_graph.nodes]
        fig, ax = plt.subplots(figsize=(15, 9))
        nx.draw_networkx(sensor_graph, pos=pos, ax=ax, node_color=color_map, **plot_options)
        st.pyplot(fig)  
        
    
    st.subheader("🤖 Ask a Question")
    user_query = st.text_input("Enter your query:")
    if user_query:
        answer = query_graph(user_query,sensor_graph,G_cu)
        cleaned_content = re.sub(r'\n\n', ' ', answer["content"] if isinstance(answer, dict) else answer.content)

        st.write("### 🤖 Response:")
        st.success(cleaned_content)

