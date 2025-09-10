import streamlit as st
import base64
import pandas as pd
import requests
import streamlit.components.v1 as components

API_URL = "http://localhost:8000/analyze"



st.set_page_config(page_title="Multi-Agent PDF RAG", page_icon="📄", layout="wide")


st.markdown(
"""
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Header */
h2, h3 {
    text-align: center;
    color: #2c3e50;
}
p {
    text-align: center;
    font-style: italic;
    color: #7f8c8d;
}

/* Sidebar */
[aria-label="Sidebar"] > div {
    padding: 20px 10px 20px 20px;
    background-color: #f0f4f8;
}
.stTextInput>div>input, .stFileUploader>div>input {
    border: 1.8px solid #4b6cb7;
    border-radius: 6px !important;
    padding: 8px 10px;
    font-size: 16px;
}
.stTextInput>div>input:focus, .stFileUploader>div>input:focus {
    outline: none;
    border-color: #182848;
    box-shadow: 0 0 8px rgba(24,40,72,0.5);
}
div.stButton > button {
    background-color: #4b6cb7;
    color: white;
    padding: 12px 25px;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
    box-shadow: 2px 4px 6px rgba(75,108,183,0.4);
    transition: all 0.3s ease;
    width: 100%;
    margin-top: 15px;
    cursor: pointer;
}
div.stButton > button:hover {
    background-color: #182848;
    box-shadow: 2px 4px 12px rgba(24,40,72,0.8);
}

/* Expanders */
.stExpander {
    background: #ffffff;
    border-radius: 12px;
    border: 1.5px solid #4b6cb7;
    box-shadow: 3px 5px 10px rgba(75,108,183,0.15);
    padding: 14px 20px;
    margin-bottom: 20px;
}
.stExpander:hover {
    box-shadow: 3px 5px 20px rgba(24,40,72,0.25);
}

/* Images */
div[data-testid="stImage"] {
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    margin-bottom: 12px;
}

/* Tables */
table {
    width: 100% !important;
    border-collapse: collapse;
    border-radius: 8px;
    overflow: hidden;
    font-size: 16px;
    margin-top: 12px;
    box-shadow: 0 3px 12px rgba(75,108,183,0.15);
}
th, td {
    padding: 12px 15px;
    border: 1px solid #e1e5ea;
    text-align: left;
}
th {
    background-color: #4b6cb7;
    color: white;
    font-weight: 700;
}
tbody tr:nth-child(even) {
    background-color: #f9faff;
}
tbody tr:hover {
    background-color: #dbe6ff;
}

/* Alert Styling */
[data-testid="stInfo"] {
    background-color: #d4e3fc !important;
    border-left: 5px solid #4b6cb7;
}
[data-testid="stError"] {
    background-color: #fcdada !important;
    border-left: 5px solid #e74c3c;
}
[data-testid="stSuccess"] {
    background-color: #d7fbe8 !important;
    border-left: 5px solid #27ae60;
}
</style>
""",
unsafe_allow_html=True
)

# App Header
st.markdown("## 📄 Multi-Agent PDF RAG Pipeline")
st.markdown("_Analyze text, tables, and images from PDFs using AI agents_")

# Sidebar Inputs
with st.sidebar:
    st.header("Input")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    query = st.text_input("Enter your research query")
    run_button = st.button("Run Analysis")


# Main Logic
if run_button:
    if uploaded_file and query.strip():
        with st.spinner("Sending to backend..."):
            try:
                response = requests.post(
                    API_URL,
                    files={"file": uploaded_file},
                    data={"query": query}
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")
                st.stop()

        if response.status_code == 200:
            result = response.json()
            st.success("✅ Analysis Complete!")

            # Final Answer
            with st.expander("🗣️ Final Answer", expanded=True):
                st.write(result.get("final_answer", "No answer returned."))

            #  Images
            images = result.get("images", [])
            if images:
                with st.expander(" Extracted Images"):
                    cols = st.columns(3)
                    for i, img in enumerate(images):
                        try:
                            img_data = base64.b64decode(img["base64"])
                            cols[i % 3].image(img_data, caption=img["id"])
                        except Exception as e:
                            st.error(f"Error rendering image {img['id']}: {e}")
            else:
                st.info("No images found in the PDF.")

            # Tables
            tables = result.get("tables", {})
            if tables:
                with st.expander(" Extracted Tables"):
                    for table_id, table_records in tables.items():
                        if isinstance(table_records, dict) and "error" in table_records:
                            st.warning(f"{table_id}: {table_records['error']}")
                        else:
                            try:
                                df = pd.DataFrame(table_records)
                                st.markdown(f"**{table_id}**")
                                st.dataframe(df)
                            except Exception as e:
                                st.error(f"Error rendering {table_id}: {e}")
            else:
                st.info("No tables found in the PDF.")

        else:
            st.error(f"Backend error: {response.text}")
    else:
        st.error("Please upload a PDF and enter a query.")
