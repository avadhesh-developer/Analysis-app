from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from research import multiagent_pdf_rag_pipeline, process_pdf, image_data_store, table_data_store
import pandas as pd
import tempfile
import os

app = FastAPI(title="Multi-Agent PDF RAG API")

@app.post("/analyze")
async def analyze_pdf(file: UploadFile, query: str = Form(...)):
    # Validate input
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(content={"error": "Invalid File Type. Please upload PDF File"}, status_code=400)

    if not query.strip():
        return JSONResponse(content={"error": "Query Cannot be Empty"}, status_code=400)

    # Save uploaded PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_path = tmp.name
        content = await file.read()
        tmp.write(content)

    response_data = {}
    try:
        # Process PDF and run analysis
        process_pdf(pdf_path)  # make sure this closes file internally
        final_answer = multiagent_pdf_rag_pipeline(query, pdf_path)

        response_data = {
            "status": "success",
            "final_answer": final_answer,
        }

        # Include images only if available
        if image_data_store:
            images = [{"id": img_id, "base64": img_b64} for img_id, img_b64 in image_data_store.items()]
            response_data["images"] = images

        # Include tables only if available
        if table_data_store:
            tables = {}
            for table_id, table_data in table_data_store.items():
                try:
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    tables[table_id] = df.to_dict(orient="records")
                except Exception:
                    tables[table_id] = {"error": "could not parse into DataFrame"}
            response_data["tables"] = tables

    finally:
        # Safe file cleanup (close before removing)
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
        except PermissionError:
            # fallback: delay delete
            import time
            time.sleep(0.5)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
