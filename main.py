import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# Load .env and OpenAI API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Define FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQL Server connection
CONNECTION_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=4.155.146.132,1433;"
    "DATABASE=WideWorldImporters-Full;"
    "UID=ANAND;"
    "PWD=Swathi@123456;"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

class QueryRequest(BaseModel):
    user_query: str

# Extract database schema
def extract_db_schema():
    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
        """)
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        cursor.close()
        conn.close()
        return pd.DataFrame.from_records(rows, columns=columns)
    except Exception as e:
        print("❌ Schema extraction error:", e)
        return pd.DataFrame()

# Format schema into prompt
def build_schema_prompt(schema_df):
    prompt = "You are a SQL Server expert. Use the following schema:\n"
    grouped = schema_df.groupby(['TABLE_SCHEMA', 'TABLE_NAME'])
    for (schema, table), group in grouped:
        prompt += f"\nTable: {schema}.{table}\nColumns:\n"
        for _, row in group.iterrows():
            prompt += f" - {row['COLUMN_NAME']} ({row['DATA_TYPE']})\n"
    return prompt

# Generate SQL only
def generate_sql(user_query, schema_string):
    full_prompt = f"""
You are a SQL Server expert. Based on the schema and user request, return a single valid T-SQL SELECT query. 
⚠️ Do not return any explanation. Just the SQL query. No markdown. No commentary.

Schema:
{schema_string}

User Query: {user_query}

Only output the SQL query (T-SQL for Microsoft SQL Server):
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You generate T-SQL queries without any explanation."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.2
        )
        raw_output = response.choices[0].message.content.strip()

        if "```" in raw_output:
            raw_output = raw_output.replace("```sql", "").replace("```", "").strip()

        if not raw_output.lower().startswith("select"):
            lines = raw_output.splitlines()
            for i, line in enumerate(lines):
                if line.strip().lower().startswith("select"):
                    raw_output = "\n".join(lines[i:])
                    break

        return raw_output
    except Exception as e:
        print("❌ OpenAI API Error:", e)
        return ""

# Execute SQL query
def execute_sql(sql_query):
    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
        return pd.DataFrame.from_records(rows, columns=columns)
    except Exception as e:
        print("❌ SQL Execution Error:", e)
        return pd.DataFrame()

# API Endpoint
@app.post("/ask")
def ask_query(request: QueryRequest):
    try:
        schema_df = extract_db_schema()
        if schema_df.empty:
            return JSONResponse(status_code=500, content={"error": "Failed to load database schema"})

        schema_prompt = build_schema_prompt(schema_df)
        sql = generate_sql(request.user_query, schema_prompt)
        if not sql:
            return JSONResponse(status_code=500, content={"error": "Failed to generate SQL query"})

        result_df = execute_sql(sql)
        if result_df.empty:
            return JSONResponse(status_code=404, content={"error": "No data returned from query"})

        return result_df.to_dict(orient="records")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
