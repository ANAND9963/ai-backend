import json
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
if not OPENAI_API_KEY:
    print("❌ OpenAI API Key missing in .env")
    exit()
client = OpenAI(api_key=OPENAI_API_KEY)

# Define FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQL Server connection string
CONNECTION_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost,1433;"
    "DATABASE=WideWorldImporters;"
    "UID=dbuser01;"
    "PWD=dbuser01;"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

# Request model
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

# Format schema into a prompt
def build_schema_prompt(schema_df):
    prompt = "You are a SQL Server expert. Use the following schema:\n"
    grouped = schema_df.groupby(['TABLE_SCHEMA', 'TABLE_NAME'])
    for (schema, table), group in grouped:
        prompt += f"\nTable: {schema}.{table}\nColumns:\n"
        for _, row in group.iterrows():
            prompt += f" - {row['COLUMN_NAME']} ({row['DATA_TYPE']})\n"
    return prompt

# Generate SQL query from prompt
def generate_sql(user_query, schema_string):
    full_prompt = f"""
You are a SQL Server expert. Based on the schema and user request, return a single valid T-SQL SELECT query.
If any table contains geography/geometry types, exclude those columns in your SELECT query.
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

        # Clean markdown formatting
        if "```" in raw_output:
            raw_output = raw_output.replace("```sql", "").replace("```", "").strip()

        if not raw_output.lower().startswith("select"):
            lines = raw_output.splitlines()
            for i, line in enumerate(lines):
                if line.strip().lower().startswith("select"):
                    raw_output = "\n".join(lines[i:])
                    break

        print("\n✅ Cleaned SQL Query:\n", raw_output)
        return raw_output

    except Exception as e:
        print("❌ OpenAI API Error:", e)
        return ""

# Execute SQL query and filter unsupported types
def execute_sql(sql_query):
    if not sql_query:
        print("⚠️ No valid SQL query.")
        return pd.DataFrame()

    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(sql_query)

        if sql_query.strip().lower().startswith("select"):
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            types = [desc[1] for desc in cursor.description]

            # Skip unsupported types like geography, geometry, sql_variant
            unsupported_sql_types = {-151, -150, -16, -11}
            valid_indices = [i for i, t in enumerate(types) if t not in unsupported_sql_types]
            valid_columns = [columns[i] for i in valid_indices]

            if valid_columns:
                filtered_data = [
                    {columns[i]: row[i] for i in valid_indices}
                    for row in rows
                ]
                df = pd.DataFrame(filtered_data)
                return df
            else:
                print("⚠️ No supported columns to display.")
                return pd.DataFrame()
        else:
            conn.commit()
            print("✅ Non-select query executed.")
            return pd.DataFrame()

    except Exception as e:
        print("❌ SQL Execution Error:", e)
        return pd.DataFrame()

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
