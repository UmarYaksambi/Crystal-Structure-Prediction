import os
from supabase import create_client, Client
import streamlit as st
# Initialize Supabase client
def init_supabase():
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    return create_client(url, key)

# Insert data into Supabase
def insert_data(client: Client, table_name: str, data: dict):
    response = client.table(table_name).insert(data).execute()
    return response

# Fetch data from Supabase
def fetch_data(client: Client, table_name: str):
    response = client.table(table_name).select("*").execute()
    return response


