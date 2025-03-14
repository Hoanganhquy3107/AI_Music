import streamlit as st

supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]

print("✅ SUPABASE_URL:", supabase_url)
print("✅ SUPABASE_KEY:", supabase_key[:10] + "...")
