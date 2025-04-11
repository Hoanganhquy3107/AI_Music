import streamlit as st
from supabase import create_client

# 🟢 Đọc thông tin từ secrets.toml (chỉ hoạt động nếu file đã được thiết lập)
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# 🟢 Kết nối với Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🛠 Kiểm tra kết nối
st.write("🔍 Supabase đã kết nối thành công!")
