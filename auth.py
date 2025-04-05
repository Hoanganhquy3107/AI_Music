
# auth.py

import os
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv
import bcrypt

# Load biến môi trường
load_dotenv()

# Kết nối Supabase
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================
# 1. HÀM ĐĂNG KÝ NGƯỜI DÙNG (Sign Up)
# ============================================
import re  # Thêm trên đầu file nếu chưa có

def register_user(email, password, full_name):
    try:
        # Mã hóa mật khẩu
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Thêm người dùng vào bảng auth.users
        user_data = supabase.table("auth.users").insert({
            "email": email,
            "password": hashed_password,
            "full_name": full_name
        }).execute()

        # Lấy ID của người dùng vừa tạo
        user_id = user_data.data[0]["id"]

        # Thêm thông tin vào bảng user_profiles
        supabase.table("user_profiles").insert({
            "id": user_id,  # ID khớp với auth.users
            "full_name": full_name,
            "role": "user"  # Mặc định vai trò là "user"
        }).execute()

        return True, "✅ Đăng ký thành công!"
    except Exception as e:
        return False, f"❌ Lỗi khi đăng ký: {str(e)}"




# ============================================
# 2. HÀM ĐĂNG NHẬP (Sign In)
# ============================================
def login_user(email, password):
    try:
        # Gửi yêu cầu đăng nhập
        result = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        user = result.user
        session = result.session

        if not session:
            return False, "❌ Sai email hoặc mật khẩu."

        if user.email_confirmed_at is None:
            return False, "📩 Vui lòng xác minh email trước khi đăng nhập."

        # Lưu vào session_state của Streamlit
        st.session_state["user"] = {
            "id": user.id,
            "email": user.email
        }

        # ============================================
        # TẠO PROFILE VÀ VÍ CREDITS nếu chưa tồn tại
        # ============================================
        # Kiểm tra user_profiles
        profile_check = supabase.table("user_profiles").select("id").eq("id", user.id).execute()
        if not profile_check.data:
            supabase.table("user_profiles").insert({
                "id": user.id,
                "full_name": user.email.split("@")[0],
                "role": "client"
            }).execute()

            supabase.table("credits_wallet").insert({
                "user_id": user.id,
                "credit": 10000000
            }).execute()

        return True, f"🎉 Xin chào {user.email}!"

    except Exception as e:
        return False, f"❌ Lỗi đăng nhập: {e}"

# ============================================
# 3. EXPORT LẠI SUPABASE ĐỂ DÙNG RESET PASSWORD
# ============================================
# Trong app.py sẽ gọi: from auth import supabase
def get_user_credit(user_id):
    res = supabase.table("credits_wallet").select("credit").eq("user_id", user_id).execute()
    if res.data:
        return res.data[0]['credit']
    return 0

def deduct_credit(user_id, amount):
    current = get_user_credit(user_id)
    if current < amount:
        return False, f"❌ Bạn không đủ credit (hiện tại: {current}). Vui lòng nạp thêm!"
    
    supabase.table("credits_wallet").update({
        "credit": current - amount
    }).eq("user_id", user_id).execute()

    # Ghi vào lịch sử sử dụng
    supabase.table("credits_history").insert({
        "user_id": user_id,
        "action": "use",
        "amount": amount,
        "note": "Tạo bài hát Feel The Beat"
    }).execute()

    return True, "✅ Đã trừ credit"
def save_song(user_id, title, lyrics, genre, audio_url, style, instruments, is_public=False):
    try:
        supabase.table("songs").insert({
            "user_id": user_id,
            "title": title,
            "lyrics": lyrics,
            "genre": genre,
            "audio_url": audio_url,
            "style": style,
            "instruments": instruments,
            "is_public": is_public
        }).execute()
        return True
    except Exception as e:
        print("Lỗi lưu bài hát:", e)
        return False

def sync_user_profiles():
    try:
        # Lấy tất cả người dùng từ bảng auth.users
        users = supabase.table("auth.users").select("id", "full_name").execute()

        for user in users.data:
            user_id = user["id"]
            full_name = user["full_name"]

            # Kiểm tra nếu người dùng đã tồn tại trong bảng user_profiles
            profile_data = supabase.table("user_profiles").select("id").eq("id", user_id).execute()
            if not profile_data.data or len(profile_data.data) == 0:
                # Thêm mới vào bảng user_profiles
                supabase.table("user_profiles").insert({
                    "id": user_id,
                    "full_name": full_name,
                    "role": "user"  # Mặc định vai trò là "user"
                }).execute()

        print("✅ Đồng bộ dữ liệu thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi đồng bộ dữ liệu: {str(e)}")
