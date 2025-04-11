
import re
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

# Load biến môi trường từ .env

load_dotenv()


SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================
# 1. ĐĂNG KÝ TÀI KHOẢN
# =============================


# def register_user(email, password, full_name):
#     try:
#         # Kiểm tra định dạng email
#         email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
#         if not re.match(email_regex, email):
#             return False, "❌ Email không hợp lệ."

#         # Kiểm tra email đã tồn tại chưa
#         user_list = supabase.auth.admin.list_users()  # Lấy danh sách tất cả người dùng
#         existing_user = next((user for user in user_list.users if user.email == email), None)

#         if existing_user:
#             return False, "⚠️ Email này đã được đăng ký. Vui lòng đăng nhập hoặc sử dụng email khác."

#         # Đăng ký tài khoản
#         res = supabase.auth.sign_up({
#             "email": email,
#             "password": password
#         })

#         if not res.user:
#             return False, "⚠️ Không thể đăng ký tài khoản, vui lòng thử lại."

#         # Tạo hồ sơ người dùng trong bảng user_profiles
#         supabase.table("user_profiles").insert({
#             "id": res.user.id,
#             "full_name": full_name,
#             "role": "client"
#         }).execute()

#         return True, f"✅ Đăng ký thành công! Vui lòng xác minh email: {email}"

#     except Exception as e:
#         return False, f"❌ Lỗi đăng ký: {str(e)}"
import re  # Đảm bảo bạn đã import re cho biểu thức chính quy

def register_user(email, password, full_name):
    try:
        # Kiểm tra định dạng email
        email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        if not re.match(email_regex, email):
            return False, "❌ Email không hợp lệ."

        # Đăng ký tài khoản
        res = supabase.auth.sign_up({
            "email": email,
            "password": password,
            
        })

        if not res.user:
            return False, "⚠️ Không thể đăng ký tài khoản, vui lòng thử lại."
        
        # Tạo hồ sơ người dùng trong bảng user_profiles
        supabase.table("user_profiles").insert({
            "id": res.user.id,
            "email": email,
            "full_name": full_name,
            "role": "client"
        }).execute()

        # Khởi tạo tín dụng cho người dùng mới (0 tín dụng ban đầu)
        supabase.table("user_credits").insert({
            "id": res.user.id,
            "credits": 9
        }).execute()
    
        return True, f"✅ Đăng ký thành công! Vui lòng xác minh email: {email}"

    except Exception as e:
        return False, f"❌ Lỗi đăng ký: {str(e)}"

# =============================
# 2. ĐĂNG NHẬP
# =============================

def login_user(email, password):
    try:

        result = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        user = result.user

        if user.email_confirmed_at is None:
            return False, "📩 Vui lòng xác minh email trước khi đăng nhập."
        
        # Lấy thông tin profile từ user_profiles
        profile_data = supabase.table("user_profiles").select("*").eq("id", user.id).execute()
        # Kiểm tra xem có profile hay chưa
        if profile_data.data:
            user_profile = profile_data.data[0]
            # Lưu thông tin user đầy đủ vào session
            st.session_state["user"] = {
                "id": user.id,
                "email": user.email,
                "full_name": user_profile.get("full_name", ""),
                "role": user_profile.get("role", "client"),
                "created_at": user_profile.get("created_at", "")
            }
        else:
            # Nếu chưa có profile, tạo mới
            new_profile = {
                "id": user.id,
                "email": user.email,
                "full_name": user_profile.get("full_name", ""),
                "role": "client"
            }
            
            
            # Lưu thông tin vào session
            st.session_state["user"] = {
                "id": user.id,
                "email": user.email,
                "full_name": new_profile["full_name"],
                "role": new_profile["role"]
            }

        # Kiểm tra xem người dùng đã có bản ghi credits chưa
        user_credits = supabase.table("user_credits").select("*").eq("id", user.id).execute()
        
        # Nếu chưa có bản ghi credits, tạo mới
        if not user_credits.data:
            supabase.table("user_credits").insert({
                "id": user.id,
                "credits": 9
            }).execute()

        return True, f"🎉 Đăng nhập thành công, xin chào {st.session_state['user']['full_name']}!"

    except Exception as e:
        return False, f"❌ Lỗi đăng nhập: {e}"
