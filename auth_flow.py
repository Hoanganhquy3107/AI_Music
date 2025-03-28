import time
import streamlit as st
from supabase import Client

def rerun():
    st.rerun()

def create_or_fetch_profile(supabase: Client, user_id: str, email: str):
    """Tạo hồ sơ nếu chưa có trong bảng profiles"""
    profile = supabase.table("profiles").select("*").eq("id", user_id).execute().data
    if not profile:
        supabase.table("profiles").insert({
            "id": user_id,
            "nickname": email.split("@")[0],
            "avatar_url": "a-minimalist-logo-design-on-a-black-back.jpeg"
        }).execute()
        time.sleep(0.5)
        profile = supabase.table("profiles").select("*").eq("id", user_id).execute().data
    return profile[0]

def handle_login(supabase: Client, email: str, password: str):
    try:
        auth = supabase.auth.sign_in_with_password({"email": email, "password": password})

        st.session_state.user = auth.user
        st.session_state.user_profile = create_or_fetch_profile(supabase, auth.user.id, email)
        st.session_state.show_login = False
        rerun()
    except Exception as e:
        st.error(f"❌ Lỗi: {str(e)}")

def handle_register(supabase: Client, email: str, password: str):
    try:
        sign_up = supabase.auth.sign_up({"email": email, "password": password})
        supabase.table("profiles").insert({
            "id": sign_up.user.id,
            "nickname": email.split("@")[0],
            "avatar_url": "a-minimalist-logo-design-on-a-black-back.jpeg"
        }).execute()

        st.success("✅ Vui lòng kiểm tra email để xác minh trước khi đăng nhập.")
        time.sleep(2)
        st.session_state.show_login = False
        rerun()
    except Exception as e:
        st.error(f"❌ Lỗi khi đăng ký tài khoản: {str(e)}")


def handle_email_verification():
    """Kiểm tra xác minh email (nếu cần)"""
    if st.session_state.user and not getattr(st.session_state.user, "email_verified", True):
        st.warning("⚠️ Bạn cần xác minh email để tiếp tục.")
        st.stop()
