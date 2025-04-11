# Writing the corrected HTML code into a Python file properly formatted.


import streamlit as st
from streamlit.components.v1 import html
import asyncio
from auth import register_user, login_user, supabase  # Import from auth.py

# HTML content for the layout
def render_html():
    page_content = '''
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SUNO Music</title>
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                background-color: #121212;
                color: #fff;
                margin: 0;
                padding: 0;
                line-height: 1.6;
            }
            header {
                background: #1F1F1F;
                padding: 20px 0;
                position: sticky;
                top: 0;
                z-index: 1000;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            .logo {
                font-size: 30px;
                font-weight: bold;
                color: #F7A200;
                text-transform: uppercase;
                margin-left: 30px;
                display: inline-block;
            }
            header nav ul {
                display: flex;
                justify-content: flex-start;
                list-style-type: none;
                margin: 20px 0 0 0;
                padding-left: 0;
            }
            header nav ul li {
                margin-right: 30px;
            }
            header nav ul li a {
                color: #fff;
                font-size: 16px;
                text-decoration: none;
                text-transform: uppercase;
                transition: color 0.3s ease;
            }
            header nav ul li a:hover {
                color: #F7A200;
            }
            main {
                display: flex;
                justify-content: space-between;
                padding: 30px 20px;
                max-width: 1200px;
                margin: auto;
            }
            .sidebar {
                width: 250px;
                background: #1F1F1F;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            .sidebar ul {
                list-style: none;
                padding-left: 0;
            }
            .sidebar ul li {
                margin: 15px 0;
            }
            .sidebar ul li a {
                color: #fff;
                font-size: 18px;
                text-decoration: none;
                transition: color 0.3s ease;
            }
            .sidebar ul li a:hover {
                color: #F7A200;
            }
            .main-content {
                width: 100%;
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
            }
            .content-card {
                background: #242424;
                border-radius: 10px;
                padding: 20px;
                width: calc(50% - 20px);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                position: relative;
                overflow: hidden;
                transition: transform 0.3s ease;
            }
            .content-card:hover {
                transform: scale(1.05);
            }
            .content-card img {
                width: 100%;
                border-radius: 10px;
                height: 200px;
                object-fit: cover;
            }
            .content-card .content-info {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                text-align: center;
                color: #fff;
            }
            .content-card .content-info h3 {
                font-size: 24px;
                margin: 10px 0;
            }
            .content-card .content-info p {
                color: #ccc;
                font-size: 16px;
            }
            .cta-button {
                background-color: #F7A200;
                padding: 12px 25px;
                border-radius: 30px;
                border: none;
                color: white;
                cursor: pointer;
                font-size: 18px;
                text-transform: uppercase;
                transition: background-color 0.3s ease;
                display: inline-block;
                margin-top: 10px;
            }
            .cta-button:hover {
                background-color: #FF8A00;
            }
            footer {
                text-align: center;
                padding: 20px;
                background-color: #1F1F1F;
                color: white;
                position: relative;
                bottom: 0;
                width: 100%;
            }
        </style>
    </head>
    <body>
        <header>
            <div class="logo">SUNO</div>
            <nav>
                <ul>
                    <li><a href="#home">Home</a></li>
                    <li><a href="#create">Create</a></li>
                    <li><a href="#library">Library</a></li>
                    <li><a href="#explore">Explore</a></li>
                    <li><a href="#search">Search</a></li>
                </ul>
            </nav>
        </header>

        <main>
            <div class="sidebar">
                <ul>
                    <li><a href="#home">Dashboard</a></li>
                    <li><a href="#create">Create Song</a></li>
                    <li><a href="#library">My Library</a></li>
                    <li><a href="#explore">Explore Music</a></li>
                    <li><a href="#search">Search</a></li>
                </ul>
            </div>
            <div class="main-content">
                <div class="content-card">
                    <img src="https://via.placeholder.com/600x400" alt="Featured Music">
                    <div class="content-info">
                        <h3>Global ✨ POP</h3>
                        <p>Modern sounds from different cultures, a mix of pop sounds, trap, and lyrical rap.</p>
                        <button class="cta-button">Play</button>
                    </div>
                </div>
            </div>
        </main>

        <footer>
            <p>&copy; 2025 SUNO Music. All Rights Reserved.</p>
        </footer>
    </body>
    </html>
    '''
    return page_content


# Streamlit Page Setup
def main():
    st.set_page_config(page_title="SUNO Music", layout="wide")
    
    # Render HTML content in Streamlit
    html(render_html(), height=900)

    # Authentication: Sign Up and Sign In
    with st.sidebar:
        st.image("your-logo.jpg", use_container_width=True)
        
        # User authentication flow
        if "user" not in st.session_state:
            auth_menu = st.radio("Account", ["Sign In", "Sign Up"], horizontal=True)

            if auth_menu == "Sign Up":
                st.subheader("Sign Up")
                full_name = st.text_input("Full Name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.button("Sign Up"):
                    success, msg = register_user(email, password, full_name)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

            elif auth_menu == "Sign In":
                st.subheader("Sign In")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.button("Sign In"):
                    success, msg = login_user(email, password)
                    if success:
                        st.success(msg)
                        st.session_state.user = {"email": email}
                        st.rerun()
                    else:
                        st.error(msg)

if __name__ == "__main__":
    main()

