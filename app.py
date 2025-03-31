import os
import bcrypt
import re  # Thêm thư viện kiểm tra email hợp lệ
from openai import OpenAI
import openai
import numpy as np
import streamlit as st
import base64
import pytube
import os
import subprocess 
import librosa
import tempfile 
from pydub import AudioSegment
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import tensorflow as tf
from statistics import mode
from tensorflow import keras
from keras import regularizers
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Activation)
from streamlit_option_menu import option_menu
import time
from dotenv import load_dotenv
from supabase import create_client, Client
import requests  # Dùng để gửi yêu cầu API
from auth import save_song
import asyncio 

st.set_page_config(page_title="Music AI Website", layout="wide")
# Lấy API key từ secrets.toml
openai.api_key = st.secrets["general"]["OPENAI_API_KEY"]

# Khởi tạo client OpenAI
client = openai.api_key

# Kết nối Supabase
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print(os.path.exists("D:/test/Music-Genre-Recognition-main/.streamlit/secrets.toml"))


def generate_lyrics(prompt):
    """Gửi prompt đến OpenAI API để tạo lời bài hát"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Hoặc "gpt-3.5-turbo" nếu tài khoản không có quyền truy cập GPT-4
            messages=[
                {"role": "system", "content": "Bạn là một nhạc sĩ sáng tác lời bài hát chuyên nghiệp."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=900
        )

        # ✅ Lấy nội dung phản hồi đúng cách
        return response.choices[0].message.content  

    except Exception as e:
        return f"⚠️ Lỗi khi tạo lời bài hát: {str(e)}"

# Test thử hàm
#prompt = "Viết lời bài hát về tình yêu mùa thu"
#lyrics = generate_lyrics(prompt)
#print(lyrics)

st.markdown(
    """
    <style>
        /* Đặt hình nền chung cho toàn bộ trang */
        body, .stApp {
            background: url("https://i.pinimg.com/originals/c3/aa/cd/c3aacdb10d1c0d550b7fa08b6d0bddb1.jpg") no-repeat center center fixed;
            background-size: cover;
        }

        /* Sidebar trong suốt, giữ nền đồng nhất */
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(5px);
            border-right: 1px solid rgba(255, 255, 255, 0.2);
        }

        /* Làm mờ nhẹ phần nội dung chính để nổi bật hơn */
        .stApp > div:nth-child(1) {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
        }

        /* Chỉnh màu chữ để dễ đọc trên nền */
        h1, h2, h3, p {
            color: white !important;
        }

        /* Tùy chỉnh nút bấm */
        .stButton>button {
            background: linear-gradient(to right, #ff758c, #ff7eb3);
            color: white;
            font-size: 16px;
            border: none;
            padding: 10px;
            border-radius: 8px;
            transition: 0.3s;
        }

        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(to right, #ff5f6d, #ffc371);
        }

        /* Ô nhập liệu trong suốt */
        .stTextInput>div>div>input {
            background-color: rgba(255, 255, 255, 0.2) !important;
            border-radius: 5px;
            border: 1px solid rgba(255, 255, 255, 0.5) !important;
            padding: 10px !important;
            font-size: 14px !important;
            color: white !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.image("a-minimalist-logo-design-on-a-black-back.jpeg", use_container_width=True)
    # Nếu chưa đăng nhập thì hiển thị menu Đăng ký/Đăng nhập/Quên mật khẩu
    if "user" not in st.session_state:
        auth_menu = st.radio("🔐 Tài khoản", ["Đăng nhập", "Đăng ký", "Quên mật khẩu"], horizontal=True)

        if auth_menu == "Đăng ký":
            st.subheader("✍️ Đăng ký tài khoản")
            full_name = st.text_input("Họ tên")
            email = st.text_input("Email")
            password = st.text_input("Mật khẩu", type="password")
            if st.button("🚀 Đăng ký"):
                from auth import register_user
                success, msg = register_user(email, password, full_name)
                if success:
                    st.success(msg)
                    st.info("📧 Vui lòng kiểm tra hộp thư để xác minh tài khoản trước khi đăng nhập.")
                else:
                    st.error(msg)

        elif auth_menu == "Đăng nhập":
            st.subheader("🔑 Đăng nhập")
            email = st.text_input("Email đăng nhập")
            password = st.text_input("Mật khẩu", type="password")
            if st.button("🔓 Đăng nhập"):
                from auth import login_user
                success, msg = login_user(email, password)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        elif auth_menu == "Quên mật khẩu":
            st.subheader("📧 Đặt lại mật khẩu")
            email = st.text_input("Nhập email đã đăng ký")
            
            if st.button("Gửi email đặt lại mật khẩu"):
                from auth import supabase
                try:
                    res = supabase.auth.reset_password_for_email(email)
                    st.success("📬 Đã gửi email đặt lại mật khẩu. Vui lòng kiểm tra hộp thư đến.")
                except Exception as e:
                    st.error(f"❌ Lỗi khi gửi email: {e}")


    # Hiển thị thông tin user hoặc guest
    if "user" in st.session_state:
        st.markdown(f"👋 Xin chào, **{st.session_state['user']['email']}**")
        st.markdown("📌 Bạn có thể sử dụng toàn bộ chức năng")
        if st.button("🚪 Đăng xuất"):
            st.session_state.clear()
            st.success("✅ Đã đăng xuất.")
            st.rerun()
    else:
        st.markdown("👤 Bạn đang truy cập với tư cách **khách**")
        st.info("👉 Vui lòng đăng nhập để mở khoá các tính năng chính.")

    # Menu chính
    menu = option_menu(
        menu_title="Navigation",
        options=["Home", "Create Lyrics", "Feel The Beat", "Classify", "Explore", "Library", "Search", "Quản lý thanh toán"],  # Thêm "Quản lý thanh toán"
        icons=["house", "music-note-list", "soundwave", "graph-up", "globe", "book", "search", "credit-card"],  # Thêm icon cho "Quản lý thanh toán"
        menu_icon="menu-button-wide",
        default_index=0,
        styles={
            "container": {"background-color": "rgba(0,0,0,0.8)", "padding": "5px"},
            "icon": {"color": "#feb47b", "font-size": "20px"},
            "nav-link": {"font-size": "18px", "color": "#ffffff", "text-align": "left", "margin": "5px"},
            "nav-link-selected": {"background-color": "#ff7e5f"},
        }
    )

# 🚫 Chặn menu nếu chưa đăng nhập
protected_menus = ["Create Lyrics", "Feel The Beat", "Classify", "Explore", "Library", "Quản lý thanh toán"]

if menu in protected_menus and "user" not in st.session_state:
    st.warning("🔒 Vui lòng đăng nhập để truy cập chức năng này.")
    st.stop()





# Nếu chọn "Classify", hiển thị nội dung này
if menu == "Classify":
    st.markdown("<h1 style='text-align: center; color: white;'>Music Genre Recognition</h1>", unsafe_allow_html=True)

    # Upload file mp3
    st.write("## Upload an MP3 file to classify:")
    mp3_file = st.file_uploader("Upload an audio file", type=["mp3"], label_visibility="collapsed")    
    
    if mp3_file is not None:
        st.write("**Play the song below:**")
        st.audio(mp3_file, "audio/mp3")

        # Hàm chuyển đổi MP3 sang WAV
        def convert_mp3_to_wav(music_file):  
            sound = AudioSegment.from_mp3(music_file)
            sound.export("music_file.wav", format="wav")

        # Hàm tạo Mel Spectrogram
        def create_melspectrogram(wav_file):  
            y, sr = librosa.load(wav_file)  
            mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))    
            plt.figure(figsize=(10, 5))
            plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
            librosa.display.specshow(mel_spec, x_axis="time", y_axis='mel', sr=sr)
            plt.margins(0)
            plt.savefig('melspectrogram.png')

        # Xây dựng mô hình CNN
        def GenreModel(input_shape=(100,200,4), classes=10):
            classifier = Sequential()
            classifier.add(Conv2D(8, (3, 3), input_shape=input_shape, activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2, 2)))
            classifier.add(Conv2D(16, (3, 3), activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2, 2)))
            classifier.add(Conv2D(32, (3, 3), activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2, 2)))
            classifier.add(Conv2D(64, (3, 3), activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2, 2)))
            classifier.add(Conv2D(128, (3, 3), activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2, 2)))
            classifier.add(Flatten())
            classifier.add(Dropout(0.5))
            classifier.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
            classifier.add(Dropout(0.25))
            classifier.add(Dense(10, activation='softmax'))
            classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return classifier

        # Dự đoán thể loại nhạc
        def predict(image_data, model):   
            image = img_to_array(image_data)   
            image = np.reshape(image, (1, 100, 200, 4))   
            prediction = model.predict(image / 255)   
            prediction = prediction.reshape((10,))     
            class_label = np.argmax(prediction)     
            return class_label, prediction

        # Nhãn của các thể loại nhạc
        class_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

        # Load mô hình
        model = GenreModel(input_shape=(100, 200, 4), classes=10)
        model.load_weights("music_genre_recog_model.h5")

        # Hiệu ứng loading
        with st.spinner("🔍 Analyzing music genre..."):
            time.sleep(2)

        # Chuyển đổi file và tạo spectrogram
        convert_mp3_to_wav(mp3_file)
        audio_full = AudioSegment.from_wav('music_file.wav')

        class_labels_total = []
        predictions_total = []
        for w in range(int(round(len(audio_full) / 3000, 0))):
            audio_3sec = audio_full[3 * (w) * 1000: 3 * (w + 1) * 1000]
            audio_3sec.export(out_f="audio_3sec.wav", format="wav")
            create_melspectrogram("audio_3sec.wav")
            image_data = load_img('melspectrogram.png', color_mode='rgba', target_size=(100, 200))   
            class_label, prediction = predict(image_data, model)
            class_labels_total.append(class_label)
            predictions_total.append(prediction)

        # Lấy thể loại có dự đoán cao nhất
        class_label_final = mode(class_labels_total)
        predictions_final = np.mean(predictions_total, axis=0)

        # Hiển thị kết quả
        st.success(f"✅ The genre of your song is: **{class_labels[class_label_final]}**")
        # Hiển thị biểu đồ xác suất dự đoán
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(class_labels, predictions_final, color=cm.viridis(np.linspace(0, 1, len(class_labels))))
        ax.set_xlabel("Music Genre")
        ax.set_ylabel("Prediction Probability")
        ax.set_title("Genre Prediction Probability Distribution")
        ax.set_xticklabels(class_labels, rotation=45)
        # Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)




if menu == "Create Lyrics":
    import pyperclip
    st.markdown("<h1>🎶 AI Lyric Generator 🎵</h1>", unsafe_allow_html=True)

    # Người dùng nhập thể loại nhạc và chủ đề
    genre = st.text_input("🎼 Chọn thể loại nhạc: ",
                        placeholder="Pop, Rock, Hip-Hop, Jazz, Ballad, EDM,....")
    mood = st.text_input("🎭 Chọn cảm xúc: ",
                        placeholder="Vui vẻ, Buồn, Hào hứng, Thư giãn, Kịch ,....")
    theme = st.text_input("✍️ Mô tả bản nhạc bạn muốn tạo:",
                        placeholder="Tình yêu, Mùa thu, Tuổi trẻ, ...")
    
    if "lyrics_input" in st.session_state:
        lyrics = st.session_state.lyrics_input
    else:
        lyrics = ""
    
    if st.button("🎤 Sáng tác ngay!"):
        if theme.strip():
            with st.spinner("🎶 AI đang sáng tác lời bài hát cho bạn..."):
                prompt = f"Hãy viết lời bài hát thể loại {genre} về chủ đề '{theme}', với cảm xúc {mood}."
                lyrics = generate_lyrics(prompt)
        else:
            st.warning("⚠️ Vui lòng nhập chủ đề bài hát trước khi tạo!")

    # Hiển thị text_area và lưu giá trị trực tiếp vào lyrics
    lyrics_input = st.text_area("🎼 Lời bài hát AI tạo:", lyrics, height=300)
    # Kiểm tra nếu nội dung text_area thay đổi và tự động sao chép vào clipboard
    
    if st.button("Copy Lyrics"):
            pyperclip.copy(lyrics_input)  # Sao chép lyrics vào clipboard
            lyrics = lyrics_input
            st.session_state.lyrics = lyrics
            st.success("Lyrics have been copied to clipboard and Feel The Beat")  # Hiển thị thông báo thành công

    if lyrics_input != lyrics:
        lyrics = lyrics_input
        st.session_state.lyrics_input = lyrics


import time
import requests
import streamlit as st

# Hàm tạo nhạc từ API
async def generate_music(api_token, prompt, custom_mode, style, title, instrumental):
    api_url = "https://apibox.erweima.ai/api/v1/generate"
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    
    if custom_mode == True:
        data = {
            "prompt": prompt,
            "style": style,
            "title": title,
            "customMode": custom_mode,
            "instrumental": instrumental,
            "model": "V3_5",
            "callBackUrl": "https://api.example.com/callback"
        }
    else:
        data = {
            "prompt": prompt,
            "customMode": custom_mode,
            "instrumental": instrumental,
            "model": "V3_5",
            "callBackUrl": "https://api.example.com/callback"
        }

    with st.spinner("🎼 Đang tạo nhạc..."):
        response = await asyncio.to_thread(requests.post, api_url, json=data, headers=headers)

    # Kiểm tra mã trạng thái của phản hồi từ API
    if response.status_code == 200:
        try:
            response_json = response.json()  # Cố gắng phân tích dữ liệu JSON từ phản hồi

            if response_json and 'data' in response_json:  # Kiểm tra xem 'data' có tồn tại không
                task_id = response_json["data"].get("taskId")
                if task_id:
                    return task_id
                else:
                    st.error("🚨 Không tìm thấy taskId trong phản hồi!")
            else:
                st.error("🚨 Dữ liệu 'data' không tồn tại trong phản hồi API!")
        except ValueError as e:
            st.error(f"🚨 Không thể phân tích JSON từ API: {e}")
            st.write("📄 Nội dung API trả về:", response.text)
    else:
        st.error(f"🚨 API trả về lỗi: {response.status_code}")
        st.write("📄 Nội dung lỗi:", response.text)
    return None

# Hàm kiểm tra và hiển thị nhạc
async def check_music_status(api_token, task_id):
    check_url = f"https://apibox.erweima.ai/api/v1/generate/record-info?taskId={task_id}"
    headers = {"Authorization": f"Bearer {api_token}", "Accept": "application/json"}

    for _ in range(60):  # Lặp tối đa 60 lần (5 phút)
        check_response = await asyncio.to_thread(requests.get, check_url, headers=headers)

        if check_response.status_code == 200:
            try:
                music_info = check_response.json()
                data = music_info.get("data", {})
                status = data.get("status", "PENDING")

                if status == "SUCCESS":
                    suno_data = data.get("response", {}).get("sunoData", [])
                    if suno_data:
                        return [(item.get("audioUrl"), item.get("title"), item.get("imageUrl")) for item in suno_data]
            except ValueError as e:
                st.error(f"🚨 Lỗi khi phân tích JSON từ API: {e}")
                st.write("📄 Nội dung API trả về:", check_response.text)
                break
        else:
            st.error(f"🚨 Lỗi khi kiểm tra nhạc: {check_response.status_code}")
            break
        time.sleep(5)  # Chờ 5 giây trước khi kiểm tra lại
    return None

def render_music_player(title, audio_url, image_url):
    """
    Displays the music player interface with title, cover art and music player.
    """
    st.markdown(
        """
        <style>
            .audio-container {
                text-align: left;
                padding: 20px;
                position: relative;
            }
            audio {
                width: 100%;
                border: 4px solid #ff7e5f;
                border-radius: 30px;
                box-shadow: 0px 0px 15px #feb47b;
            }
            audio::-webkit-media-controls-timeline {
                background: linear-gradient(90deg, #ff7e5f, #feb47b) !important;
                border-radius: 30px;
                height: 6px;
                box-shadow: 0px 0px 10px rgba(255, 126, 95, 0.8);
                transition: all 0.3s ease-in-out;
                padding: 1px;
            }
            audio::-webkit-media-controls-play-button {
                background-color: #ff7e5f !important;
                box-shadow: 0px 0px 10px rgba(255, 126, 95, 0.8);
                border-radius: 50%;
            }
            audio::-webkit-media-controls-volume-slider {
                background: linear-gradient(90deg, #ff7e5f, #feb47b) !important;
                border-radius: 30px;
                height: 6px;
                box-shadow: 0px 0px 10px rgba(255, 126, 95, 0.8);
                transition: all 0.3s ease-in-out;
                margin-top: 11px;
                padding-top:1px;
                padding-bottom:1px;
            }
            .song-title {
                font-size: 20px;
                font-weight: bold;
                color: white;
                text-align: left;
                margin-top: 10px;
                text-shadow: 0px 0px 10px rgba(255, 126, 95, 0.8);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(image_url, width=150)
    with col2:
        st.markdown(f'<div class="song-title">{title}</div>', unsafe_allow_html=True)
        st.audio(audio_url, format="audio/mp3")


# Hàm hiển thị trò chơi chờ nhạc
def render_game_html():
    game_html = """
    <iframe src="https://chromedino.com/color/" frameborder="0" scrolling="no" width="100%" height="100%" loading="lazy"></iframe>
    <div style="
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #0E1117; /* Màu nền */
        color: #FFA500; /* Màu chữ cam */
        font-size: 24px; /* Kích thước chữ */
        font-weight: bold; /* Đậm chữ */
        z-index: 102;
        display: flex; /* Căn giữa */
        align-items: center; /* Căn giữa theo chiều dọc */
        justify-content: center; /* Căn giữa theo chiều ngang */
        white-space: pre-line; /* Giữ nguyên xuống dòng */
        flex-direction: column; /* Xếp nội dung theo chiều dọc */
        text-align: center; /* Căn giữa chữ */
    ">
        <div>
        🔥 Survive until the music is over 🔥
        </div>
        <p style="font-size: 16px; font-weight: normal;">
            You can play Running Dinosaur while waiting for the music (up to 5 minutes).  
            Press Space to start the game online and jump your Dino, use down arrow (↓) to duck.
        </p>
    </div>
    
    <style type="text/css">
    iframe { 
        margin-top: 20px;
        position: absolute; 
        width: 100%; 
        height: 100%; 
        z-index: 100; 
    }
    </style>
    """
    st.components.v1.html(game_html, height=320)


# Phần chính của ứng dụng
async def Feel_The_Beat():
    st.title("🎵 Feel The Beat - Tạo Nhạc AI")

    api_token = "2d551602f3a39d8f3e219db2c94d7659"
    custom_mode = st.toggle("Custom Mode", value=True)
    if "lyrics" in st.session_state:
        lyrics = st.session_state.lyrics
        prompt = st.text_area("💡 Enter a description of the track you want to create:", 
                              value=lyrics, 
                              placeholder="A relaxing piano piece with a gentle melody...")
    else:
        prompt = st.text_area("💡 Enter a description of the track you want to create:", 
                              placeholder="A relaxing piano piece with a gentle melody...")

    if custom_mode == True:
        # Danh sách gợi ý phong cách nhạc
        music_styles = ["Classical", "Jazz", "Lo-fi", "Ambient", "Rock"]

        # Nếu chưa có session_state cho style_list, đặt giá trị mặc định
        if "style_list" not in st.session_state:
            st.session_state["style_list"] = []

        # Hộp nhập phong cách nhạc (hiển thị danh sách dưới dạng chuỗi)
        style = st.text_input("🎼 Enter music style:", ", ".join(st.session_state["style_list"]))

        # Hiển thị các nút theo hàng ngang
        cols = st.columns(len(music_styles))

        for i, music in enumerate(music_styles):
            with cols[i]:
                if st.button(music, use_container_width=True):
                    if music in st.session_state["style_list"]:
                        # Nếu đã có trong danh sách thì xóa đi (bỏ chọn)
                        st.session_state["style_list"].remove(music)
                    else:
                        # Nếu chưa có thì thêm vào danh sách
                        st.session_state["style_list"].append(music)
                    
                    # Cập nhật text box với danh sách mới
                    st.rerun()  # Cập nhật giao diện ngay lập tức

        title = st.text_input("🎶 Name the song:", "My AI Music")
        instrumental = st.checkbox("🎻 Instrumental", value=False)
    # Xóa music_data khi người dùng bấm nút
    if st.button("🎧 Feel The Beat"):
        if "music_data" in st.session_state:
            del st.session_state["music_data"]  # Xóa music_data trong session_state trước khi tạo nhạc

        if not api_token or not prompt:
            st.warning("⚠️Please enter music description!")
        else:
            task_id = await generate_music(api_token, prompt, custom_mode, "", "", False)
            if task_id:
                render_game_html()  # Hiển thị trò chơi chờ nhạc
                
                music_data = await check_music_status(api_token, task_id)
                
                if music_data:
                    st.session_state["music_data"] = music_data  # Lưu nhạc vào session_state
                    for audio_url, title, image_url in music_data:
                        st.success(f"🎵 Your music is ready: [{title}]")
                        render_music_player(title, audio_url, image_url)
                else:
                    st.warning("⏳ Music not ready after 5 minutes, please try again later!")
            else:
                st.error("🚨 Error in music generation!")

    # Kiểm tra nếu có nhạc đã tạo trong session_state
    if "music_data" in st.session_state:
        music_data = st.session_state["music_data"]
        for audio_url, title, image_url in music_data:
            st.success(f"🎵 Your music is ready: [{title}]")
            render_music_player(title, audio_url, image_url)
if menu == "Feel The Beat":
    asyncio.run(Feel_The_Beat())

# =========================== QUẢN LÝ THANH TOÁN ===========================
if menu == "Quản lý thanh toán":
    st.title("💳 Quản lý thanh toán")

    # Phần 1: Hiển thị số dư tài khoản
    if "user" in st.session_state:
        user_id = st.session_state.user['id']
        wallet = supabase.table("credits_wallet").select("credit").eq("user_id", user_id).execute()
        credit_balance = wallet.data[0]["credit"] if wallet.data else 0
    else:
        st.error("❌ Bạn cần đăng nhập để quản lý thanh toán.")
        st.stop()

    st.subheader("Thông tin số dư tài khoản")
    st.write(f"💰 Số tín dụng còn lại: **{credit_balance} tín dụng**")

    # Phần 2: Mua tín dụng
    credit_options = {
        "1000 tín dụng - 5$": (1000, 5),
        "10000 tín dụng - 50$": (10000, 50),
        "105000 tín dụng - 500$": (105000, 500),
        "275000 tín dụng - 1250$": (275000, 1250)
    }
    selected_option = st.selectbox("Chọn gói tín dụng muốn mua:", list(credit_options.keys()))
    selected_credits, amount = credit_options[selected_option]

    def get_usd_to_vnd_exchange_rate():
        """Lấy tỷ giá USD-VND từ API Layer (Live Exchange Rates)."""
        try:
            api_key = "qf4h6PVtQlWfqPBrQEgStY3eHeEuk88E"  # API key của bạn
            url = "https://api.apilayer.com/currency_data/live"
            params = {"source": "USD", "currencies": "VND"}
            headers = {"apikey": api_key}

            # Gửi yêu cầu đến API
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                exchange_rate = data["quotes"]["USDVND"]
                st.write(f"Tỷ giá USD-VND: {exchange_rate}")  # Hiển thị tỷ giá trên giao diện Streamlit
                return exchange_rate
            else:
                st.error("Không thể lấy tỷ giá USD-VND. Sử dụng tỷ giá mặc định.")
                return 23500  # Tỷ giá mặc định nếu API không hoạt động
        except Exception as e:
            st.error(f"Lỗi khi lấy tỷ giá: {e}")
            return 23500  # Tỷ giá mặc định nếu có lỗi

    if st.button("Thanh toán qua MoMo"):
        # Save session state before redirecting
        st.session_state["redirect_after_payment"] = True
        st.session_state["payment_in_progress"] = True  # Track payment state

        momo_endpoint = "https://test-payment.momo.vn/v2/gateway/api/create"
        momo_partner_code = "MOMO"
        momo_access_key = "F8BBA842ECF85"
        momo_secret_key = "K951B6PE1waDMi640xX08PD3vg6EkVlz"
        order_id = f"order_{int(time.time())}"
        redirect_url = "https://aimusic-kg7fjzh3yp5cvrncwxfhnf.streamlit.app/"
        ipn_url = "https://aimusic-kg7fjzh3yp5cvrncwxfhnf.streamlit.app/"
        request_id = f"req_{int(time.time())}"
        order_info = f"Mua {selected_credits} tín dụng"

        # Lấy tỷ giá USD-VND từ API Layer
        exchange_rate = get_usd_to_vnd_exchange_rate()
        amount_vnd = amount * exchange_rate  # Chuyển đổi từ USD sang VND
        amount_str = str(int(amount_vnd))  # Đảm bảo số tiền là số nguyên

        # Hiển thị giá trị đã chuyển đổi trên giao diện Streamlit
        st.write(f"Số tiền thanh toán (USD): {amount} USD")
        st.write(f"Số tiền thanh toán (VND): {amount_vnd} VND")
        st.write(f"Số tiền gửi đến MoMo: {amount_str} VND")

        # Tạo chữ ký (signature)
        raw_signature = f"accessKey={momo_access_key}&amount={amount_str}&extraData=&ipnUrl={ipn_url}&orderId={order_id}&orderInfo={order_info}&partnerCode={momo_partner_code}&redirectUrl={redirect_url}&requestId={request_id}&requestType=captureWallet"
        import hmac, hashlib
        signature = hmac.new(momo_secret_key.encode(), raw_signature.encode(), hashlib.sha256).hexdigest()

        # Tạo yêu cầu thanh toán
        payload = {
            "partnerCode": momo_partner_code,
            "accessKey": momo_access_key,
            "requestId": request_id,
            "amount": amount_str,  # Sử dụng số tiền đã chuyển đổi sang VND
            "orderId": order_id,
            "orderInfo": order_info,
            "redirectUrl": redirect_url,
            "ipnUrl": ipn_url,
            "extraData": "",
            "requestType": "captureWallet",
            "signature": signature
        }
        response = requests.post(momo_endpoint, json=payload)


        if response.status_code == 200:
            payment_url = response.json().get("payUrl")
            if payment_url:
                # Tự động chuyển hướng đến cổng thanh toán MoMo
                st.markdown(f"<meta http-equiv='refresh' content='0; url={payment_url}'>", unsafe_allow_html=True)
                st.stop()

    # Check if returning from payment
    if st.session_state.get("redirect_after_payment"):
        st.session_state["redirect_after_payment"] = False  # Reset flag
        with st.spinner("⏳ Đang kiểm tra trạng thái thanh toán..."):
            time.sleep(5)  # Wait for payment to process
            payment_status = supabase.table("transactions").select("status").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
            if payment_status.data and payment_status.data[0]["status"] == "success":
                # Update credit balance
                new_balance = credit_balance + selected_credits
                supabase.table("credits_wallet").update({"credit": new_balance}).eq("user_id", user_id).execute()
                supabase.table("credits_history").insert({
                    "user_id": user_id,
                    "action": "add",
                    "amount": selected_credits,
                    "note": f"Mua {selected_credits} tín dụng qua MoMo"
                }).execute()
                st.session_state["credit_balance"] = new_balance  # Update session state
                st.success(f"✅ Thanh toán thành công! Số dư hiện tại: {new_balance} tín dụng.")
                st.experimental_rerun()  # Redirect to refresh the page
            else:
                st.error("❌ Thanh toán chưa hoàn tất hoặc thất bại. Vui lòng thử lại.")
                st.experimental_rerun()  # Redirect to refresh the page

    # Prevent infinite redirects by ensuring the app does not loop back to the payment check
    if not st.session_state.get("payment_in_progress", False):
        st.session_state["payment_in_progress"] = False  # Ensure the flag is reset

# =========================== KIỂM TRA SỬ DỤNG MIỄN PHÍ ===========================
if menu == "Feel The Beat":
    # Kiểm tra số lần sử dụng miễn phí
    if "free_uses" not in st.session_state:
        st.session_state.free_uses = 3  # Mặc định 3 lần miễn phí

    if st.session_state.free_uses > 0:
        st.info(f"Bạn còn {st.session_state.free_uses} lần sử dụng miễn phí.")
        st.session_state.free_uses -= 1
    else:
        user_id = st.session_state.user['id']
        wallet = supabase.table("credits_wallet").select("credit").eq("user_id", user_id).execute()
        credit_balance = wallet.data[0]["credit"] if wallet.data else 0

        if credit_balance >= 25:
            new_balance = credit_balance - 25
            supabase.table("credits_wallet").update({"credit": new_balance}).eq("user_id", user_id).execute()
            supabase.table("credits_history").insert({
                "user_id": user_id,
                "action": "deduct",
                "amount": 25,
                "note": "Sử dụng tính năng Feel The Beat"
            }).execute()
            supabase.table("transactions").insert({
                "user_id": user_id,
                "credits_used": 25,
                "payment_method": "Usage Fee",
                "note": "Sử dụng tính năng Feel The Beat"
            }).execute()
            st.success(f"Đã trừ 25 tín dụng. Số dư còn lại: {new_balance} tín dụng.")
        else:
            st.error("Không đủ tín dụng để sử dụng tính năng này.")
            st.stop()
