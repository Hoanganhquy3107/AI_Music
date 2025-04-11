import streamlit as st
st.set_page_config(page_title="Music AI Website", layout="wide")
import os
import bcrypt
import re  # Thêm thư viện kiểm tra email hợp lệ
from openai import OpenAI
import numpy as np
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
import asyncio 
import streamlit.components.v1 as components    
from auth import register_user
import streamlit as st
from streamlit_cookies_manager import CookieManager
import base64
import logging


# Load API key từ file .env
load_dotenv()
client = st.secrets.get("OPENAI_API_KEY")
if not client:
    client = os.getenv("OPENAI_API_KEY")

# Kết nối Supabase
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print(os.path.exists("D:/test/Music-Genre-Recognition-main/.streamlit/secrets.toml"))

# Cấu hình logging - Lưu các lỗi vào file 'app.log'
logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s - %(message)s')

# Hàm ghi lỗi vào log
def log_error(message):
    """Ghi lỗi vào file log và hiển thị thông báo lỗi cho người dùng."""
    logging.error(message)  # Ghi lỗi vào file log
    st.error(f"🚨 Lỗi xảy ra: {message}")  # Hiển thị lỗi cho người dùng

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
import streamlit as st
from streamlit_cookies_manager import CookieManager
import base64

# Hàm mã hóa email
def encode_email(email):
    return base64.b64encode(email.encode()).decode()

# Hàm giải mã email
def decode_email(encoded):
    try:
        return base64.b64decode(encoded.encode()).decode()
    except Exception:
        return None

with st.sidebar:
    st.image("a-minimalist-logo-design-on-a-black-back.jpeg", use_container_width=True)

    cookies = CookieManager()

    # Kiểm tra cookies có sẵn và đã mã hóa email
    # if cookies.ready() and cookies.get("user_email"):
    #     decoded_email = decode_email(cookies.get("user_email"))
    #     if decoded_email:
    #         st.session_state['user'] = {'email': decoded_email}
    if cookies.ready() and cookies.get("user_email") and "user" not in st.session_state:
        decoded_email = decode_email(cookies.get("user_email"))
        if decoded_email:
            # 👉 Gọi Supabase để lấy thông tin đầy đủ từ email
            profile_data = supabase.table("user_profiles").select("*").eq("email", decoded_email).execute()
            if profile_data.data:
                profile = profile_data.data[0]
                st.session_state["user"] = {
                    "id": profile["id"],
                    "email": profile["email"],
                    "full_name": profile.get("full_name", ""),
                    "role": profile.get("role", "client"),
                    "created_at": profile.get("created_at", "")
                }

                # # Kiểm tra credits
                # credits_data = supabase.table("user_credits").select("*").eq("id", profile["id"]).execute()
                # if not credits_data.data:
                #     supabase.table("user_credits").insert({ "id": profile["id"], "credits": 0 }).execute()

    if "user" not in st.session_state:
        auth_menu = st.radio("🔐 Tài khoản", ["Đăng nhập", "Đăng ký", "Quên mật khẩu"], horizontal=True)
        if auth_menu == "Đăng ký":
            st.subheader("✍️ Đăng ký tài khoản")
            
            email = st.text_input("Email", type="default")
            password = st.text_input("Mật khẩu", type="password")
            full_name = st.text_input("Họ tên")
            # st.session_state['user']['full_name'] = full_name
            if st.button("🚀 Đăng ký"):
                from auth import register_user
                success, msg = register_user(email, password, full_name)
                if success:
                    #st.session_state['user'] = {'email': email}
                    # cookies["user_email"] = encode_email(email)
                    # cookies.save()
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
                    # st.session_state['user'] = {'email': email}
                    cookies["user_email"] = encode_email(email)
                    cookies["user_id"] = st.session_state["user"]["id"]
                    cookies.save()
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

    if "user" in st.session_state:
        st.markdown(f"👋 Xin chào")
        st.markdown("📌 Bạn có thể sử dụng toàn bộ chức năng")
        if st.button("🚪 Đăng xuất"):
            del cookies["user_email"]
            del st.session_state['user']
            cookies.save()
            st.success("✅ Đã đăng xuất.")
            st.rerun()
    else:
        st.markdown("👤 Bạn đang truy cập với tư cách **khách**")
        st.info("👉 Vui lòng đăng nhập để mở khoá các tính năng chính.")


    # Menu chính
    menu = option_menu(
        menu_title="Navigation",
        options=["Home", "Create Lyrics", "Feel The Beat", "Classify", "Explore", "Library", "Search", "Quản lý thanh toán"],
        icons=["house", "music-note-list", "soundwave", "graph-up", "globe", "book", "search", "credit-card"],
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
protected_menus = ["Create Lyrics", "Feel The Beat", "Classify", "Explore", "Library","Quản lý thanh toán"]

if menu in protected_menus and "user" not in st.session_state:
    st.warning("🔒 Vui lòng đăng nhập để truy cập chức năng này.")
    st.stop()

def handle_empty_title(music_data):
    """Kiểm tra và điền tên bài hát nếu bị rỗng."""
    for song in music_data:
        if isinstance(song, dict):  # Kiểm tra xem song có phải là dictionary không
            # Kiểm tra nếu thiếu audioUrl hoặc imageUrl
            if not song.get('audioUrl'):
                song['audioUrl'] = "https://default-audio-url.com"  # Đặt URL mặc định nếu thiếu audioUrl
            if not song.get('imageUrl'):
                song['imageUrl'] = "https://default-image-url.com"  # Đặt URL mặc định nếu thiếu imageUrl

            # Kiểm tra nếu thiếu title
            if not song.get('title'):
                song['title'] = f"Track {song.get('id', 'Unknown')}"  # Đặt tên mặc định nếu không có title
                log_error(f"Bài hát với ID {song.get('id', 'Unknown')} thiếu title. Đặt tên mặc định.")
        else:
            log_error(f"Dữ liệu bài hát không hợp lệ: {song}")
    return music_data






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

            # Kiểm tra nếu 'data' có tồn tại trong phản hồi
            data = response_json.get("data")  # Sử dụng .get() để tránh lỗi nếu 'data' không tồn tại

            if data is not None:
                task_id = data.get("taskId")  # Lấy taskId từ 'data'
                if task_id:
                    return task_id
                else:
                    st.error("🚨 Không tìm thấy taskId trong phản hồi!")
            else:
                st.error("🚨 Không có dữ liệu 'data' trong phản hồi API!")
                st.write("📄 Nội dung API trả về:", response.text)
        except ValueError as e:
            st.error(f"🚨 Lỗi khi phân tích JSON từ API: {e}")
            st.write("📄 Nội dung API trả về:", response.text)
    else:
        st.error(f"🚨 API trả về lỗi: {response.status_code}")
        st.write("📄 Nội dung lỗi:", response.text)
    return None
    # Kiểm tra mã trạng thái của phản hồi từ API
    # if response.status_code == 200:
    #     try:
    #         response_json = response.json()  # Cố gắng phân tích dữ liệu JSON từ phản hồi

    #         # Kiểm tra nếu 'data' có tồn tại trong phản hồi
    #         data = response_json.get("data")

    #         if response_json and 'data' in response_json:  # Kiểm tra xem 'data' có tồn tại không
    #             task_id = response_json["data"].get("taskId")
    #             if task_id:
    #                 return task_id
    #             else:
    #                 st.error("🚨 Không tìm thấy taskId trong phản hồi!")
    #         else:
    #             st.error("🚨 Dữ liệu 'data' không tồn tại trong phản hồi API!")
    #             st.write("📄 Nội dung API trả về:", response.text)
    #     except ValueError as e:
    #         st.error(f"🚨 Không thể phân tích JSON từ API: {e}")
    #         st.write("📄 Nội dung API trả về:", response.text)
    # else:
    #     st.error(f"🚨 API trả về lỗi: {response.status_code}")
    #     st.write("📄 Nội dung lỗi:", response.text)
    # return None

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
                st.write("🛠️ Trạng thái từ API:", status)
                st.write("📄 Full dữ liệu API trả về:", data)
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

 
    api_token = st.secrets["api_token"]  # Lấy token từ secrets
    custom_mode = st.toggle("Custom Mode", value=True)
    if "lyrics" in st.session_state:
        lyrics = st.session_state.lyrics
        prompt = st.text_area("💡 Enter a description of the track you want to create:", 
                              value=lyrics, 
                              placeholder="A relaxing piano piece with a gentle melody...")
    else:
        prompt = st.text_area("💡 Enter a description of the track you want to create:", 
                              placeholder="A relaxing piano piece with a gentle melody...")

    # Kiểm tra nếu custom_mode tắt
    if custom_mode == False:
        style = "Classical"  # Gán giá trị mặc định nếu custom_mode tắt
        title = "My AI Music"  # Gán title mặc định nếu custom_mode tắt
        instrumental = False  # Gán giá trị mặc định cho instrumental nếu custom_mode tắt
    else:
        # Danh sách gợi ý phong cách nhạc
        music_styles = ["Classical", "Jazz", "Lo-fi", "Ambient", "Rock"]

        # Nếu chưa có session_state cho style_list, đặt giá trị mặc định
        if "style_list" not in st.session_state:
            st.session_state["style_list"] = []

        # Hộp nhập phong cách nhạc (hiển thị danh sách dưới dạng chuỗi)
        style = st.text_input("🎼 Enter music style:", ", ".join(st.session_state["style_list"]))

        # Đảm bảo style được sử dụng khi gửi yêu cầu
        style = style if style else "Classical"  # Nếu người dùng không nhập, sử dụng mặc định "Classical"

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
            # ✅ Kiểm tra user đã đăng nhập
            if "user" not in st.session_state:
                st.warning("🔐 Bạn cần đăng nhập để sử dụng tính năng này.")
                st.stop()

            user_id = st.session_state["user"]["id"]

            # ✅ Kiểm tra số dư
            credit_data = supabase.table("user_credits").select("credits").eq("id", user_id).execute()
            current_credits = credit_data.data[0]["credits"] if credit_data.data else 0

            if current_credits < 25:
                st.error("❌ Bạn không đủ 25 tín dụng để sử dụng chức năng này. Vui lòng nạp thêm.")
                st.stop()

            # ✅ Xóa nhạc cũ nếu có
            if "music_data" in st.session_state:
                del st.session_state["music_data"]

            if not api_token or not prompt:
                st.warning("⚠️Please enter music description!")
            else:
                task_id = await generate_music(api_token, prompt, custom_mode, style, title, instrumental)
                if task_id:
                    render_game_html()

                    music_data = await check_music_status(api_token, task_id)

                    if music_data:
                        # ✅ Trừ tín dụng nếu nhạc tạo thành công
                        new_credits = current_credits - 25
                        supabase.table("user_credits").update({"credits": new_credits}).eq("id", user_id).execute()

                        st.session_state["music_data"] = music_data
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


if menu == "Library":
    st.markdown("<h1 style='text-align: center; color: white;'>Your Music Library</h1>", unsafe_allow_html=True)
    
    # Kiểm tra nếu người dùng đã đăng nhập
    if "user" in st.session_state:
        # Lấy email người dùng từ session state
        user_email = st.session_state['user']['email']
        # user_id = st.session_state['user']['id']
            # In ra email và id của người dùng
        st.write(f"User Email: {user_email}")
        # st.write(f"User ID: {user_id}")
 
        # Truy vấn user_id từ bảng user_profiles thay vì bảng auth.users, sử dụng email thay vì full_name
        user_id_query = supabase.table('user_profiles').select('id').eq('email', user_email).execute()
        user_id = user_id_query.data[0]['id'] if user_id_query.data else None
        
        if user_id:
            # Lấy danh sách bài hát từ bảng songs của người dùng
            songs_query = supabase.table('songs').select('*').eq('user_id', user_id).execute()
            
            if songs_query.data:
                for song in songs_query.data:
                    st.write(f"**Title**: {song['title']}")
                    if song.get('image_url'):
                        st.image(song['image_url'], width=100)  # Hiển thị ảnh bìa
                    st.audio(song['audio_url'], format="audio/mp3")  # Hiển thị player âm thanh
                    st.write(f"**Lyrics**: {song.get('lyrics', 'No lyrics available')}")
                    st.write(f"**Genre**: {song.get('genre', 'No genre available')}")
                    st.write(f"**Style**: {song.get('style', 'No style available')}")
                    st.write("---")
            else:
                st.warning("🎶 Bạn chưa có bài hát nào trong thư viện.")
        else:
            st.warning("🚨 Không tìm thấy người dùng.")
    else:
        st.warning("🔒 Bạn cần đăng nhập để xem thư viện nhạc của bạn.")

# =========================== QUẢN LÝ THANH TOÁN ===========================
import streamlit as st
import requests
import hmac
import hashlib
import uuid
from datetime import datetime
from supabase import create_client
import streamlit.components.v1 as components

# MoMo config
MOMO_CONFIG = {
    "MomoApiUrl": "https://test-payment.momo.vn/v2/gateway/api/create",
    "PartnerCode": "MOMO",
    "AccessKey": "F8BBA842ECF85",
    "SecretKey": "K951B6PE1waDMi640xX08PD3vg6EkVlz",
    "ReturnUrl": "https://aimusic-fvj4bjxfbumlktejiy6gb4.streamlit.app/",
    "IpnUrl": "https://webhook.site/b052aaf4-3be0-43c5-8bad-996d2d0c0e54",
    "RequestType": "captureWallet",
    "ExtraData": "Astronaut_Music_payment"
}

CREDIT_PACKAGES = [
    {"credits": 1000, "price_usd": 5},
    {"credits": 10000, "price_usd": 50},
    {"credits": 105000, "price_usd": 500},
    {"credits": 275000, "price_usd": 1250}
]

@st.cache_data(ttl=86400)
def get_usd_to_vnd():
    try:
        url = "https://v6.exchangerate-api.com/v6/5bfc9ccf0ed4b1708159250f/latest/USD"
        res = requests.get(url)
        if res.status_code == 200:
            rate = res.json()["conversion_rates"]["VND"]
            st.write(f"💱 Tỷ giá USD → VND (ExchangeRate-API): {rate:,.0f}")
            return int(rate)
    except:
        st.error("❌ Lỗi khi lấy tỷ giá.")
    return 25000

def generate_signature(data, secret_key):
    raw_signature = (
        f"accessKey={data['accessKey']}&amount={data['amount']}&extraData={data['extraData']}&"
        f"ipnUrl={data['ipnUrl']}&orderId={data['orderId']}&orderInfo={data['orderInfo']}&"
        f"partnerCode={data['partnerCode']}&redirectUrl={data['redirectUrl']}&"
        f"requestId={data['requestId']}&requestType={data['requestType']}"
    )
    return hmac.new(secret_key.encode(), raw_signature.encode(), hashlib.sha256).hexdigest()

if menu == "Quản lý thanh toán":
    st.title("💰 Quản Lý Thanh Toán")
    if "user" not in st.session_state:
        st.warning("🔐 Vui lòng đăng nhập.")
        st.stop()

    user_id = st.session_state["user"]["id"]

    # Lấy số dư hiện tại
    credit_data = supabase.table("user_credits").select("credits").eq("id", user_id).execute()
    credits = credit_data.data[0]["credits"] if credit_data.data else 0
    st.metric("Tín dụng hiện có", f"{credits:,} credits")

    # Bảng giá
    st.subheader("📦 Gói tín dụng")
    usd_to_vnd = get_usd_to_vnd()
    selected_package = st.selectbox(
        "Chọn gói mua:",
        [f"{p['credits']:,} credits - ${p['price_usd']}" for p in CREDIT_PACKAGES]
    )
    package = next(p for p in CREDIT_PACKAGES if f"{p['credits']:,}" in selected_package)
    price_vnd = int(package['price_usd'] * usd_to_vnd)

    # Tạo đơn hàng thanh toán
    if st.button("🔁 Thanh toán bằng MoMo"):
        order_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        order_info = f"Mua {package['credits']} credits cho user {user_id}"

        payload = {
            "partnerCode": MOMO_CONFIG["PartnerCode"],
            "accessKey": MOMO_CONFIG["AccessKey"],
            "requestId": request_id,
            "amount": str(price_vnd),
            "orderId": order_id,
            "orderInfo": order_info,
            "redirectUrl": MOMO_CONFIG["ReturnUrl"],
            "ipnUrl": MOMO_CONFIG["IpnUrl"],
            "extraData": MOMO_CONFIG["ExtraData"],
            "requestType": MOMO_CONFIG["RequestType"]
        }
        payload["signature"] = generate_signature(payload, MOMO_CONFIG["SecretKey"])

        res = requests.post(MOMO_CONFIG["MomoApiUrl"], json=payload)
        if res.status_code == 200 and res.json().get("payUrl"):
            pay_url = res.json()["payUrl"]

            # Lưu đơn hàng pending
            supabase.table("pending_payments").insert({
                "user_id": user_id,
                "order_id": order_id,
                "credits": package["credits"],
                "amount": price_vnd
            }).execute()

            # Hiển thị nút thanh toán
            st.success("✅ Đơn hàng đã được tạo. Bấm nút bên dưới để thanh toán.")
            st.markdown(f"""
                <a href="{pay_url}" target="_blank">
                    <button style="background-color:#f72585; color:white; padding:10px 20px;
                                   border:none; border-radius:5px; cursor:pointer;">
                        🚀 Mở MoMo để thanh toán
                    </button>
                </a>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Không tạo được đơn hàng.")

    # ✅ Xử lý khi quay lại từ MoMo qua ReturnUrl
    params = st.query_params
    order_id_param = params.get("orderId")
    result_code = params.get("resultCode")
    trans_id = params.get("transId")
    amount = int(params.get("amount", "0"))

    if order_id_param:
        exists = supabase.table("payment_history").select("*").eq("order_id", order_id_param).execute()
        if exists.data:
            st.info("Giao dịch đã được xử lý.")
        else:
            pending = supabase.table("pending_payments").select("*").eq("order_id", order_id_param).execute().data
            if pending:
                pending = pending[0]
                if result_code == "0":
                    supabase.table("user_credits").update({"credits": credits + pending["credits"]}).eq("id", user_id).execute()
                    supabase.table("payment_history").insert({
                        "user_id": user_id,
                        "order_id": order_id_param,
                        "amount": amount,
                        "credits": pending["credits"],
                        "status": "completed",
                        "payment_method": "momo",
                        "transaction_id": trans_id,
                        "created_at": datetime.utcnow().isoformat()
                    }).execute()
                    supabase.table("pending_payments").delete().eq("order_id", order_id_param).execute()
                    st.success(f"✅ Đã cộng {pending['credits']:,} tín dụng.")
                    st.rerun()
                else:
                    st.warning("❌ Thanh toán thất bại hoặc bị huỷ.")
    
    # ✅ Trường hợp không có orderId → Kiểm tra đơn pending chưa xác nhận
    if not order_id_param:
        pending_query = supabase.table("pending_payments").select("*").eq("user_id", user_id).execute()
        pending_data = pending_query.data[0] if pending_query.data else None

        if pending_data and not st.session_state.get("mock_payment_confirmed", False):

            if st.button("✅ Xác nhận thanh toán thành công"):
                supabase.table("user_credits").update({"credits": credits + pending_data["credits"]}).eq("id", user_id).execute()
                supabase.table("payment_history").insert({
                    "user_id": user_id,
                    "order_id": pending_data["order_id"],
                    "amount": pending_data["amount"],
                    "credits": pending_data["credits"],
                    "status": "completed",
                    "payment_method": "momo (mock)",
                    "transaction_id": str(uuid.uuid4())[:12],
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
                supabase.table("pending_payments").delete().eq("order_id", pending_data["order_id"]).execute()
                st.session_state["mock_payment_confirmed"] = True
                st.success("✅ Đã xác nhận thanh toán và cộng tín dụng.")
                st.rerun()



    
        
            
    
