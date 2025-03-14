import os
import bcrypt
import re  # Thêm thư viện kiểm tra email hợp lệ
from openai import OpenAI
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

st.set_page_config(page_title="Music AI Website", layout="wide")
# Load API key từ file .env
load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Kết nối Supabase
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print(os.path.exists("D:/test/Music-Genre-Recognition-main/.streamlit/secrets.toml"))

# Session State để lưu trạng thái đăng nhập
if "user" not in st.session_state:
    st.session_state.user = None

# Hàm kiểm tra email hợp lệ
def is_valid_email(email):
    return re.match(r"^[a-zA-Z0-9._%+-]+@gmail\.com$", email)

# Giao diện trang đăng nhập
def login_page():
    st.title("🔑 Đăng Nhập")
    email = st.text_input("📧 Email", placeholder="Nhập email của bạn")
    password = st.text_input("🔒 Mật khẩu", type="password", placeholder="Nhập mật khẩu")

    if st.button("🚀 Đăng Nhập"):
            try:
                user = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.user = user
                st.success("✅ Đăng nhập thành công!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")


    st.markdown("---")
    st.markdown("🔹 **Chưa có tài khoản?** [Đăng ký ngay](#)")
    st.markdown("🔹 **Quên mật khẩu?** [Lấy lại mật khẩu](#)")

# Giao diện trang đăng ký
def register_page():
    st.title("📝 Đăng Ký")
    email = st.text_input("📧 Email", placeholder="Nhập email")
    password = st.text_input("🔒 Mật khẩu", type="password", placeholder="Nhập mật khẩu")
    confirm_password = st.text_input("🔒 Xác nhận mật khẩu", type="password", placeholder="Nhập lại mật khẩu")

    if st.button("✅ Đăng Ký"):
        if not is_valid_email(email):
            st.error("⚠️ Vui lòng nhập địa chỉ email hợp lệ có dạng @gmail.com!")
        elif password != confirm_password:
            st.error("⚠️ Mật khẩu không khớp!")
        else:
            try:
                supabase.auth.sign_up({"email": email, "password": password})
                st.success("🎉 Đăng ký thành công! Kiểm tra email để xác nhận.")
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

# Giao diện trang quên mật khẩu
def reset_password_page():
    st.title("🔑 Quên Mật Khẩu")
    email = st.text_input("📧 Email", placeholder="Nhập email của bạn")

    if st.button("🔄 Lấy lại mật khẩu"):
        try:
            supabase.auth.reset_password_for_email(email)
            st.success("📩 Kiểm tra email để đặt lại mật khẩu!")
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")

# Giao diện trang chính sau khi đăng nhập
def main_page():
    st.title("🎉 Chào mừng bạn!")
    st.success(f"✅ Bạn đã đăng nhập với email: {st.session_state.user['user']['email']}")
    
    if st.button("🚪 Đăng xuất"):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.experimental_rerun()

# Điều hướng giữa các trang
if st.session_state.user:
    main_page()
else:
    option = st.sidebar.radio("🔹 Chọn chức năng", ["🔑 Đăng Nhập", "📝 Đăng Ký", "🔄 Quên Mật Khẩu"])
    if option == "🔑 Đăng Nhập":
        login_page()
    elif option == "📝 Đăng Ký":
        register_page()
    else:
        reset_password_page()

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





# Tạo menu Sidebar có icon
with st.sidebar:
    st.image("D:/test/Music-Genre-Recognition-main/.image/a-minimalist-logo-design-on-a-black-back_0AWYUQ3rQfy5rgcfFzPdJQ_5N7Moh5lTRa_PQanVq-UkQ.jpeg", use_container_width=True)
    


    menu = option_menu(
        menu_title="Navigation",
        options=["Home", "Create Lyrics", "Feel The Beat", "Classify", "Explore", "Library", "Search"],
        icons=["house", "music-note-list", "soundwave", "graph-up", "globe", "book", "search"],
        menu_icon="menu-button-wide",
        default_index=0,
        styles={
            "container": {"background-color": "rgba(0,0,0,0.8)", "padding": "5px"},
            "icon": {"color": "#feb47b", "font-size": "20px"},
            "nav-link": {"font-size": "18px", "color": "#ffffff", "text-align": "left", "margin": "5px"},
            "nav-link-selected": {"background-color": "#ff7e5f"},
        }
    )




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




# =================== GIAO DIỆN CHO CREATE LYRICS ===================
if menu == "Create Lyrics":
    st.markdown("<h1>🎶 AI Lyric Generator 🎵</h1>", unsafe_allow_html=True)

    # Người dùng nhập thể loại nhạc và chủ đề
    genre = st.selectbox("🎼 Chọn thể loại nhạc:", ["Pop", "Rock", "Hip-Hop", "Jazz", "Ballad", "EDM"])
    theme = st.text_input("✍️ Nhập chủ đề bài hát (VD: Tình yêu, Mùa thu, Tuổi trẻ, ...)")
    mood = st.radio("🎭 Chọn cảm xúc:", ["Vui vẻ", "Buồn", "Hào hứng", "Thư giãn", "Kịch tính"])

    if st.button("🎤 Sáng tác ngay!"):
        if theme.strip():
            with st.spinner("🎶 AI đang sáng tác lời bài hát cho bạn..."):
                prompt = f"Hãy viết lời bài hát thể loại {genre} về chủ đề '{theme}', với cảm xúc {mood}."
                lyrics = generate_lyrics(prompt)
                print(lyrics)
                st.text_area("🎼 Lời bài hát AI tạo:", lyrics, height=300)
        else:
            st.warning("⚠️ Vui lòng nhập chủ đề bài hát trước khi tạo!")
       



if menu == "Feel The Beat":
    st.title("🎵 Feel The Beat - Tạo Nhạc AI")

    # Nhập API Token
    api_token = st.text_input("🔑 Nhập API Token:", type="password")

    # Nhập mô tả nhạc cần tạo
    prompt = st.text_area("💡 Nhập mô tả bản nhạc bạn muốn tạo:", 
    placeholder="Một bản nhạc piano thư giãn với giai điệu nhẹ nhàng...")

    # Danh sách gợi ý phong cách nhạc
    music_styles = ["Classical", "Jazz", "Lo-fi", "Ambient", "Rock"]

    # Nếu chưa có session_state cho style_list, đặt giá trị mặc định
    if "style_list" not in st.session_state:
        st.session_state["style_list"] = []

    # Hộp nhập phong cách nhạc (hiển thị danh sách dưới dạng chuỗi)
    style = st.text_input("🎼 Nhập phong cách nhạc:", ", ".join(st.session_state["style_list"]))

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

    title = st.text_input("🎶 Đặt tên bản nhạc:", "My AI Music")
    instrumental = st.checkbox("🎻 Nhạc không lời?", value=False)

    # Xử lý khi bấm nút
    if st.button("🎧 Feel The Beat"):
        if not api_token or not prompt:
            st.warning("⚠️ Vui lòng nhập API Token và mô tả nhạc!")
        else:
            # Gửi yêu cầu API tạo nhạc
            api_url = "https://apibox.erweima.ai/api/v1/generate"
            headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
            data = {
                "prompt": prompt,
                "style": style,
                "title": title,
                "customMode": True,
                "instrumental": instrumental,
                "model": "V3_5",
                "callBackUrl": "https://api.example.com/callback"
            }

            with st.spinner("🎼 Đang tạo nhạc..."):
                response = requests.post(api_url, json=data, headers=headers)

            # Xử lý kết quả
            if response.status_code == 200:
                task_id = response.json().get("data", {}).get("taskId", None)
                st.write("📌 Task ID:", task_id)  # Debug Task ID

                if not task_id:
                    st.error("🚨 API không trả về Task ID!")
                else:
                    check_url = f"https://apibox.erweima.ai/api/v1/generate/record-info?taskId={task_id}"
                    headers = {
                        "Authorization": f"Bearer {api_token}",
                        "Accept": "application/json"
                    }

                    st.write("nhạc đang tạo vui lòng chờ 5 phút")
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
                        🔥 Chào mừng đến với T-Rex Game! 🔥
                        </div>
                        <p style="
                            font-size: 16px; /* Nhỏ hơn tiêu đề */
                            font-weight: normal; /* Không in đậm */
                        ">
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
                    audio_url = None

                    for _ in range(60):  # Lặp tối đa 60 lần (5 phút)
                        check_response = requests.get(check_url, headers=headers)

                        if check_response.status_code == 200:
                            try:
                                music_info = check_response.json()
                                data = music_info.get("data", {})
                                status = data.get("status", "PENDING")  # Kiểm tra trạng thái

                                if status == "SUCCESS":
                                    response_data = data.get("response", {})
                                    suno_data = response_data.get("sunoData", [])

                                    if suno_data and isinstance(suno_data, list):
                                        audio_url = suno_data[0].get("audioUrl")
                                        img_url = suno_data[0].get("imageUrl",)
                                        title_data = suno_data[0].get("title")
                                if audio_url:
                                    break  # Dừng vòng lặp nếu đã có nhạc

                            except Exception as e:
                                st.error(f"🚨 Lỗi khi xử lý JSON từ API: {e}")
                                st.write("📄 Nội dung API trả về:", check_response.text)
                                break  # Nếu lỗi, dừng luôn
                        time.sleep(5)  # Chờ 5 giây trước khi kiểm tra lại

                    # Kiểm tra kết quả sau vòng lặp
                    if audio_url:
                        status = st.empty()
                        st.success(f"🎵 Nhạc đã sẵn sàng: [{title}]({audio_url})")
                        image = img_url
                        title = title_data  # Thay bằng tiêu đề bài hát
                        # Thiết kế giao diện phát nhạc đẹp
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

                                /* Tùy chỉnh thanh tiến trình */
                                audio::-webkit-media-controls-timeline {
                                    background: linear-gradient(90deg, #ff7e5f, #feb47b) !important;
                                    border-radius: 30px;
                                    height: 6px;
                                    box-shadow: 0px 0px 10px rgba(255, 126, 95, 0.8);
                                    transition: all 0.3s ease-in-out;
                                    padding:1px;
                                }
                                
                                /* Chỉnh màu nút Play/Pause */
                                audio::-webkit-media-controls-play-button {
                                    background-color: #ff7e5f !important;
                                    box-shadow: 0px 0px 10px rgba(255, 126, 95, 0.8);
                                    border-radius: 50%;
                                }

                                audio::-webkit-media-controls-volume-slider {
                                    background: #ff7e5f !important;
                                }

                                /* Thiết kế tiêu đề bài hát */
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
                        col1, col2 = st.columns([1, 5])  # Cột trái (1 phần), cột phải (2 phần)
                        with col1:
                        # Chèn hình ảnh bài hát
                            st.image(image, width=150)
                        with col2:
                            # Hiển thị tiêu đề bài hát
                            st.markdown(f'<div class="song-title">{title}</div>', unsafe_allow_html=True)
                            
                            # Hiển thị trình phát nhạc
                            st.markdown('<div class="audio-container">', unsafe_allow_html=True)
                            st.audio(audio_url, format="audio/mp3")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("⏳ Nhạc chưa sẵn sàng sau 5 phút, hãy thử lại sau!")
            else:
                st.error(f"🚨 Lỗi API: {response.json().get('error', 'Không rõ lỗi!')}")