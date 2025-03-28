import os
import tensorflow as tf

# ✅ Để đường dẫn đến file .h5 trong thư mục gốc của dự án
MODEL_PATH = os.path.abspath("music_genre_recog_model.h5")

def check_model():
    """ Kiểm tra file .h5 và load mô hình """
    print("🔍 Đang kiểm tra mô hình...")

    # 1️⃣ Kiểm tra file .h5 có tồn tại không
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy mô hình: {MODEL_PATH}")
        print("➡ Hãy kiểm tra lại thư mục và chắc chắn rằng mô hình tồn tại.")
        return

    print(f"✅ Tìm thấy mô hình: {MODEL_PATH}")

    # 2️⃣ Thử load mô hình
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Mô hình đã load thành công!")

        # 3️⃣ Hiển thị thông tin mô hình
        print("\n📌 Thông tin mô hình:")
        model.summary()
    except Exception as e:
        print(f"❌ Lỗi khi load mô hình: {e}")
        print("➡ Có thể file .h5 bị hỏng hoặc không đúng định dạng.")

if __name__ == "__main__":
    check_model()
