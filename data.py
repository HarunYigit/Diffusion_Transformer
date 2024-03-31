import cv2
import os

# Giriş ve çıkış klasörlerini tanımlayın
input_folder = "./a"
output_folder = "./b"

# Giriş klasöründeki resimleri dolaşın
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Sadece belirli uzantılardaki dosyaları işle
        # Resmi oku
        input_path = os.path.join(input_folder, filename)
        img = cv2.imread(input_path)

        # Gri tonlamaya dönüştür
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold ile kenarları çıkar
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Çıktı dosyasının yolunu oluştur
        output_path = os.path.join(output_folder, filename)

        # Çıktıyı kaydet
        cv2.imwrite(output_path, thresh)

        print(f"{filename} işlendi ve {output_path} olarak kaydedildi.")
