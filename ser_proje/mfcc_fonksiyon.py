import os  # dosya, klasör islemleri icin
import pandas as pd   # verileri tablo seklinde saklamak, csv dosyası olusturmak icin
import opensmile   # ses işleme , ozellik cikarma

# opensmile nesnesi olusturma
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,   # sesin pitch,mfcc,enerji ozelliklerini cikarir
    feature_level=opensmile.FeatureLevel.Functionals   # her dosya icin tek satir veri uretir
)

# veri setinin(ses dosyalarinin) yolu
dataset_path = "C:/proje/ser_proje/dataset"  

# tum ses dosyalarinin ozelliklerini saklamak icin bos liste
all_features = []

# veri setindeki her duygu klasorunu tek tek geziyor
for emotion_folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, emotion_folder)
    # klasor degilse atlıyor
    if not os.path.isdir(folder_path):
        continue
    
    # klasordeki .wav dosyalarını secme
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            try:
                # ses dosyalarini isleme, ozellik cikarma
                features = smile.process_file(file_path)
                features["file"] = file_path
                # emotion klasor adı
                features["emotion"] = emotion_folder
                # ozellikler listeye eklenir
                all_features.append(features)
                print(f"✔ {file} işlendi")
            except Exception as e:
                # dosya islenemezse hata mesaji
                print(f"❌ {file} işlenemedi: {e}")

# liste bos degilse
if all_features:
    # tum data frameleri tek tablo haline getirme
    final_df = pd.concat(all_features)
    # csv dosyasina kaydetme
    final_df.to_csv("C:/proje/ser_proje/features.csv", index=False)
    print("✅ Tüm özellikler CSV dosyasına kaydedildi: features.csv")
# liste bossa
else:
    print("⚠ Hiçbir dosya işlenmedi...")