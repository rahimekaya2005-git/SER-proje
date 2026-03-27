#kutuphaneler
import os
import numpy as np
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")  # ekran olmadan calistirmak icin

# klasor yollari
DATASET_DIR = Path(r"C:\proje\ser_proje\dataset") #veri setinin cekildigi yol
OUTPUT_DIR  = Path(r"C:\proje\ser_proje\spektrogramlar") #spektrogramlarin kaydedilecegi yol

# mel spektrogram parametreleri
N_MELS     = 128
HOP_LENGTH = 512
N_FFT      = 2048
FMAX       = 8000  # konusma sesi icin 8khz yeterli

#veri setindeki ses dosyalarini yukler, normalize eder
def yukle(dosya_yolu):
    # sesi yukluyoruz, sr=None orijinal ornekleme hizini koru demek
    # mono=True stereo olursa tek kanala indir
    y, sr = librosa.load(dosya_yolu, sr=None, mono=True)

    if len(y) == 0:
        raise ValueError("ses dosyasi bos!")

    # normalize et (max degere gore)
    y = y / (np.max(np.abs(y)) + 1e-9)  # sifira bolme hatasini onlemek icin kucuk sayi ekledim
    return y, sr

#ses sinyalinin mel-spektrogramini hesaplar, dB olcegine cevirir
def mel_spektrogram_hesapla(y, sr):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=FMAX
    )
    # power_to_db ile log olcegine cevir
    # top_db=80 ; dinamik araligi 80 dB ile sinirla, cok karanlik olmamasi icin
    S_dB = librosa.power_to_db(S, ref=np.max, top_db=80)
    return S_dB

#hesaplanan spektrogrami gorsellestirir, png olarak kaydeder
def kaydet(S_dB, sr, baslik, cikti_yolu):
    fig, ax = plt.subplots(figsize=(10, 4))

    # specshow librosa'nin gorsellestirme fonksiyonu
    img = librosa.display.specshow(
        S_dB,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmax=FMAX,
        ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(baslik)
    plt.tight_layout()

    # klasor yoksa olustur
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cikti_yolu, dpi=100)
    plt.close(fig)  # bellek sizmasi olmasin diye kapat

#veri setindeki tum ses dosyalarini gezerek yukaridaki fonsiyonlari cagirip sonucu gosterir
def main():
    if not DATASET_DIR.exists():
        print(f"HATA: klasor bulunamadi -> {DATASET_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    duygular = [d for d in DATASET_DIR.iterdir() if d.is_dir()]
    print(f"toplam {len(duygular)} duygu klasoru bulundu")

    basarili = 0
    basarisiz = 0

    for duygu_klasoru in duygular:
        duygu_adi = duygu_klasoru.name
        print(f"\n-- {duygu_adi} isleniyor --")

        dosyalar = [f for f in duygu_klasoru.iterdir()
                    if f.suffix.lower() in (".wav", ".mp3", ".flac")]

        for dosya in dosyalar:
            cikti = OUTPUT_DIR / duygu_adi / dosya.with_suffix(".png").name

            # zaten varsa tekrar isleme
            if cikti.exists():
                continue

            try:
                y, sr  = yukle(dosya)
                S_dB   = mel_spektrogram_hesapla(y, sr)
                baslik = f"{duygu_adi} - {dosya.name}"
                kaydet(S_dB, sr, baslik, cikti)

                print(f"  ok: {dosya.name}")
                basarili += 1

            except Exception as hata:
                print(f"  HATA: {dosya.name} -> {hata}")
                basarisiz += 1

    print(f"\nBitti! Basarili: {basarili}, Hata: {basarisiz}")


if __name__ == "__main__":
    main()