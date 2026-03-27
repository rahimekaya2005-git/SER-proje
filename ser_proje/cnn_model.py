import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical

# klasor yolu ve ayarlar
SPEKTROGRAM_DIR = Path(r"C:\proje\ser_proje\spektrogramlar")

IMG_SIZE   = (128, 128)
BATCH_SIZE = 32
EPOCHS     = 50
TEST_SIZE  = 0.2   # %20 test, %80 egitim
SINIFLAR   = ["Angry", "Happy", "Sad", "Calm"]


# 1.veri hazirlama
# klasordeki tum spektrogramlari okur, etiketleriyle numpy dizisi olarak dondurur
def veri_yukle():
    gorseller = []
    etiketler = []

    for sinif in SINIFLAR:
        sinif_klasor = SPEKTROGRAM_DIR / sinif

        # alt klasorler varsa onlarin icine bak (konusmaci klasorleri)
        png_dosyalari = list(sinif_klasor.rglob("*.png"))

        print(f"{sinif}: {len(png_dosyalari)} gorsel bulundu")

        for dosya in png_dosyalari:
            try:
                img = Image.open(dosya).convert("RGB")
                img = img.resize(IMG_SIZE)
                gorseller.append(np.array(img))
                etiketler.append(sinif)
            except Exception as hata:
                print(f"  HATA: {dosya.name} -> {hata}")

    X = np.array(gorseller, dtype="float32") / 255.0
    y = np.array(etiketler)
    return X, y

# gorselleri yukler, etiketleri sayisallastirir, %80 egitim %20 test ayirir
def veri_hazirla():
    print("Veriler yukleniyor...")
    X, y_ham = veri_yukle()
    print(f"Toplam {len(X)} gorsel yuklendi\n")

    # etiketleri sayiya cevir
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_ham)
    y = to_categorical(y_encoded)

    print("Sinif eslestirmeleri:")
    for i, sinif in enumerate(encoder.classes_):
        print(f"  {sinif} -> {i}")

    # stratify -> her siniftan esit oranda ornekle
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y_encoded
    )

    print(f"\nEgitim seti: {len(X_train)} gorsel")
    print(f"Test seti  : {len(X_test)} gorsel")

    return X_train, X_test, y_train, y_test, encoder

# 2. cnn modeli
# 4 bloklu cnn modeli olusturur ve derler
def model_olustur(sinif_sayisi):
    model = models.Sequential([
        Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        # 1. blok
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # 2. blok
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # 3. blok
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # 4. blok - yeni eklendi, daha derin ozellik ogrenmesi icin
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),  # Flatten yerine, parametre sayisini azaltir

        # tam bagli katmanlar
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(sinif_sayisi, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# 3. model egitimi ve sonuclar
# model egitim verileriyle egitilir, en iyi agirliklar korunur
def egit(model, X_train, X_test, y_train, y_test):
    print("\nModel egitiliyor...")

    # patience=10 -> 10 epoch iyilesme olmazsa dur (onceki 5'ti, cok erkendi)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",  # val_loss yerine val_accuracy izle
        patience=10,
        restore_best_weights=True
    )

    # her 10 epochta learning rate'i yariya indir
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )

    gecmis = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, lr_scheduler]
    )

    return gecmis

# egitim surecindeki accuracy ve loss degerlerini grafik olarak kaydeder
def sonuclari_goster(gecmis):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(gecmis.history["accuracy"], label="egitim")
    ax1.plot(gecmis.history["val_accuracy"], label="dogrulama")
    ax1.set_title("Model Dogrulugu")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dogruluk")
    ax1.legend()

    ax2.plot(gecmis.history["loss"], label="egitim")
    ax2.plot(gecmis.history["val_loss"], label="dogrulama")
    ax2.set_title("Model Kaybi")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Kayip")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(r"C:\proje\ser_proje\egitim_grafigi.png", dpi=100)
    plt.close()
    print("Grafik kaydedildi: egitim_grafigi.png")

# yukaridaki fonksiyonlari bastan sona sirayla calistirir
def main():
    X_train, X_test, y_train, y_test, encoder = veri_hazirla()

    sinif_sayisi = len(encoder.classes_)
    model = model_olustur(sinif_sayisi)
    model.summary()

    gecmis = egit(model, X_train, X_test, y_train, y_test)

    test_kayip, test_dogruluk = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Dogrulugu: %{test_dogruluk * 100:.2f}")
    print(f"Test Kaybi    : {test_kayip:.4f}")

    sonuclari_goster(gecmis)

    model.save(r"C:\proje\ser_proje\duygu_modeli.keras")
    print("Model kaydedildi: duygu_modeli.keras")

if __name__ == "__main__":
    main()