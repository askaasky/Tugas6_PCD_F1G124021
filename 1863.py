import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# 1. Konfigurasi folder Utama & folder hasil
FOLDER_INPUT = "TTD_ORI"
FOLDER_HASIL = "Hasil"

FOLDERS = [
    "1_citra_asli",
    "2_grayscale",
    "3_global_thresh",
    "4_global_opening",
    "5_global_closing",
    "6_otsu_thresh",
    "7_otsu_opening",
    "8_otsu_closing",
    "9_perbandingan"
]

# Membuat semua folder hasil
for folder in FOLDERS:
    os.makedirs(os.path.join(FOLDER_HASIL, folder), exist_ok=True)


# 2. Parameter cropping & decision rule
CROP_TTD = (2175, 1708, 893, 337)
CROP_KOSONG = (2791, 1634, 657, 250)

THRESHOLD_PERCENTAGE = 2.0


# 3. Fungsi proses gambar
def proses_dan_ekspor_folder(image_path, is_empty_sample=False):

    img = cv2.imread(image_path)

    if img is None:
        print(f"Gagal membaca file: {image_path}")
        return None

    nama_file = os.path.basename(image_path)
    nama_tanpa_ext = os.path.splitext(nama_file)[0]

    # Crop ROI original
    if "_NoTTD" in nama_file:
        x, y, w, h = CROP_KOSONG
    else:
        x, y, w, h = CROP_TTD

    roi = img[y:y+h, x:x+w]

    # Simpan citra ROI original
    cv2.imwrite(
        os.path.join(
            FOLDER_HASIL,
            FOLDERS[0],
            f"{nama_tanpa_ext}_roi.png"
        ),
        roi
    )

    # 4. Grayscale
    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    cv2.imwrite(
        os.path.join(
            FOLDER_HASIL,
            FOLDERS[1],
            f"{nama_tanpa_ext}_gray.png"
        ),
        gray
    )

    # 5. Global Thresholding
    # Fixed threshold = 127
    _, thresh_global = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    cv2.imwrite(
        os.path.join(
            FOLDER_HASIL,
            FOLDERS[2],
            f"{nama_tanpa_ext}_global.png"
        ),
        thresh_global
    )

    # 6. Otsu Thresholding
    otsu_val, thresh_otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    cv2.imwrite(
        os.path.join(
            FOLDER_HASIL,
            FOLDERS[5],
            f"{nama_tanpa_ext}_otsu.png"
        ),
        thresh_otsu
    )

    # 7. Operasi Morfologi
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    # Morfologi pada Global Threshold
    global_opening = cv2.morphologyEx(
        thresh_global,
        cv2.MORPH_OPEN,
        kernel
    )

    global_closing = cv2.morphologyEx(
        global_opening,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Morfologi pada Otsu Threshold
    otsu_opening = cv2.morphologyEx(
        thresh_otsu,
        cv2.MORPH_OPEN,
        kernel
    )

    otsu_closing = cv2.morphologyEx(
        otsu_opening,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Simpan hasil Global Opening
    cv2.imwrite(
        os.path.join(
            FOLDER_HASIL,
            FOLDERS[3],
            f"{nama_tanpa_ext}_global_opening.png"
        ),
        global_opening
    )

    # Simpan hasil Otsu Opening
    cv2.imwrite(
        os.path.join(
            FOLDER_HASIL,
            FOLDERS[3],
            f"{nama_tanpa_ext}_otsu_opening.png"
        ),
        otsu_opening
    )

    # Simpan hasil Global Closing
    cv2.imwrite(
        os.path.join(
            FOLDER_HASIL,
            FOLDERS[4],
            f"{nama_tanpa_ext}_global_closing.png"
        ),
        global_closing
    )

    # Simpan hasil Otsu Closing
    cv2.imwrite(
        os.path.join(
            FOLDER_HASIL,
            FOLDERS[4],
            f"{nama_tanpa_ext}_otsu_closing.png"
        ),
        otsu_closing
    )

    # 8. Perhitungan jumlah piksel & Status Keputusan
    total_pixels = gray.shape[0] * gray.shape[1]

    # Menghitung jumlah piksel foreground
    fg_global = cv2.countNonZero(
        global_closing
    )

    fg_otsu = cv2.countNonZero(
        otsu_closing
    )

    # Menghitung rasio foreground
    ratio_global = (
        fg_global / total_pixels
    ) * 100

    ratio_otsu = (
        fg_otsu / total_pixels
    ) * 100

    # 9. Decision Rule
    if ratio_global < 0.5 or (
        otsu_val > 220 and ratio_otsu > 15.0
    ):

        status = "SIGNATURE ABSENT"
        jumlah_piksel = fg_global
        ratio_final = ratio_global

    else:

        status = (
            "SIGNATURE PRESENT"
            if ratio_otsu > THRESHOLD_PERCENTAGE
            else "SIGNATURE ABSENT"
        )

        jumlah_piksel = fg_otsu
        ratio_final = ratio_otsu

    # 10. Membuat gambar perbandingan gabungan
    plt.figure(figsize=(16, 8))

    plt.suptitle(
        f"Analisis Morfologi & Deteksi: {nama_file}",
        fontsize=13,
        fontweight="bold"
    )

    # Baris pertama:
    # Citra asli, grayscale, global threshold,
    # global opening, global closing

    plt.subplot(2, 5, 1)

    plt.imshow(
        cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("Citra ROI")
    plt.axis("off")

    plt.subplot(2, 5, 2)

    plt.imshow(
        gray,
        cmap="gray"
    )

    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(2, 5, 3)

    plt.imshow(
        thresh_global,
        cmap="gray"
    )

    plt.title(
        "Global Threshold\n(T = 127)"
    )

    plt.axis("off")

    plt.subplot(2, 5, 4)

    plt.imshow(
        global_opening,
        cmap="gray"
    )

    plt.title("Global Opening")
    plt.axis("off")

    plt.subplot(2, 5, 5)

    plt.imshow(
        global_closing,
        cmap="gray"
    )

    plt.title("Global Closing")
    plt.axis("off")

    # Baris kedua:
    # Otsu thresholding dan hasil morfologi

    plt.subplot(2, 5, 6)

    plt.imshow(
        thresh_otsu,
        cmap="gray"
    )

    plt.title(
        f"Otsu Threshold\n(T = {int(otsu_val)})"
    )

    plt.axis("off")

    plt.subplot(2, 5, 7)

    plt.imshow(
        otsu_opening,
        cmap="gray"
    )

    plt.title("Otsu Opening")
    plt.axis("off")

    plt.subplot(2, 5, 8)

    plt.imshow(
        otsu_closing,
        cmap="gray"
    )

    plt.title("Otsu Closing")
    plt.axis("off")

    plt.subplot(2, 5, 9)
    plt.axis("off")

    plt.subplot(2, 5, 10)

    warna_teks = (
        "green"
        if status == "SIGNATURE PRESENT"
        else "red"
    )

    plt.text(
        0.5,
        0.65,
        status,
        fontsize=12,
        color=warna_teks,
        ha="center",
        va="center",
        weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            edgecolor=warna_teks,
            facecolor="white",
            lw=2
        )
    )

    plt.text(
        0.5,
        0.40,
        f"Piksel: {jumlah_piksel} px",
        fontsize=10,
        ha="center",
        va="center"
    )

    plt.text(
        0.5,
        0.25,
        f"Rasio: {ratio_final:.2f}%",
        fontsize=10,
        ha="center",
        va="center"
    )

    plt.title("Status Keputusan")
    plt.axis("off")

    plt.tight_layout()

    # Simpan gambar perbandingan
    path_perbandingan = os.path.join(
        FOLDER_HASIL,
        FOLDERS[8],
        f"Perbandingan_{nama_tanpa_ext}.png"
    )

    plt.savefig(
        path_perbandingan,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # 11. Menampilkan hasil proses
    print(
        f"Selesai Memproses: "
        f"{nama_file:<35} | "
        f"Piksel: {jumlah_piksel:<6} | "
        f"Rasio: {ratio_final:.2f}% | "
        f"Status: {status}"
    )

    # 12. Data untuk CSV
    return [
        nama_file,
        "Area Kosong"
        if is_empty_sample
        else "Area TTD",
        jumlah_piksel,
        f"{ratio_final:.2f}%",
        status
    ]


# 13. Execution program
if __name__ == "__main__":

    if os.path.exists(FOLDER_INPUT):

        files = sorted([
            f
            for f in os.listdir(FOLDER_INPUT)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        print(
            "\nMEMPROSES DAN MENYIMPAN "
            "HASIL PER SUB-FOLDER\n"
        )

        rekap_data = []

        for file_name in files:

            full_path = os.path.join(
                FOLDER_INPUT,
                file_name
            )

            # Menentukan area uji berdasarkan nama file
            is_empty = "_NoTTD" in file_name

            row = proses_dan_ekspor_folder(
                full_path,
                is_empty_sample=is_empty
            )

            if row:
                rekap_data.append(row)

        # 14. Simpan Rekapitulasi Data ke CSV
        csv_path = os.path.join(
            FOLDER_HASIL,
            "rekapitulasi_hasil.csv"
        )

        with open(
            csv_path,
            mode="w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "Nama File",
                "Area Uji",
                "Jumlah Piksel Foreground",
                "Rasio Foreground",
                "Status Keputusan"
            ])

            writer.writerows(rekap_data)

        print(
            "\n[BERHASIL] Seluruh hasil gambar "
            "dan CSV tersimpan lengkap "
            f"di folder '{FOLDER_HASIL}/'!"
        )

    else:

        print(
            f"Folder '{FOLDER_INPUT}' "
            "tidak ditemukan!"
        )