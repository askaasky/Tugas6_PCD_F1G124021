import cv2
import numpy as np
import pandas as pd
import os
import glob


# 1. konfigurasi

INPUT_FOLDER = "TTD_ORI"
OUTPUT_FOLDER = "Hasil"

CROP_TTD = (2175, 1708, 893, 337)

GLOBAL_THRESHOLD = 145
REMOVE_SIGNATURE_THRESHOLD = 150
KERNEL_SIZE = 3


# 2. buat folder output

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

os.makedirs(
    os.path.join(OUTPUT_FOLDER, "ada_ttd"),
    exist_ok=True
)

os.makedirs(
    os.path.join(OUTPUT_FOLDER, "tanpa_ttd"),
    exist_ok=True
)

os.makedirs(
    os.path.join(OUTPUT_FOLDER, "perbandingan"),
    exist_ok=True
)


# 3. crop

def crop_image(image, crop_area):

    x, y, w, h = crop_area

    crop = image[
        y:y + h,
        x:x + w
    ]

    return crop


# 4. grayscale

def convert_grayscale(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return gray


# 5. global threshold

def global_threshold(gray):

    _, binary = cv2.threshold(
        gray,
        GLOBAL_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV
    )

    return binary


# 6. otsu threshold

def otsu_threshold(gray):

    threshold_value, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return binary, threshold_value


# 7. morphology

def apply_morphology(binary):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (KERNEL_SIZE, KERNEL_SIZE)
    )

    opening = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel
    )

    return opening, closing


# 8. hitung foreground

def calculate_foreground(binary):

    foreground_pixels = np.count_nonzero(
        binary
    )

    total_pixels = binary.size

    foreground_ratio = (
        foreground_pixels /
        total_pixels
    ) * 100

    return (
        foreground_pixels,
        foreground_ratio
    )


# 9. membuat citra tanpa tanda tangan

def create_without_signature(crop):

    gray = convert_grayscale(
        crop
    )

    _, signature_mask = cv2.threshold(
        gray,
        REMOVE_SIGNATURE_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (KERNEL_SIZE, KERNEL_SIZE)
    )

    signature_mask = cv2.morphologyEx(
        signature_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    signature_mask = cv2.dilate(
        signature_mask,
        kernel,
        iterations=1
    )

    without_signature = cv2.inpaint(
        crop,
        signature_mask,
        3,
        cv2.INPAINT_TELEA
    )

    return without_signature


# 10. simpan perbandingan

def save_comparison(
    gray,
    global_result,
    otsu_result,
    output_path
):

    gray_image = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    global_image = cv2.cvtColor(
        global_result,
        cv2.COLOR_GRAY2BGR
    )

    otsu_image = cv2.cvtColor(
        otsu_result,
        cv2.COLOR_GRAY2BGR
    )

    cv2.putText(
        gray_image,
        "Grayscale",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        global_image,
        "Global Threshold",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        otsu_image,
        "Otsu Threshold",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    comparison = cv2.hconcat([
        gray_image,
        global_image,
        otsu_image
    ])

    cv2.imwrite(
        output_path,
        comparison
    )


# 11. proses satu citra

def process_image(
    crop,
    output_prefix,
    comparison_name
):

    gray = convert_grayscale(
        crop
    )

    global_binary = global_threshold(
        gray
    )

    global_opening, global_closing = (
        apply_morphology(
            global_binary
        )
    )

    global_pixels, global_ratio = (
        calculate_foreground(
            global_closing
        )
    )

    otsu_binary, otsu_value = (
        otsu_threshold(
            gray
        )
    )

    otsu_opening, otsu_closing = (
        apply_morphology(
            otsu_binary
        )
    )

    otsu_pixels, otsu_ratio = (
        calculate_foreground(
            otsu_closing
        )
    )

    cv2.imwrite(
        output_prefix + "_crop.jpg",
        crop
    )

    cv2.imwrite(
        output_prefix + "_grayscale.jpg",
        gray
    )

    cv2.imwrite(
        output_prefix + "_global_thresh.jpg",
        global_binary
    )

    cv2.imwrite(
        output_prefix + "_global_opening.jpg",
        global_opening
    )

    cv2.imwrite(
        output_prefix + "_global_closing.jpg",
        global_closing
    )

    cv2.imwrite(
        output_prefix + "_otsu_thresh.jpg",
        otsu_binary
    )

    cv2.imwrite(
        output_prefix + "_otsu_opening.jpg",
        otsu_opening
    )

    cv2.imwrite(
        output_prefix + "_otsu_closing.jpg",
        otsu_closing
    )

    comparison_path = os.path.join(
        OUTPUT_FOLDER,
        "perbandingan",
        comparison_name
    )

    save_comparison(
        gray,
        global_closing,
        otsu_closing,
        comparison_path
    )

    return {
        "global_pixels": global_pixels,
        "global_ratio": global_ratio,
        "otsu_pixels": otsu_pixels,
        "otsu_ratio": otsu_ratio,
        "otsu_threshold": otsu_value
    }


# 12. cari semua gambar

image_files = []

for extension in [
    "*.jpg",
    "*.jpeg",
    "*.png"
]:

    image_files.extend(
        glob.glob(
            os.path.join(
                INPUT_FOLDER,
                extension
            )
        )
    )

image_files = sorted(
    image_files
)


# 13. cek gambar

if len(image_files) == 0:

    print()
    print("ERROR: Tidak ada gambar ditemukan!")
    print(
        "Masukkan gambar ke folder:",
        INPUT_FOLDER
    )

    exit()


print()
print("=" * 80)
print("signature presence detection")
print("=" * 80)

print(
    f"Jumlah gambar ditemukan: {len(image_files)}"
)


# 14. siapkan hasil

results = []


# 15. proses citra ada tanda tangan

print()
print("=" * 80)
print("TAHAP 1 : CITRA DENGAN TANDA TANGAN")
print("=" * 80)


for file in image_files:

    filename = os.path.basename(
        file
    )

    # lewati file NoTTD asli

    if "_NoTTD" in filename:
        continue

    image = cv2.imread(
        file
    )

    if image is None:

        print(
            f"Gagal membaca: {filename}"
        )

        continue

    crop = crop_image(
        image,
        CROP_TTD
    )

    base_name = os.path.splitext(
        filename
    )[0]

    output_prefix = os.path.join(
        OUTPUT_FOLDER,
        "ada_ttd",
        base_name
    )

    comparison_name = (
        base_name +
        "_ADA_TTD_comparison.jpg"
    )

    data = process_image(
        crop,
        output_prefix,
        comparison_name
    )

    results.append({

        "Citra": filename,

        "Kondisi":
            "ADA TANDA TANGAN",

        "Metode":
            "Global",

        "Threshold":
            GLOBAL_THRESHOLD,

        "Foreground_Pixel":
            data["global_pixels"],

        "Foreground_Ratio":
            data["global_ratio"],

        "Prediksi": ""

    })

    results.append({

        "Citra": filename,

        "Kondisi":
            "ADA TANDA TANGAN",

        "Metode":
            "Otsu",

        "Threshold":
            data["otsu_threshold"],

        "Foreground_Pixel":
            data["otsu_pixels"],

        "Foreground_Ratio":
            data["otsu_ratio"],

        "Prediksi": ""

    })

    print(
        f"{filename:<45}"
        f"Global = "
        f"{data['global_ratio']:.2f}% | "
        f"Otsu = "
        f"{data['otsu_ratio']:.2f}%"
    )


# 16. proses citra tanpa tanda tangan sintetis

print()
print("=" * 80)
print("TAHAP 2 : CITRA TANPA TANDA TANGAN")
print("=" * 80)


for file in image_files:

    filename = os.path.basename(
        file
    )

    # lewati file NoTTD asli

    if "_NoTTD" in filename:
        continue

    image = cv2.imread(
        file
    )

    if image is None:

        print(
            f"Gagal membaca: {filename}"
        )

        continue

    crop = crop_image(
        image,
        CROP_TTD
    )

    # membuat crop tanpa tanda tangan

    crop_without_signature = (
        create_without_signature(
            crop
        )
    )

    base_name = os.path.splitext(
        filename
    )[0]

    output_prefix = os.path.join(
        OUTPUT_FOLDER,
        "tanpa_ttd",
        base_name
    )

    comparison_name = (
        base_name +
        "_TANPA_TTD_comparison.jpg"
    )

    data = process_image(
        crop_without_signature,
        output_prefix,
        comparison_name
    )

    results.append({

        "Citra":
            base_name +
            "_TANPA_TTD",

        "Kondisi":
            "TIDAK ADA TANDA TANGAN",

        "Metode":
            "Global",

        "Threshold":
            GLOBAL_THRESHOLD,

        "Foreground_Pixel":
            data["global_pixels"],

        "Foreground_Ratio":
            data["global_ratio"],

        "Prediksi": ""

    })

    results.append({

        "Citra":
            base_name +
            "_TANPA_TTD",

        "Kondisi":
            "TIDAK ADA TANDA TANGAN",

        "Metode":
            "Otsu",

        "Threshold":
            data["otsu_threshold"],

        "Foreground_Pixel":
            data["otsu_pixels"],

        "Foreground_Ratio":
            data["otsu_ratio"],

        "Prediksi": ""

    })

    print(
        f"{base_name + '_TANPA_TTD':<45}"
        f"Global = "
        f"{data['global_ratio']:.2f}% | "
        f"Otsu = "
        f"{data['otsu_ratio']:.2f}%"
    )


# 17. buat dataframe

df = pd.DataFrame(
    results
)


# 18. tentukan rule klasifikasi

global_data = df[
    df["Metode"] == "Global"
].copy()

present_data = global_data[
    global_data["Kondisi"]
    == "ADA TANDA TANGAN"
]

absent_data = global_data[
    global_data["Kondisi"]
    == "TIDAK ADA TANDA TANGAN"
]

min_present = present_data[
    "Foreground_Ratio"
].min()

max_absent = absent_data[
    "Foreground_Ratio"
].max()

RULE_THRESHOLD = (
    min_present +
    max_absent
) / 2


print()
print("=" * 80)
print("ATURAN KLASIFIKASI")
print("=" * 80)

print(
    f"Foreground minimum ADA TTD   : "
    f"{min_present:.2f}%"
)

print(
    f"Foreground maksimum TANPA TTD: "
    f"{max_absent:.2f}%"
)

print(
    f"Threshold keputusan          : "
    f"{RULE_THRESHOLD:.2f}%"
)


# 19. fungsi klasifikasi

def classify_signature(
    foreground_ratio
):

    if foreground_ratio >= RULE_THRESHOLD:

        return "SIGNATURE PRESENT"

    return "SIGNATURE ABSENT"


# 20. prediksi global

df.loc[
    df["Metode"] == "Global",
    "Prediksi"
] = df.loc[
    df["Metode"] == "Global",
    "Foreground_Ratio"
].apply(
    classify_signature
)


# 21. hasil Otsu untuk perbandingan

df.loc[
    df["Metode"] == "Otsu",
    "Prediksi"
] = "PERBANDINGAN"


# 22. evaluasi

evaluation = df[
    df["Metode"] == "Global"
].copy()

evaluation["Benar"] = (

    (
        evaluation["Kondisi"]
        == "ADA TANDA TANGAN"
    )
    &
    (
        evaluation["Prediksi"]
        == "SIGNATURE PRESENT"
    )

    |

    (
        evaluation["Kondisi"]
        == "TIDAK ADA TANDA TANGAN"
    )
    &
    (
        evaluation["Prediksi"]
        == "SIGNATURE ABSENT"
    )
)


# 23. hitung akurasi

accuracy = (
    evaluation["Benar"].mean()
) * 100


# 24. simpan hasil CSV

df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "hasil_threshold.csv"
    ),
    index=False
)

evaluation.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "hasil_pengujian.csv"
    ),
    index=False
)


# 25. tampilkan hasil

print()
print("=" * 110)
print("HASIL PENGUJIAN SISTEM")
print("=" * 110)

print(
    f"{'Citra':<45}"
    f"{'Kondisi':<25}"
    f"{'Foreground':<15}"
    f"{'Prediksi'}"
)

print("-" * 110)


for _, row in evaluation.iterrows():

    print(
        f"{row['Citra']:<45}"
        f"{row['Kondisi']:<25}"
        f"{row['Foreground_Ratio']:.2f}%"
        f"{'':<8}"
        f"{row['Prediksi']}"
    )


# 26. hasil akhir

print()
print("=" * 80)

print(
    f"AKURASI SISTEM : "
    f"{accuracy:.2f}%"
)

print("=" * 80)

print()
print("Folder hasil:")

print(
    os.path.abspath(
        OUTPUT_FOLDER
    )
)

print()
print("File CSV:")
print("- Hasil/hasil_threshold.csv")
print("- Hasil/hasil_pengujian.csv")

print()
print("Proses selesai.")