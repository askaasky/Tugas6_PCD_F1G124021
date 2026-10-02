# Mini Project PCD - Deteksi Tanda Tangan

## Deskripsi
Mini project ini merupakan implementasi pengolahan citra digital untuk mendeteksi keberadaan tanda tangan pada dokumen.

Tahapan pengolahan citra yang digunakan meliputi:
* Crop area tanda tangan
* Konversi citra ke grayscale
* Global Thresholding
* Otsu Thresholding
* Operasi morfologi Opening
* Operasi morfologi Closing
* Perhitungan jumlah foreground pixel
* Pembuatan citra tanpa tanda tangan
* Penentuan keberadaan tanda tangan

Pada project ini, area tanda tangan yang dianalisis adalah tanda tangan Dekan pada dokumen ijazah universitas. Untuk pengujian citra tanpa tanda tangan, tanda tangan pada area crop dihilangkan menggunakan proses inpainting.

---

## Teknologi yang Digunakan
* Python
* OpenCV
* NumPy
* Pandas

---

## How to Run

### 1. Clone Repository
Buka Command Prompt atau PowerShell, kemudian jalankan:
```bash
git clone https://github.com/askaasky/Tugas6_PCD_F1G124021.git
```

Kemudian masuk ke folder project:
```bash
cd Tugas6_PCD_F1G124021
```
### 2. Siapkan Dataset

Masukkan citra dokumen ijazah ke dalam folder:

`TTD_ORI/`

Area crop tanda tangan yang digunakan pada program adalah:

```python
CROP_TTD = (2175, 1708, 893, 337)

### 3. Install Library
Install library yang diperlukan dengan perintah:
```bash
pip install opencv-python numpy pandas
```

### 4. Jalankan Program
Jalankan program dengan:
```bash
python main.py
```

---

## Hasil Pengolahan

Program melakukan beberapa tahap pengolahan:
* Grayscale
* Global Thresholding
* Otsu Thresholding
* Opening
* Closing
* Perhitungan foreground pixel
* Pembuatan citra tanpa tanda tangan

Hasil pengolahan disimpan pada folder:
`Hasil/`

---

## Metode yang Digunakan

* **Global Thresholding**  
  Global threshold digunakan untuk memisahkan foreground dan background menggunakan nilai ambang tertentu. Pada program ini digunakan nilai threshold 145.

* **Otsu Thresholding**  
  Metode Otsu menentukan nilai threshold secara otomatis berdasarkan distribusi intensitas citra.

* **Opening**  
  Opening digunakan untuk mengurangi noise kecil pada hasil thresholding.

* **Closing**  
  Closing digunakan untuk membantu menghubungkan bagian foreground yang terputus dan mengisi celah kecil.

* **Inpainting**
  Inpainting digunakan untuk membuat citra tanpa tanda tangan dengan menghilangkan bagian tanda tangan pada area crop.

* **Foreground Pixel**
  Jumlah foreground pixel digunakan untuk mengetahui banyaknya bagian citra yang dianggap sebagai foreground. Nilai ini kemudian digunakan untuk menentukan keberadaan tanda tangan.
---

## Output

Output program berupa:
* Citra hasil crop
* Citra grayscale
* Hasil Global Thresholding
* Hasil Otsu Thresholding
* Hasil Opening
* Hasil Closing
* Jumlah foreground pixel
* Persentase foreground
* Status keputusan (SIGNATURE PRESENT / SIGNATURE ABSENT)
* File `hasil_threshold.csv`
* File `hasil_pengujian.csv`

---

## Dataset

Dataset yang digunakan terdiri dari beberapa citra dokumen ijazah dengan kondisi kualitas citra yang berbeda.  
Contohnya:
* High Quality
* Low Contrast
* Blurred
* High Noise
* Low Resolution
* Faded / Underexposed
* Color Shift
* JPEG Compression
* Combined Degradation

---

## Author

* **Nama:** Yus Askia
* **NIM:** F1G124021
* **Mata Kuliah:** Pengolahan Citra Digital
* **Universitas:** Universitas Halu Oleo