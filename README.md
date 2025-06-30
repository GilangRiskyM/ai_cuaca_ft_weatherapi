# AI Prakiraan Cuaca dengan Framework Flask

Proyek ini menggunakan Python dan framework Flask untuk memprediksi cuaca berbasis data historis dan API eksternal. Model machine learning dilatih untuk memberikan prediksi cuaca pada lokasi tertentu.

## Cara Penggunaan

1. **Pastikan Python dan pip sudah terinstal**

   - Cek instalasi:
     ```
     python --version
     pip --version
     ```

2. **(Opsional) Buat virtual environment**

   - Buat environment:
     ```
     python -m venv venv
     ```
   - Aktifkan environment:
     - **Windows:**
       ```
       venv\Scripts\activate
       ```
     - **Linux/Mac OS:**
       ```
       source venv/bin/activate
       ```

3. **Instal semua dependency dari `requirements.txt`**

   ```
   pip install -r requirements.txt
   ```

4. **Buat file `.env`, salin isinya dari `.env.example`**

   - Edit `.env` untuk mengisi `API_KEY` dari [WeatherAPI](https://www.weatherapi.com/).

5. **Jalankan `get_weather_data.py`** (hanya jika API_KEY kamu adalah premium dan ingin mengunduh data historis dalam jumlah besar)

   ```
   python get_weather_data.py
   ```

6. **Jalankan `train_model.py`** untuk melatih model machine learning.

   ```
   python train_model.py
   ```

7. **Jalankan `app.py`** untuk menjalankan API Flask.

   ```
   python app.py
   ```

8. **Cek endpoint dengan Postman**
   - Buka aplikasi Postman.
   - Pilih method `POST`.
   - Masukkan URL:
     ```
     http://localhost:5000/predict-cuaca
     ```
   - Pada bagian Body, pilih `raw` dan format `JSON`, lalu isi dengan:
     ```json
     {
       "lokasi": "Jakarta"
     }
     ```
   - Klik **Send**.

---

**Catatan:**

- Untuk penggunaan data historis yang masif, API_KEY WeatherAPI harus bertipe premium.
- Model yang sudah dilatih dan file pendukung akan tersimpan di folder `model/`.
