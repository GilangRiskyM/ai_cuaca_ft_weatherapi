import os
import requests
import json
import pandas as pd
import time
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY') #WeatherAPI key
KOTA = 'Yogyakarta'
TANGGAL_MULAI = '2025-01-01'
TANGGAL_SELESAI = date.today().strftime('%Y-%m-%d')
NAMA_FILE = 'data_cuaca_histori.csv'
MAX_RETRY = 5
RETRY_SLEEP = 5  # detik

def request_with_retry(url):
    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"🔁 Percobaan {attempt} gagal: {e}")
            if attempt < MAX_RETRY:
                print(f"⏳ Menunggu {RETRY_SLEEP} detik sebelum retry...")
                time.sleep(RETRY_SLEEP)
            else:
                print("❌ Gagal setelah beberapa percobaan.")
                return None

def ambil_data():
    # Cek file jika ada untuk resume
    if os.path.exists(NAMA_FILE):
        df_exist = pd.read_csv(NAMA_FILE)
        tanggal_sudah = set(pd.to_datetime(df_exist['time']).dt.date.astype(str))
        print(f"🔄 Mode resume: {len(tanggal_sudah)} hari sudah diambil sebelumnya.")
    else:
        df_exist = pd.DataFrame()
        tanggal_sudah = set()

    all_data = []
    tanggal_awal = datetime.strptime(TANGGAL_MULAI, '%Y-%m-%d')
    tanggal_akhir = datetime.strptime(TANGGAL_SELESAI, '%Y-%m-%d')
    jumlah_hari = (tanggal_akhir - tanggal_awal).days + 1

    print(f"📦 Mengambil data cuaca untuk {KOTA} dari {TANGGAL_MULAI} hingga {TANGGAL_SELESAI}...")

    for i in range(jumlah_hari):
        tanggal = tanggal_awal + timedelta(days=i)
        tanggal_str = tanggal.strftime('%Y-%m-%d')

        if tanggal_str in tanggal_sudah:
            print(f"⏩ {tanggal_str} sudah ada, lewati.")
            continue

        url = f"https://api.weatherapi.com/v1/history.json?key={API_KEY}&q={KOTA}&dt={tanggal_str}"
        
        response = request_with_retry(url)
        if response is None:
            print(f"❌ Gagal ambil data tanggal: {tanggal_str} setelah retry.")
            continue

        try:
            data = response.json()
            if 'forecast' in data and 'forecastday' in data['forecast'] and len(data['forecast']['forecastday']) > 0:
                hourly = data['forecast']['forecastday'][0]['hour']
                for jam in hourly:
                    all_data.append({
                        'time': jam['time'],
                        'hour': pd.to_datetime(jam['time']).hour,
                        'temp_c': jam.get('temp_c', 0),
                        'humidity': jam.get('humidity', 0),
                        'wind_kph': jam.get('wind_kph', 0),
                        'cloud': jam.get('cloud', 0),
                        'pressure_mb': jam.get('pressure_mb', 0),
                        'uv': jam.get('uv', 0),
                        'precip_mm': jam.get('precip_mm', 0),
                        'dewpoint_c': jam.get('dewpoint_c', 0),
                        'condition': jam['condition']['text']
                    })
                print(f"✅ {tanggal_str} - data {len(hourly)} jam berhasil diambil")
            else:
                print(f"⚠️  Tidak ada data untuk tanggal: {tanggal_str}")

        except json.JSONDecodeError:
            print(f"❌ JSON error untuk tanggal: {tanggal_str}")
            print("Response mentah:", response.text)

        # Delay supaya tidak kena rate limit
        time.sleep(1)

    # Gabungkan data baru dengan data lama (jika ada)
    if all_data:
        df_baru = pd.DataFrame(all_data)
        if not df_exist.empty:
            df_final = pd.concat([df_exist, df_baru], ignore_index=True)
            df_final = df_final.drop_duplicates(subset=['time'])  # Hindari duplikat
        else:
            df_final = df_baru
        df_final.to_csv(NAMA_FILE, index=False)
        print(f"\n✅ Selesai. Data berhasil disimpan ke `{NAMA_FILE}`")
    elif not df_exist.empty:
        print(f"ℹ️ Tidak ada data baru. Data lama tetap di `{NAMA_FILE}`")
    else:
        print("❌ Tidak ada data yang disimpan.")

if __name__ == '__main__':
    ambil_data()