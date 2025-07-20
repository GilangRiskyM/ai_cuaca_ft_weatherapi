from flask import Flask, request, jsonify
import pandas as pd
import joblib
import requests
import logging
import os
from dotenv import load_dotenv

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load .env
load_dotenv()
API_KEY = os.getenv('API_KEY') # WeatherAPI key
if not API_KEY:
    logger.critical("Variabel environment API_KEY tidak diatur.")

# Konstanta
FORECAST_DAYS = 3
MODEL_PATH = 'model/model_cuaca.pkl'
ENCODER_PATH = 'model/label_encoder.pkl'
AKURASI_PATH = 'model/accuracy.txt'
FEATURE_ORDER_PATH = 'model/feature_order.pkl'
LAST_UPDATED_PATH = 'model/last_updated.txt'   # Tambahan path

# Flask App
app = Flask(__name__)

# Terjemahan label sesuai mapping akhir
TERJEMAHAN_CUACA = {
    'Clear': 'Cerah',
    'Cloudy': 'Berawan',
    'Overcast': 'Mendung',
    'Patchy Rain Possible': 'Kemungkinan Hujan Ringan',
    'Rain': 'Hujan',
    'Heavy Rain': 'Hujan Lebat',
    'Thunderstorm': 'Hujan Petir',
    'Fog': 'Berkabut',
    'Snow': 'Salju',
    'Other': 'Lainnya'
}

# Load model
def load_ml_components():
    try:
        model = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        logger.info("✅ Model dan Label Encoder dimuat.")
        return model, le
    except Exception as e:
        logger.critical(f"❌ Gagal memuat model/encoder: {e}")
        return None, None

# Load akurasi
def load_accuracy(path):
    try:
        with open(path, 'r') as f:
            return float(f.read().strip())
    except Exception as e:
        logger.warning(f"⚠️ Gagal memuat akurasi: {e}")
        return None

# Load urutan fitur
def load_feature_order(path):
    try:
        return joblib.load(path)
    except Exception as e:
        logger.critical(f"❌ Gagal memuat urutan fitur: {e}")
        return None

# Load last updated timestamp
def load_last_updated(path):
    try:
        with open(path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"⚠️ Gagal memuat timestamp terakhir model diperbarui: {e}")
        return None

model, le = load_ml_components()
AKURASI_MODEL = load_accuracy(AKURASI_PATH)
FEATURE_ORDER = load_feature_order(FEATURE_ORDER_PATH)
LAST_UPDATED = load_last_updated(LAST_UPDATED_PATH)

# Ambil data cuaca dari API
def get_weather_data(lokasi):
    if not API_KEY:
        raise ValueError("API key tidak tersedia.")
    url = f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={lokasi}&days={FORECAST_DAYS}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Fungsi normalisasi label untuk hasil API eksternal jika perlu
def normalize_condition(label):
    label = label.lower()
    if 'clear' in label or 'sunny' in label:
        return 'Clear'
    elif 'partly cloudy' in label or 'cloudy' in label:
        return 'Cloudy'
    elif 'overcast' in label:
        return 'Overcast'
    elif 'patchy rain' in label or 'light rain' in label:
        return 'Patchy Rain Possible'
    elif 'moderate rain' in label or (label.strip() == 'rain'):
        return 'Rain'
    elif 'heavy rain' in label:
        return 'Heavy Rain'
    elif 'thunderstorm' in label:
        return 'Thunderstorm'
    elif 'mist' in label or 'fog' in label:
        return 'Fog'
    elif 'snow' in label:
        return 'Snow'
    else:
        return 'Other'

# Proses prediksi
def process_forecast(forecast_data, ml_model, label_encoder):
    hasil = []

    for hari in forecast_data:
        try:
            tanggal = hari['date']
            jam_12 = hari.get('hour', [{}]*24)[12]  # jam 12 siang

            logger.info(f"📅 Memproses tanggal: {tanggal}")

            fitur_dict = {
                'temp_c': hari['day']['avgtemp_c'],
                'humidity': hari['day']['avghumidity'],
                'wind_kph': hari['day']['maxwind_kph'],
                'cloud': hari['day'].get('daily_chance_of_rain', 50),
                'dewpoint_c': jam_12.get('dewpoint_c', 20.0),
                'hour': 12,
                'precip_mm': jam_12.get('precip_mm', 0.0),
                'pressure_mb': jam_12.get('pressure_mb', 1010.0),
                'uv': jam_12.get('uv', 6.0),
            }

            fitur = pd.DataFrame([fitur_dict])
            fitur = fitur[FEATURE_ORDER]  # pastikan urutan cocok

            logger.info(f"📊 Fitur: {fitur.to_dict(orient='records')[0]}")

            # Prediksi
            prediksi = ml_model.predict(fitur)
            label_en = label_encoder.inverse_transform(prediksi)[0]
            label_id = TERJEMAHAN_CUACA.get(label_en, label_en)

            hasil.append({
                'tanggal': tanggal,
                'prediksi_cuaca': label_id,
                'detail': {
                    'suhu': fitur_dict['temp_c'],
                    'kelembapan': fitur_dict['humidity'],
                    'kecepatan_angin': fitur_dict['wind_kph'],
                    'kemungkinan_hujan': fitur_dict['cloud']
                }
            })

        except Exception as e:
            logger.error(f"❌ Gagal prediksi pada {tanggal}: {e}")
            continue

    return hasil

# Endpoint
@app.route('/predict-cuaca', methods=['POST'])
def predict():
    if not model or not le or not FEATURE_ORDER:
        return jsonify({"error": "Model belum dimuat"}), 503

    data = request.get_json()
    lokasi = data.get('lokasi')
    if not lokasi:
        return jsonify({"error": "Parameter 'lokasi' wajib diisi"}), 400

    try:
        weather_data = get_weather_data(lokasi)
        forecast = weather_data['forecast']['forecastday']
        hasil = process_forecast(forecast, model, le)

        return jsonify({
            'lokasi': lokasi,
            'hasil': hasil,
            'akurasi_model': f"{AKURASI_MODEL:.2f}%" if AKURASI_MODEL else None,
            'terakhir_model_diperbarui': LAST_UPDATED if LAST_UPDATED else None
        })
    except Exception as e:
        logger.error(f"❌ Gagal memproses prediksi: {e}")
        return jsonify({"error": "Gagal mengambil atau memproses data cuaca"}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=False)