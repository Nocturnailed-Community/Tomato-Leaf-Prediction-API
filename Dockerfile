# Gunakan image dasar yang mengandung Python
FROM python:3.8-slim

# Tentukan direktori kerja di dalam container
WORKDIR /app

# Salin requirements.txt ke dalam container
COPY requirements.txt .

# Instal dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi ke dalam container
COPY . .

# Ekspose port yang akan digunakan oleh aplikasi Flask
EXPOSE 5000

# Tentukan perintah untuk menjalankan aplikasi Flask dengan Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]