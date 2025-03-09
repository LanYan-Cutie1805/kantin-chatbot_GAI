# petranesian-lapar
Small generative AI project with Ollama + LLamaIndex.
This project utilises RAG to find relevant entries for food, drinks, and stalls in the campus cafetaria.



# menjalankan app.py aplikasi chatbot RAG menggunakan streamlit

# langkah:
# 1. unzip streamlit-rag-chatbot
# 2. buat virtual environment python di dalam folder & aktifkan venv
# 3. run aplikasi streamlit
# P.S. Ketika pertama kali run karena code menggunakan embeddings fastembed, akan dibutuhkan waktu yang lama untuk mendownload embeddings # mohon ditunggu. Untuk run yang berikutnya akan membutuhkan waktu untuk indexing saja (lama tapi tidak selama yang dulu)





# create virtual env
conda create rag1

# aktivasi virtual env
conda activate rag1

# install yang perlu
pip install -r requirement.txt

# jalankan streamlit di port 50000  pada address localhost gpu3
streamlit run app.py --server.port=50000 --server.address=127.0.0.2
streamlit run genshin.py --server.port=55000 --server.address=127.0.0.2
streamlit run sl_e.py --server.port=55000 --server.address=127.0.0.2

# buka terminal baru lalu generate reverse proxy
reverse-proxy-publish generate

# cek file services.toml, edit port number yang digunakan, gunakan port 50000, samakan dengan streamlit run 

# lalu apply reverse proxy
reverse-proxy-publish apply

# akan muncul URL yg bisa diclick dijalankan secara global. jalankan pada browser 
# misal https://u1002-streamlit.gpu3.petra.ac.id/
