# app.py
import streamlit as st
from main import main as main_page
from crawling_data import main as crawling_data_page
from preprocessing_data import main as preprocessing_data_page
from term_frequency import main as term_frequency_page
from topic_modelling import main as topic_modelling_page
from clustering_3topik import main as clustering_3topik_page
from clustering_4topik import main as clustering_4topik_page
from crawling_data_berita import main as crawling_data_berita_page

# Sidebar untuk navigasi
selected_page = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["Halaman Utama", "Halaman Crawling Data",
     "Halaman Preprocessing Data", "Halaman Term Frequency", "Halaman Topic Modelling",
     "Halaman Clustering dengan 3 Fitur", "Halaman Clustering dengan 4 Fitur",
     "Halaman Crowling Data Berita"
     ]
)

st.write("""
    Nama    : Alisa Sugiarti
    NIM     : 200411100194
    Kelas   : PPW-B
    """)

# Menampilkan konten halaman yang dipilih
if selected_page == "Halaman Utama":
    main_page()
elif selected_page == "Halaman Crawling Data":
    crawling_data_page()
elif selected_page == "Halaman Preprocessing Data":
    preprocessing_data_page()
elif selected_page == "Halaman Term Frequency":
    term_frequency_page()
elif selected_page == "Halaman Topic Modelling":
    topic_modelling_page()
elif selected_page == "Halaman Clustering dengan 3 Fitur":
    clustering_3topik_page()
elif selected_page == "Halaman Clustering dengan 4 Fitur":
    clustering_4topik_page
else:
    crawling_data_berita_page()
