from nltk.tokenize import sent_tokenize
import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
import math
from collections import Counter
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt


def scrape_detik(url):
    req = requests.get(url)
    sop = BeautifulSoup(req.text, 'html.parser')
    li = sop.find('div', class_='list media_rows list-berita')
    lin = li.find_all('article')

    x = lin[0]  # Ambil hanya satu artikel, yaitu yang pertama
    link = x.find('a')['href']
    date = x.find('a').find('span', class_='date').text.replace(
        'WIB', '').replace('detikNews', '').split(',')[1]
    headline = x.find('a').find('h2').text

    ge_ = requests.get(link).text
    sop_ = BeautifulSoup(ge_, 'html.parser')
    content = sop_.find('div', class_='detail__body-text itp_bodycontent')

    paragraphs = content.find_all('p')
    content_ = ''.join([p.get_text(strip=True) for p in paragraphs])

    return content_


def tokenisasi_text(teks):
    # Tokenisasi kalimat
    kalimat = sent_tokenize(teks)
    return kalimat


def main():
    st.title("Halaman Crowling Data Berita")
    st.write("""
        Crowling data adalah proses otomatis untuk mengumpulkan dan mengindeks data dari berbagai sumber seperti situs web, database, atau dokumen.
        Crowling data dilakukan pada website detik.com
        """)

    # Input URL
    url_input = st.text_input(
        "Masukkan URL Detik.com:", "https://www.detik.com/search/searchnews?query=pemilu+2024&sortby=time&page=1")

    # Tombol untuk memulai scraping
    if st.button("Scrape"):
        try:
            # Panggil fungsi untuk scraping
            content = scrape_detik(url_input)

            # Tampilkan hasil scraping
            st.write("Konten dari artikel:")
            st.write(content)

        except Exception as e:
            st.write("Error:", e)

    nltk.download('punkt')

    st.header("Tokenisasi Kalimat")
    st.write("Tokenisasi kalimat adalah proses memecah teks menjadi kalimat-kalimat individual. Tujuan dari tokenisasi kalimat adalah untuk memisahkan teks menjadi unit-unit yang lebih kecil, yaitu kalimat, sehingga memudahkan analisis dan pemrosesan lebih lanjut pada tingkat kalimat.")

    # Panggil fungsi untuk tokenisasi satu artikel
    if "content" in locals():
        sentence = tokenisasi_text(content)
        st.write(sentence)

        # Menghitung jumlah kata dalam setiap kalimat
        tf_kalimat = [Counter(words.split()) for words in sentence]

        # Menghitung IDF (Inverse Document Frequency)
        def calculate_tf_idf(tf_kalimat):
            num_documents = len(tf_kalimat)
            idf_values = {}

            # Mencari setiap kata dalam setiap kalimat untuk menghitung IDF
            for tf in tf_kalimat:
                for word, count in tf.items():
                    if word in idf_values:
                        idf_values[word] += 1
                    else:
                        idf_values[word] = 1

            tf_idf_kalimat = []

            for tf in tf_kalimat:
                tf_idf = {}
                for word, count in tf.items():
                    tf_idf[word] = (count / len(tf)) * \
                        math.log(num_documents / idf_values[word])
                tf_idf_kalimat.append(tf_idf)

            return tf_idf_kalimat

        # Hasil TF-IDF
        hasil_tf_idf = calculate_tf_idf(tf_kalimat)

        # Mengonversi list of dicts ke dalam DataFrame
        df_tf_idf = pd.DataFrame(hasil_tf_idf)
        df_tf_idf.fillna(0, inplace=True)  # Mengganti nilai NaN dengan 0

        # Tampilkan hasil dalam DataFrame
        st.header("TF-IDF")
        st.write("Term Frequency-Inverse Document Frequency adalah teknik yang digunakan dalam pemrosesan bahasa alami dan pengelolaan informasi untuk mengevaluasi seberapa penting sebuah kata dalam suatu dokumen relatif terhadap kumpulan dokumen yang lebih besar. TF-IDF memberikan bobot numerik untuk setiap kata dalam dokumen, mencerminkan seberapa sering kata tersebut muncul dalam dokumen tersebut dan seberapa uniknya di antara seluruh kumpulan dokumen.")
        st.dataframe(df_tf_idf)

        # Mengonversi list dari kamus ke DataFrame pandas
        df_hub_kal = pd.DataFrame(hasil_tf_idf)
        df_hub_kal = df_tf_idf.fillna(0)  # Mengganti nilai NaN dengan 0

        # Mengonversi DataFrame ke array NumPy untuk perhitungan kesamaan kosinus
        tfidf_matrix = df_hub_kal.to_numpy()

        # Menghitung kesamaan kosinus
        similarity_matrix = cosine_similarity(tfidf_matrix)
        df_hub_kal = pd.DataFrame(similarity_matrix)

        kalimat = ["Kalimat " + str(i)
                   for i in range(1, len(similarity_matrix) + 1)]
        df_hub_kal = df_hub_kal.set_axis(kalimat, axis=0)
        df_hub_kal = df_hub_kal.set_axis(kalimat, axis=1)

        st.header("Cosine Smilarity")
        st.write("Cosine similarity adalah metode pengukuran kesamaan antara dua vektor non-nol dalam ruang vektor, terutama sering digunakan dalam konteks pemrosesan bahasa alami dan analisis teks. Metode ini membandingkan sudut antara dua vektor untuk menentukan sejauh mana vektor-vektor tersebut mirip satu sama lain.")
        st.dataframe(df_hub_kal)

        # Buat Graph
        G = nx.Graph()
        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix[0])):
                if i != j:
                    similarity = round(similarity_matrix[i][j], 2)
                    G.add_edge(i, j, weight=similarity)

        # Visualisasi grafik
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {(i, j): f"{weight:.2f}" for (
            i, j), weight in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        st.header("Visualisasi Kalimat Ke dalam Bentuk Graph")
        st.write("Graph adalah struktur data yang terdiri dari simpul-simpul (node) dan sisi-sisi (edge) yang menghubungkan simpul-simpul tersebut. Graf digunakan untuk memodelkan berbagai konsep, termasuk jaringan komputer, relasi antar entitas, dan banyak lagi.")
        plt.title("Visualisasi Graph dengan Kesamaan Kosinus (Bulatan 2 Desimal)")
        st.pyplot(plt)

        G = nx.Graph()
        threshold = 0.06  # Threshold untuk menyambungkan node

        # Tambahkan semua node ke grafik
        num_nodes = len(similarity_matrix)
        G.add_nodes_from(range(num_nodes))

        # Tambahkan edge antara node yang nilainya melebihi threshold
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and similarity_matrix[i][j] > threshold:
                    # Bulatkan nilai ke 2 angka dibelakang koma
                    similarity = round(similarity_matrix[i][j], 2)
                    G.add_edge(i, j, weight=similarity)

        # Visualisasi grafik
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)  # Menentukan layout grafik

        # Buat mapping untuk label node
        node_labels = {i: f"Kalimat {i + 1}" for i in range(num_nodes)}

        nx.draw(G, pos, with_labels=True, labels=node_labels,
                font_weight='bold')  # Menggambar grafik dengan label node
        # Mendapatkan atribut edge (bobot)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        # Mengubah nilai bobot menjadi string dengan 2 angka di belakang koma
        edge_labels = {(i, j): f"{weight:.2f}" for (
            i, j), weight in edge_labels.items()}
        # Menampilkan label bobot pada edge
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # Tambahkan label pada node yang tidak terhubung
        isolated_nodes = list(nx.isolates(G))
        if isolated_nodes:
            pos_extra = {node: (pos[node][0], pos[node][1] + 0.1)
                         for node in isolated_nodes}

        st.subheader("Visualisasi Kalimat dengan treshold")
        plt.title(
            f"Visualisasi Graph dengan Kesamaan Kosinus (Threshold: {threshold})")
        st.pyplot(plt)

        G = nx.DiGraph()  # Menggunakan Directed Graph agar dapat mengakses in_degree dan out_degree
        threshold = 0.06  # Threshold untuk menyambungkan node

        # Tambahkan semua node ke grafik
        num_nodes = len(similarity_matrix)
        G.add_nodes_from(range(num_nodes))

        # Tambahkan edge antara node yang nilainya melebihi threshold
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and similarity_matrix[i][j] > threshold:
                    # Bulatkan nilai ke 2 angka dibelakang koma
                    similarity = round(similarity_matrix[i][j], 2)
                    G.add_edge(i, j, weight=similarity)

        # Hitung indegree dan outdegree untuk setiap node
        indegree = dict(G.in_degree())
        outdegree = dict(G.out_degree())

        # Visualisasi grafik
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)  # Menentukan layout grafik

        # Buat mapping untuk label node
        node_labels = {
            i: f"Kalimat {i + 1}\nIn: {indegree[i]}, Out: {outdegree[i]}" for i in range(num_nodes)}

        nx.draw(G, pos, with_labels=True, labels=node_labels,
                font_weight='bold')  # Menggambar grafik dengan label node
        # Mendapatkan atribut edge (bobot)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        # Mengubah nilai bobot menjadi string dengan 2 angka di belakang koma
        edge_labels = {(i, j): f"{weight:.2f}" for (
            i, j), weight in edge_labels.items()}
        # Menampilkan label bobot pada edge
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # Tambahkan label pada node yang tidak terhubung
        isolated_nodes = list(nx.isolates(G))
        if isolated_nodes:
            pos_extra = {node: (pos[node][0], pos[node][1] + 0.1)
                         for node in isolated_nodes}

        st.subheader("menampilkan Indegree dan Outdegree")
        plt.title(
            f"Visualisasi Graph dengan Kesamaan Kosinus (Threshold: {threshold})")
        st.pyplot(plt)

        # Menghitung closeness centrality dari graph
        closeness = nx.closeness_centrality(G)

        # Menampilkan closeness centrality
        st.header("Closeness Centrality")
        st.write('''Closeness centrality adalah sebuah metrik dalam analisis jaringan yang mengukur sejauh mana suatu simpul (node) dalam jaringan mendekati simpul-simpul lainnya. Simpul dengan closeness centrality yang tinggi dianggap lebih dekat atau lebih 'sentral' dalam arti jarak atau aksesibilitas dalam jaringan. 
                Metrik closeness centrality dihitung berdasarkan reciprok dari total jarak dari suatu simpul ke semua simpul lainnya dalam jaringan. Semakin kecil total jarak, semakin besar nilai closeness centrality. Ini berarti simpul yang lebih dekat dengan simpul-simpul lainnya memiliki nilai closeness centrality yang lebih tinggi.''')

        for node, closeness_value in closeness.items():
            st.write(f"Node {node}: {closeness_value}")

        # Menampilkan 3 kalimat dengan PageRank tertinggi
        sorted_closeness = sorted(
            closeness.items(), key=lambda x: x[1], reverse=True)

        st.write("=============================")
        st.subheader("Top 3 sentences based on closeness centrality:")
        for node, rank in sorted_closeness[:3]:
            # Pastikan sentence telah didefinisikan sebelumnya
            st.write(sentence[node])

        st.write("=============================")
        st.subheader("Top 3 node based on closeness centrality:")
        for node, rank in sorted_closeness[:3]:
            st.write(f"Node {node} dengan Closeness Centrality {rank:.4f}")

        # Hitung PageRank
        pagerank = nx.pagerank(G)

        # Menampilkan Closeness Centrality
        st.header("PageRank:")
        st.write('''PageRank adalah algoritma yang digunakan oleh mesin pencari Google untuk memberikan peringkat pada halaman-halaman web dalam hasil pencarian. Algoritma ini dikembangkan oleh Larry Page dan Sergey Brin, pendiri Google, dan dinamai dari nama Larry Page.
                Tujuan utama PageRank adalah mengukur seberapa penting atau otoritatif sebuah halaman web dengan mempertimbangkan struktur tautan antar halaman-halaman tersebut. Ide dasar di balik PageRank adalah bahwa halaman web yang banyak dihubungkan oleh halaman-halaman lain memiliki tingkat otoritas yang lebih tinggi.''')

        for node, rank in pagerank.items():
            print(f"Kalimat {node + 1}: {rank}")

        # Menampilkan 3 kalimat dengan PageRank tertinggi
        sorted_pagerank = sorted(
            pagerank.items(), key=lambda x: x[1], reverse=True)

        st.write("=============================")
        st.subheader("Top 4 sentences based on PageRank:")
        for node, rank in sorted_pagerank[:4]:
            st.write(sentence[node + 1])

        st.write("=============================")
        st.subheader("Top 4 kalimat based on PageRank:")
        for node, rank in sorted_pagerank[:4]:
            st.write(f"Kalimat {node + 1} dengan PageRank {rank:.4f}")

# sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    main()
