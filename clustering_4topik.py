# import pandas as pd
# from sklearn.cluster import KMeans
# import streamlit as st
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from yellowbrick.cluster import SilhouetteVisualizer


# def main():
#     st.title("Halaman Clustering dengan 4 Fitur")
#     st.write("4 Topik di clustering menjadi 2 label")

#     dt = pd.read_excel("HasilPreposPTA.xlsx")
#     data = dt['cleaning']

#     # Membuat DataFrame dari data teks
#     dt_lda = pd.DataFrame(data)

#     dt_lda['cleaning'] = dt_lda['cleaning'].fillna('')

#     # mengonversi teks menjadi matriks hitungan
#     vectorizer = CountVectorizer()
#     count_matrix = vectorizer.fit_transform(dt_lda['cleaning'])

#     # model LDA
#     k = 4
#     alpha = 0.1
#     beta = 0.2

#     lda = LatentDirichletAllocation(
#         n_components=k, doc_topic_prior=alpha, topic_word_prior=beta, random_state=42)
#     lda.fit(count_matrix)

#     # distribusi topik pada setiap dokumen
#     doc_topic_distribution = lda.transform(count_matrix)

#     topic_names = [f"Topik {i+1}" for i in range(k)]
#     df = pd.DataFrame(columns=['Abstrak'] + topic_names)

#     for i, topic_name in enumerate(topic_names):
#         df[topic_name] = doc_topic_distribution[:, i]

#     # Menampilkan DataFrame
#     df['Abstrak'] = dt_lda['cleaning'].values

#     # Menambahkan kolom berisikan jumlah total proporsi semua topik
#     df['Total Proporsi Topik'] = df[topic_names].sum(axis=1)

#     # Menyimpan DataFrame sebagai file CSV
#     output_csv_file = "empat_topik_in_document.csv"
#     df.to_csv(output_csv_file, index=False)

#     st.dataframe(df)

#     # Baca data dari file CSV
#     data = pd.read_csv('empat_topik_in_document.csv')

#     # Pilih fitur untuk clustering
#     selected_data = data[['Topik 1', 'Topik 2', 'Topik 3', 'Topik 4']]

#     # Inisialisasi dan latih model K-Means
#     kmeans = KMeans(n_clusters=2)
#     kmeans.fit(selected_data)

#     # Mendapatkan label kelompok
#     labels = kmeans.labels_

#     # silhoutte skor
#     silhouette_viz = SilhouetteVisualizer(kmeans)

#     # Simpan hasil clustering dan data lainnya dalam bentuk CSV
#     data.drop('Total Proporsi Topik', axis=1, inplace=True)
#     data['Judul'] = dt['Judul']
#     data['Cluster'] = labels
#     data.to_csv('hasil_clustering2.csv', index=False)

#     # Tampilkan hasil clustering
#     st.markdown("## Hasil Clustering")
#     hasil_clustering = pd.read_csv('hasil_clustering2.csv')
#     st.dataframe(hasil_clustering)

#     silhouette_viz.fit(selected_data)
#     silhouette_viz.show()


# if __name__ == "__main__":
#     main()

import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score  # Tambahkan impor ini
from yellowbrick.cluster import SilhouetteVisualizer


def main():
    st.title("Halaman Clustering dengan 4 Fitur")
    st.write("4 Topik di clustering menjadi 2 label")

    dt = pd.read_excel("HasilPreposPTA.xlsx")
    data = dt['cleaning']

    # Membuat DataFrame dari data teks
    dt_lda = pd.DataFrame(data)

    dt_lda['cleaning'] = dt_lda['cleaning'].fillna('')

    # mengonversi teks menjadi matriks hitungan
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(dt_lda['cleaning'])

    # model LDA
    k = 4
    alpha = 0.1
    beta = 0.2

    lda = LatentDirichletAllocation(
        n_components=k, doc_topic_prior=alpha, topic_word_prior=beta, random_state=42)
    lda.fit(count_matrix)

    # distribusi topik pada setiap dokumen
    doc_topic_distribution = lda.transform(count_matrix)

    topic_names = [f"Topik {i+1}" for i in range(k)]
    df = pd.DataFrame(columns=['Abstrak'] + topic_names)

    for i, topic_name in enumerate(topic_names):
        df[topic_name] = doc_topic_distribution[:, i]

    # Menampilkan DataFrame
    df['Abstrak'] = dt_lda['cleaning'].values

    # Menambahkan kolom berisikan jumlah total proporsi semua topik
    df['Total Proporsi Topik'] = df[topic_names].sum(axis=1)

    # Menyimpan DataFrame sebagai file CSV
    output_csv_file = "empat_topik_in_document.csv"
    df.to_csv(output_csv_file, index=False)

    st.dataframe(df)

    # Baca data dari file CSV
    data = pd.read_csv('empat_topik_in_document.csv')

    # Pilih fitur untuk clustering
    selected_data = data[['Topik 1', 'Topik 2', 'Topik 3', 'Topik 4']]

    # Inisialisasi dan latih model K-Means
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(selected_data)

    # Mendapatkan label kelompok
    labels = kmeans.labels_

    # Simpan hasil clustering dan data lainnya dalam bentuk CSV
    data.drop('Total Proporsi Topik', axis=1, inplace=True)
    data['Judul'] = dt['Judul']
    data['Cluster'] = labels
    data.to_csv('hasil_clustering2.csv', index=False)

    # Tampilkan hasil clustering
    st.markdown("## Hasil Clustering")
    hasil_clustering = pd.read_csv('hasil_clustering2.csv')
    st.dataframe(hasil_clustering)

    # Silhouette Score
    sil_score = silhouette_score(selected_data, labels)
    st.write(f"Silhouette Score: {sil_score:.2f}")


if __name__ == "__main__":
    main()
