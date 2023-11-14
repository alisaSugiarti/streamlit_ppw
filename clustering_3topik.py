import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
from sklearn.metrics import silhouette_score  # Tambahkan impor ini


def main():
    st.title("Halaman Clustering dengan 3 Fitur")
    st.write("3 Topik di clustering menjadi 2 label")

    # Baca data dari file CSV
    data = pd.read_csv('topik_in_document.csv')

    # Pilih fitur untuk clustering
    selected_data = data[['Topik 1', 'Topik 2', 'Topik 3']]

    # Inisialisasi dan latih model K-Means
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(selected_data)

    # Mendapatkan label kelompok
    labels = kmeans.labels_

    # Simpan hasil clustering dan data lainnya dalam bentuk CSV
    data.drop('Total Proporsi Topik', axis=1, inplace=True)
    data['Cluster'] = labels
    data.to_csv('hasil_clustering.csv', index=False)

    # Tampilkan hasil clustering
    st.markdown("## Hasil Clustering")
    hasil_clustering = pd.read_csv('hasil_clustering.csv')
    st.dataframe(hasil_clustering)

    # Silhouette Score
    sil_score = silhouette_score(selected_data, labels)
    st.write(f"Silhouette Score: {sil_score:.2f}")


if __name__ == "__main__":
    main()
