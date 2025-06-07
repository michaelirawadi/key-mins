import os
import time
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from ml.kmeans import KMeans, rmse
from ml.data_preprocessing import preprocess

UPLOAD_FOLDER = 'uploads'
os.makedirs("result", exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/cluster', methods=['GET', 'POST'])
def cluster():
    if request.method == 'POST':
        k = int(request.form.get('k', 3))
        file = request.files['file']
        if file:
            filename = "dataset" + ".csv"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            df = pd.read_csv(path)
            column_names = df.columns.tolist()
            data = preprocess(df)

            model = KMeans(k=k)
            model.fit(data)
            labels = model.predict(data)

            if k > 1 and data.shape[1] >= 2 and len(set(labels)) > 1:
                silhouette = silhouette_score(data, labels)
            else:
                silhouette = None 

            if data.shape[1] >= 2:
                plt.figure()
                for cluster_id in range(k):
                    cluster_points = data[labels == cluster_id]
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id+1}')
                plt.title("K-Means Clustering")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                plt.legend()

                plot_filename = "plot.png"
                plot_path = os.path.join("result", plot_filename)
                plt.savefig(plot_path)
                plt.close()

                return render_template("cluster.html", plot_url=f"plot.png?v={int(time.time())}", silhouette=silhouette)

            return "Plotting requires at least 2 features"

    return render_template("cluster.html", plot_url=None, silhouette=None)

@app.route('/result/<filename>')
def result_file(filename):
    return send_from_directory('result', filename)

@app.route('/template')
def template():
    return send_from_directory(directory='./dataset/', filename='wine-clustering.csv', as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)