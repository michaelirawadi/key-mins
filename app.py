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
from ml.hmm_gmm import GMM
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

UPLOAD_FOLDER = 'uploads'
os.makedirs("result", exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/test_plot')
def test_plot():
    import plotly.express as px
    df = px.data.iris()
    fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_length', color='species')
    return fig.to_html(full_html=True)

@app.route('/cluster', methods=['GET', 'POST'])
def cluster():
    
    # Get data from submitted form
    if request.method == 'POST':
        k = int(request.form.get('k', 3))
        file = request.files.get('file')
        current_file = request.form.get('current_file')
        
        if file and file.filename != "":
            filename = "dataset.csv"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            current_file = filename
        elif current_file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], current_file)
            if not os.path.exists(path):
                return "Saved file not found. Please re-upload."
        else:
            return "No file uploaded."

        df = pd.read_csv(path)
        # column_names = df.columns.tolist()

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_columns]
        column_names = df.columns.tolist()
        df = df[:1000] 

        # Check if columns are not empty
        x_column = request.form.get('x_column')
        y_column = request.form.get('y_column')
        z_column = request.form.get('z_column')
        if not x_column and column_names:
            x_column = column_names[0]
        if not y_column and len(column_names) > 1:
            y_column = column_names[1]
        if not z_column and len(column_names) > 2:
            z_column = column_names[2]

        # Preprocessing
        data = preprocess(df)

        # K-Means clustering
        model = KMeans(k=k)
        model.fit(data)
        labels = model.predict(data)
        df['cluster'] = labels

        if k > 1 and data.shape[1] >= 2 and len(set(labels)) > 1:
            silhouette = silhouette_score(data, labels)
        else:
            silhouette = None 

        # if x_column in df.columns and y_column in df.columns:
        #     plt.figure()
        #     for cluster_id in range(k):
        #         cluster_points = df.loc[np.array(labels) == cluster_id, [x_column, y_column]]
        #         plt.scatter(cluster_points[x_column], cluster_points[y_column], label=f'Cluster {cluster_id+1}')
        #     plt.title("K-Means Clustering")
        #     plt.xlabel(x_column)
        #     plt.ylabel(y_column)
        #     plt.legend()

        #     plot_filename = "plot.png"
        #     plot_path = os.path.join("result", plot_filename)
        #     plt.savefig(plot_path)
        #     plt.close()

        #     return render_template("cluster.html",
        #                            plot_url=f"plot.png?v={int(time.time())}",
        #                            silhouette=silhouette,
        #                            columns=column_names,
        #                            current_file=current_file)
        
        if x_column in df.columns and y_column in df.columns and z_column in df.columns:
            # Convert selected columns to numeric and drop non-numeric rows
            df[x_column] = pd.to_numeric(df[x_column], errors='coerce')
            df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
            df[z_column] = pd.to_numeric(df[z_column], errors='coerce')
            df_numeric = df.dropna(subset=[x_column, y_column, z_column]).copy()

            # Add cluster information for coloring
            df_numeric['cluster'] = np.array(labels).astype(str)

            # Create an interactive 3D scatter plot using Plotly
            fig = px.scatter_3d(
                df_numeric,
                x=x_column,
                y=y_column,
                z=z_column,
                color='cluster',
                title="3D K-Means Clustering",
                labels={x_column: x_column, y_column: y_column, z_column: z_column, 'cluster': 'Cluster'}
            )
            fig.update_traces(marker=dict(size=5, opacity=0.8), selector=dict(mode='markers'))
            # Convert the Plotly figure to HTML to embed in the template.
            plot_html = fig.to_html(full_html=False)

            cluster_summary = df.groupby('cluster').mean(numeric_only=True)
            print(cluster_summary)
            cluster_summary_html = cluster_summary.to_html(classes="cluster-summary", float_format="%.2f", border=0)


            return render_template("cluster.html",
                                plot_html=plot_html,
                                silhouette=silhouette,
                                columns=column_names,
                                current_file=current_file,
                                cluster_summary=cluster_summary_html)
        else:
            return "Plotting requires valid x, y and z columns"

        # return "Plotting requires valid x and y columns"

    return render_template("cluster.html", plot_url=None, silhouette=None, columns=[], current_file=None)

# @app.route('/cluster', methods=['GET', 'POST'])
# def cluster():
#     if request.method == 'POST':
#         k = int(request.form.get('k', 3))
#         file = request.files.get('file')
#         current_file = request.form.get('current_file')
        
#         if file and file.filename != "":
#             filename = "dataset.csv"
#             path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(path)
#             current_file = filename
#         elif current_file:
#             path = os.path.join(app.config['UPLOAD_FOLDER'], current_file)
#             if not os.path.exists(path):
#                 return "Saved file not found. Please re-upload."
#         else:
#             return "No file uploaded."

#         df = pd.read_csv(path)
#         column_names = df.columns.tolist()

#         x_column = request.form.get('x_column')
#         y_column = request.form.get('y_column')
#         if not x_column and column_names:
#             x_column = column_names[0]
#         if not y_column and len(column_names) > 1:
#             y_column = column_names[1]

#         data = preprocess(df)

#         model = GMM(k=k)
#         model.fit(data)
#         labels = model.predict(data)

#         if k > 1 and data.shape[1] >= 2 and len(set(labels)) > 1:
#             silhouette = silhouette_score(data, labels)
#         else:
#             silhouette = None 

#         if x_column in df.columns and y_column in df.columns:
#             plt.figure()
#             for cluster_id in range(k):
#                 cluster_points = df.loc[np.array(labels) == cluster_id, [x_column, y_column]]
#                 plt.scatter(cluster_points[x_column], cluster_points[y_column], label=f'Cluster {cluster_id+1}')
#             plt.title("GMM Clustering")
#             plt.xlabel(x_column)
#             plt.ylabel(y_column)
#             plt.legend()

#             plot_filename = "plot.png"
#             plot_path = os.path.join("result", plot_filename)
#             plt.savefig(plot_path)
#             plt.close()

#             return render_template("cluster.html",
#                                    plot_url=f"plot.png?v={int(time.time())}",
#                                    silhouette=silhouette,
#                                    columns=column_names,
#                                    current_file=current_file)
#         return "Plotting requires valid x and y columns"

#     return render_template("cluster.html", plot_url=None, silhouette=None, columns=[], current_file=None)


@app.route('/result/<filename>')
def result_file(filename):
    return send_from_directory('result', filename)

@app.route('/template')
def template():
    # return send_from_directory(directory='./dataset/', path='wine-clustering.csv', as_attachment=True)
    return send_from_directory(directory='./dataset/', path='consumer_behavior_dataset.csv', as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)