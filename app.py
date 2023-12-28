from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
# ...

# ...

@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        # Mengganti bagian ini untuk membaca data dari file yang sudah dideklarasikan
        file_path = 'student_data.csv'
        df = pd.read_csv(file_path)
        k = int(request.form['k'])

        # Melakukan K-Means Clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['cluster'] = kmeans.fit_predict(df[['Math', 'English', 'Science']]) + 1  # Tambahkan 1 agar cluster dimulai dari 1

        # Plot hasil clustering
        plt.scatter(df['Math'], df['English'], c=df['cluster'], cmap='rainbow')
        plt.title('K-Means Clustering')
        plt.xlabel('Math Scores')
        plt.ylabel('English Scores')

        # Simpan plot ke dalam bentuk gambar
        img_data = BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        plt.close()

        # Mengonversi gambar ke format yang dapat ditampilkan di HTML
        img_base64 = base64.b64encode(img_data.read()).decode('utf-8')

        return render_template('index.html', table=df.to_html(classes='table table-striped table-bordered table-hover'), image=img_base64)

    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')

# ...


# ...


if __name__ == '__main__':
    app.run(debug=True)
