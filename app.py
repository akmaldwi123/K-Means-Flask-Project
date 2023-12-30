from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Define a global variable for the dataframe
df = pd.DataFrame()

# Define a function to categorize clusters based on characteristics
def categorize_cluster(row):
    math_score = row['Math']
    english_score = row['English']
    science_score = row['Science']

    # Definisikan standar nilai rendah, biasa, dan tinggi sesuai kebutuhan Anda
    low_threshold = 70
    high_threshold = 80

    if math_score >= high_threshold and english_score >= high_threshold and science_score >= high_threshold:
        return 'High Math & English & Science'
    elif math_score >= high_threshold and english_score >= high_threshold and science_score < high_threshold:
        return 'High Math & English, Low Science'
    elif math_score < high_threshold and english_score >= high_threshold and science_score >= high_threshold:
        return 'Low Math, High English & Science'
    elif math_score >= high_threshold and english_score < high_threshold and science_score >= high_threshold:
        return 'High Math, Low English & High Science'
    elif math_score < high_threshold and english_score < high_threshold and science_score >= high_threshold:
        return 'Low Math & English, High Science'
    elif math_score >= high_threshold and english_score < high_threshold and science_score < high_threshold:
        return 'High Math, Low English & Low Science'
    elif math_score < high_threshold and english_score >= high_threshold and science_score < high_threshold:
        return 'Low Math & High English, Low Science'
    elif math_score < high_threshold and english_score < high_threshold and science_score < high_threshold:
        return 'Low Math, Low English, Low Science'
    else:
        return 'Undefined Cluster'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        # Read values from the form
        math_score = float(request.form['math'])
        english_score = float(request.form['english'])
        science_score = float(request.form['science'])
        student_name = request.form['name']

        # Read data from the CSV file
        file_path = 'student_data.csv'
        df = pd.read_csv(file_path)

        # Add new student data
        new_data = pd.DataFrame({'Name': [student_name], 'Math': [math_score], 'English': [english_score], 'Science': [science_score]})
        df = pd.concat([df, new_data], ignore_index=True)

        # Perform K-Means Clustering with different values of k
        distortions = []
        K_range = range(1, 7)  # You can adjust the range as needed
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df[['Math', 'English', 'Science']])
            distortions.append(kmeans.inertia_)

        # Find the optimal k using the Elbow Method
        optimal_k = 6  # You can replace this with your own logic to find the optimal k

        # Plot the Elbow Method graph
        plt.plot(K_range, distortions, marker='o')
        plt.title('Elbow Method For Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Distortion')
        img_data = BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        plt.close()

        # Convert the image to a format that can be displayed in HTML
        img_base64_elbow = base64.b64encode(img_data.read()).decode('utf-8')

        # Perform K-Means Clustering with the optimal number of clusters from the Elbow Method
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        df['cluster'] = kmeans.fit_predict(df[['Math', 'English', 'Science']]) + 1  # Add 1 to start clusters from 1

        # Determine cluster names based on characteristics
        df['cluster_name'] = df.apply(categorize_cluster, axis=1)

        # Plot clustering results
        plt.scatter(df['Math'], df['English'], c=df['cluster'], cmap='rainbow')
        plt.title('K-Means Clustering')
        plt.xlabel('Math Scores')
        plt.ylabel('English Scores')
        img_data = BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        plt.close()

        # Convert the image to a format that can be displayed in HTML
        img_base64_result = base64.b64encode(img_data.read()).decode('utf-8')

        # Prepare data for display on the result page
        clustered_data = df[['Name', 'Math', 'English', 'Science', 'cluster_name']].values.tolist()
        cluster_assignments = df['cluster'].values.tolist()

        # Create a dictionary to map cluster numbers to characteristics
        cluster_characteristics = {
            1: 'High Math & English & Science',
            2: 'High Math, Moderate English, High Science',
            3: 'Balanced Performance',
            4: 'Low Math, High English, Low Science',
            5: 'Low Math & English, High Science',
            6: 'Low Math & English, Low Science'
        }

        # Map cluster numbers to characteristics
        df['cluster_characteristics'] = df['cluster'].map(cluster_characteristics)

        return render_template('result.html', elbow_plot_url=img_base64_elbow, plot_url=img_base64_result, clustered_data=clustered_data, cluster_assignments=cluster_assignments, optimal_k=optimal_k, cluster_characteristics=df['cluster_characteristics'].tolist())

    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
