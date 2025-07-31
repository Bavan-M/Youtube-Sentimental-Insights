import io
import logging
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import re
import pickle
import matplotlib.dates as mdates
import boto3
import matplotlib
matplotlib.use('Agg')

from flask import Flask,jsonify,request,send_file
from flask_cors import CORS
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
from io import BytesIO

app=Flask(__name__)
CORS(app)

def preprocess_comment(comment):
    try:
        comment=comment.lower()
        comment=comment.strip()
        comment=re.sub('\n',' ',comment)
        comment=re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words=set(stopwords.words('english'))-{'not','but','however','yet','no'}
        comment=' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer=WordNetLemmatizer()
        comment=' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error occured during the preprocesing the comment {e}")
        raise

def load_model_and_vectorizer(model_name,model_version,vectorizer_path):
    try:

        mlflow.set_tracking_uri("http://xxx-xx-xxx-xxx-xxx.xx-xxxx-x.compute.amazonaws.com:5000/")
        model_uri="s3://__________________________________________/artifacts/lgbm_model/"
        print("Loading model from:", model_uri)
        model=mlflow.pyfunc.load_model(model_uri)
        
        print(model_uri)
        if vectorizer_path.startswith("s3://"):
            s3=boto3.client("s3")
            bucket_name=vectorizer_path.split('/')[2]
            key = '/'.join(vectorizer_path.split('/')[3:]) 
            response=s3.get_object(Bucket=bucket_name,Key=key)                                                                                               
            vectorizer=pickle.load(BytesIO(response["Body"].read()))
        else:
            with open(vectorizer_path,"rb") as file:
                vectorizer=pickle.load(file)
        return model,vectorizer
    except Exception as e:
        print(f"Error occured while loading the model and vectorizer {e}")
        raise e

model,vectorizer=load_model_and_vectorizer(
    model_name="youtube_chromes_plugin_model1",
    model_version="1",
    vectorizer_path="s3://_________________________________________/artifacts/tfidf_vectorizer.pkl"
    
)


@app.route('/')
def home():
    return "Welcome to flask api"


label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()

    comments = data.get('comments')
    if not comments:
        return jsonify({"error": "No 'comments' provided."}), 400

    if isinstance(comments, str):
        comments = [comments]
    elif not isinstance(comments, list):
        return jsonify({"error": "'comments' should be a string or a list of strings."}), 400

    try:
        preprocessed_comments = [preprocess_comment(comment=c) for c in comments]
        transformed = vectorizer.transform(preprocessed_comments)
        dense_comments = transformed.toarray()
        df_input = pd.DataFrame(dense_comments, columns=vectorizer.get_feature_names_out())
        predictions = model.predict(df_input)
        result = [{"comment": comment,"sentiment": int(sentiment),"label": label_map.get(int(sentiment), "Unknown")} for comment, sentiment in zip(comments, predictions)]
        return jsonify(result), 200

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]
        preprocessed_comments = [preprocess_comment(comment=c) for c in comments]
        transformed = vectorizer.transform(preprocessed_comments)
        dense_comments = transformed.toarray()
        df_input = pd.DataFrame(dense_comments, columns=vectorizer.get_feature_names_out())
        predictions = model.predict(df_input)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [
    {
        "comment": comment,
        "sentiment": int(sentiment),  # 👈 ensure native int
        "timestamp": timestamp
    }
    for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
]

    return jsonify(response)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)




