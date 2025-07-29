# app.py
from flask import Flask, jsonify, request
from data import data_fetcher
from flask_cors import CORS
from classifier.ai_classifier import FakeNewsClassifier
app = Flask(__name__)
CORS(app)  # allow React to access this API
ai_model = FakeNewsClassifier()  
fetcher = data_fetcher.DataFetcher("data/news_dataset.csv", ai_model=ai_model)

def safe_date(date_value):
    import pandas as pd
    if pd.notnull(date_value):
        return date_value.strftime('%Y-%m-%d')
    return ""

@app.route("/get_posts")
def get_posts():
    # Get query params
    platform = request.args.get("platform")  # e.g., "Twitter"
    region = request.args.get("region")      # e.g., "India"
    label = request.args.get("label")        # e.g., "True"/"False"
    search = request.args.get("search")      # e.g., "Modi"
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 10))

    # Filter data
    df = fetcher.df
    if platform:
        df = df[df['Platform'].str.contains(platform, case=False, na=False)]
    if region:
        df = df[df['Region'].str.contains(region, case=False, na=False)]
    if label:
        df = df[df['Label'].astype(str).str.lower() == label.lower()]
    if search:
        df = df[df['Eng_Trans_Statement'].str.contains(search, case=False, na=False) |
                df['Eng_Trans_News_Body'].str.contains(search, case=False, na=False)]

    # Pagination
    start = (page - 1) * limit
    end = start + limit
    df = df.iloc[start:end]

    # Prepare results
    posts = []
    for _, row in df.iterrows():
        ai_pred = ai_model.predict(row['Eng_Trans_Statement'] + " " + row['Eng_Trans_News_Body'])
        posts.append({
            "title": row['Eng_Trans_Statement'],
            "body": row['Eng_Trans_News_Body'],
            "date": safe_date(row['Publish_Date']),
            "platform": row['Platform'],
            "region": row['Region'],
            "image": row['Media_Link'],
            "label": row['Label'],
            "ai_prediction": ai_pred
            })
    return jsonify(posts)


@app.route("/get_trends")
def get_trends():
    # Count fake vs real
    df = fetcher.df
    counts = {
        "dataset_labels": df['Label'].value_counts().to_dict(),
        "platforms": df['Platform'].value_counts().to_dict(),
        "regions": df['Region'].value_counts().to_dict()
    }
    return jsonify(counts)


@app.route("/get_ai_trends")
def get_ai_trends():
    ai_counts = fetcher.df['ai_prediction'].value_counts().to_dict()
    return jsonify({
        "ai_prediction_counts": ai_counts
    })

@app.route("/get_filters")
def get_filters():
    def clean_values(series):
        cleaned = set()
        for val in series.dropna().tolist():
            for item in str(val).split(","):
                name = item.strip().title()
                if name:
                    cleaned.add(name)
        return sorted(list(cleaned))

    platforms = clean_values(fetcher.df['Platform'])
    regions = clean_values(fetcher.df['Region'])
    languages = clean_values(fetcher.df['Language'])

    return jsonify({
        "platforms": platforms,
        "regions": regions,
        "languages": languages
    })


if __name__ == "__main__":
    app.run(debug=True)
