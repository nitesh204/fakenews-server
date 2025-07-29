# data_fetcher.py
import pandas as pd
import random
from datetime import datetime

class DataFetcher:
    def __init__(self, path="news_dataset.csv", ai_model=None):
        self.df = pd.read_csv(path, encoding="cp1252", on_bad_lines="skip")
        # Keep only needed columns
        self.df = self.df[['Eng_Trans_Statement', 'Eng_Trans_News_Body', 'Publish_Date', 
                        'Language', 'Platform', 'Region', 'Media_Link', 'Label']]
        # Drop rows without text or label
        self.df = self.df.dropna(subset=['Eng_Trans_Statement', 'Label'])
        # Convert dates
        self.df['Publish_Date'] = pd.to_datetime(self.df['Publish_Date'], errors='coerce')
        self.df = self.df.fillna('')  # avoid NaN in optional fields
        if ai_model:
            print("Precomputing AI predictions for all posts... (one-time)")
            self.df['ai_prediction'] = self.df.apply(
                lambda row: ai_model.predict(f"{row['Eng_Trans_Statement']} {row['Eng_Trans_News_Body']}"),
                axis=1
            )

    def get_random_posts(self, count=10):
        """Simulate fetching random posts."""
        sample = self.df.sample(count)
        posts = []
        for _, row in sample.iterrows():
            posts.append({
                "title": row['Eng_Trans_Statement'],
                "body": row['Eng_Trans_News_Body'],
                "date": row['Publish_Date'].strftime('%Y-%m-%d') if pd.notnull(row['Publish_Date']) else '',
                "platform": row['Platform'],
                "region": row['Region'],
                "image": row['Media_Link'],
                "label": row['Label']
            })
        return posts

if __name__ == "__main__":
    fetcher = DataFetcher("news_dataset.csv")
    posts = fetcher.get_random_posts(5)
    for post in posts:
        print(post)
        print("-" * 40)
