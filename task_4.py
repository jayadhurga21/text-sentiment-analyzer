# Step 1: Download NLTK resources (only needed once)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 2: Import necessary libraries
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load dataset
df = pd.read_csv("sentiment_data.csv")

# Create analyzer
analyzer = SentimentIntensityAnalyzer()

# Define function to get sentiment
def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

# Apply to data
df["Predicted_Sentiment"] = df["Review"].apply(get_sentiment)

# Show results
print(df)
print("\nSentiment counts:")
print(df["Predicted_Sentiment"].value_counts())




import matplotlib.pyplot as plt

# Count of predicted sentiments
sentiment_counts = df['Predicted_Sentiment'].value_counts()

# Bar chart
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.show()


if sentiment_counts['positive'] > sentiment_counts['negative']:
    print("Overall sentiment is positive. Customers seem satisfied.")
else:
    print("Overall sentiment is more negative. Needs improvement.")

# STEP 5: Business insight
print("\n Insight:")
print("Use the results to improve marketing, address negative feedback, and develop better features.")
