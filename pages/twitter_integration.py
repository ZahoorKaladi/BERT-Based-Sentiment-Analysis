import streamlit as st
import tweepy
import pandas as pd
import time
def render():
    # Function for Twitter API Integration

#def twitter_api():
    st.title("Social sites API Integration")

    st.markdown("### Choose API to Integrate")
    
    # Sidebar for navigation and API selection
    api_choice = st.selectbox(
        "Select Social Media Platform", 
        ["Twitter", "Facebook", "Instagram"], 
        key="social_media_platform_selectbox"  # Unique key for the selectbox
    )

    # Add icons for platforms
    social_icons = {
        "Twitter": "https://upload.wikimedia.org/wikipedia/commons/6/60/X_logo.svg",
        "Facebook": "https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg",
        "Instagram": "https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png",
    }
    st.image(social_icons[api_choice], width=80)

    # Set up Twitter API credentials
    twitter_api_key = st.text_input("Twitter API Key", type="password", key="unique_api_key")
    twitter_api_secret = st.text_input("Twitter API Secret", type="password", key="unique_apisecrete_key")
    twitter_access_token = st.text_input("Twitter Access Token", type="password", key="unique_apitoken_key")
    twitter_access_token_secret = st.text_input("Twitter Access Token Secret", type="password", key="unique_api_key_tokensecrete")

    # Button to proceed after entering credentials
    proceed_button = st.button("Proceed with Authentication", key="proceed_key")

    # Function to authenticate and scrape Twitter
    def authenticate_twitter():
        try:
            auth = tweepy.OAuth1UserHandler(
                consumer_key=twitter_api_key, 
                consumer_secret=twitter_api_secret,
                access_token=twitter_access_token,
                access_token_secret=twitter_access_token_secret
            )
            api = tweepy.API(auth)
            # Test authentication
            api.verify_credentials()
            st.success("Authentication successful!")
            return api
        except tweepy.TweepError as e:
            st.error(f"Authentication failed: {e}")
            return None

    # Function to scrape tweets
    def scrape_tweets(api, keyword, count=100):
        st.write("Fetching tweets...")
        tweets = api.search(q=keyword, count=count, lang="en", tweet_mode="extended")
        tweet_data = []

        for tweet in tweets:
            tweet_data.append({
                "tweet": tweet.full_text,
                "username": tweet.user.screen_name,
                "date": tweet.created_at,
            })

        return tweet_data

    # Function to save data to CSV
    def save_to_csv(data, filename="scraped_data.csv"):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        st.write(f"Data saved to {filename}")

    # Main function logic
    if api_choice == "Twitter":
        if all([twitter_api_key, twitter_api_secret, twitter_access_token, twitter_access_token_secret]):
            if proceed_button:
                with st.spinner('Verifying credentials...'):
                    api = authenticate_twitter()

                    if api:
                        st.success("Authentication successful. You can now proceed to scrape data.")

                        # Input keyword for searching tweets
                        keyword = st.text_input("Enter keyword or hashtag to search for tweets")

                        if keyword:
                            scrape_button = st.button("Scrape Tweets")

                            # Add progress bar when button is clicked
                            if scrape_button:
                                progress_bar = st.progress(0)

                                with st.spinner('Scraping tweets, please wait...'):
                                    total_steps = 10  # Define the total steps for progress bar

                                    for step in range(total_steps):
                                        time.sleep(1)  # Simulate delay for each step
                                        progress_percentage = (step + 1) / total_steps
                                        progress_bar.progress(progress_percentage)

                                    # Perform scraping
                                    tweets_data = scrape_tweets(api, keyword)
                                    save_to_csv(tweets_data, "scraped_tweets.csv")
                                    st.write("Data scraped successfully!")
                    else:
                        st.error("Authentication failed. Please check your credentials.")
        else:
            if proceed_button:
                st.error("Please enter your Twitter API credentials to proceed.")
    else:
        st.warning(f"{api_choice} API integration is coming soon!")

