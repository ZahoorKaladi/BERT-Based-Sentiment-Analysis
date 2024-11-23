import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import requests
from requests.exceptions import SSLError, ConnectionError, Timeout, RequestException
import time

def render():
    def show_progress(progress_bar, current_page, total_pages):
        progress_percentage = current_page / total_pages  # Normalize the progress value
        progress_bar.progress(progress_percentage)  # Update progress bar

    # Function to scrape reviews or articles from a given URL
    def scrape_data(url, pages, progress_bar, tag, attribute, class_name):
        data = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        with st.spinner('Data Scraping, please wait...'):
            for i in range(1, pages + 1):
                show_progress(progress_bar, i, pages)  # Update progress bar
                time.sleep(0.1)
                # Construct the paginated URL
                paginated_url = f"{url}/page/{i}"
                
                try:
                    # Scrape the content
                    response = requests.get(paginated_url, headers=headers, timeout=10)
                    response.raise_for_status()  # Raise an HTTPError for bad responses

                    content = BeautifulSoup(response.content, "html.parser")
                    for item in content.find_all(tag, {attribute: class_name}):
                        data.append(item.get_text(strip=True))
                
                except SSLError:
                    st.error(f"SSL error occurred while fetching page {i}. Please check your connection or the website's SSL certificate.")
                    break  # Stop scraping if an SSL error occurs
                except ConnectionError:
                    st.error(f"Connection error occurred while fetching page {i}. Please check your internet connection.")
                    break  # Stop scraping if there's a connection error
                except Timeout:
                    st.error(f"Request timed out while fetching page {i}. Please try again later.")
                    break  # Stop scraping if the request times out
                except RequestException as e:
                    st.error(f"An error occurred while fetching page {i}: {e}")
                    break  # Stop scraping if there's any other request error

        return data

    # Main function to build the Streamlit app
    st.title("Web Scraping Tool")
    st.write("Scrape Sentimental Textual Data")

    # Choose website category
    website_category = st.selectbox(
        "Choose the category of website you want to scrape:",
        ["Airways Reviews", "Blogs", "News", "Product Reviews (Amazon/Daraz)", "Custom URL"]
    )
    
    # Input the number of pages to scrape
    pages = st.number_input("Enter the number of pages to scrape", min_value=1, max_value=50, value=10)
    
    # Add a progress bar
    progress_bar = st.progress(0)
    
    # Select URL and scraping parameters based on website category
    if website_category == "Airways Reviews":
        base_url = st.text_input("Enter the Airways URL to scrape", value="https://www.airlinequality.com/airline-reviews/british-airways")
        tag, attribute, class_name = "div", "class", "text_content"
    elif website_category == "Blogs":
        base_url = st.text_input("Enter the blog URL to scrape", value="https://example.com/blog")
        tag, attribute, class_name = "article", "class", "blog-entry"
    elif website_category == "News":
        base_url = st.text_input("Enter the news site URL to scrape", value="https://example.com/news")
        tag, attribute, class_name = "div", "class", "news-article"
    elif website_category == "Product Reviews (Amazon/Daraz)":
        base_url = st.text_input("Enter the product page URL to scrape", value="https://example.com/product-reviews")
        tag, attribute, class_name = "span", "class", "review-text"
    else:
        base_url = st.text_input("Enter the custom URL to scrape")
        tag = st.text_input("Enter the HTML tag (e.g., div, span, article)", value="div")
        attribute = st.text_input("Enter the HTML attribute (e.g., class, id)", value="class")
        class_name = st.text_input("Enter the value of the HTML attribute", value="text-content")

    st.write("Click the button below to start scraping.")
    
    # Start scraping when the user clicks the button
    if st.button("Scrape Data"):
        st.write(f"Scraped Data from {base_url}")
        
        # Scrape the data using the custom or predefined function
        scraped_data = scrape_data(base_url, pages, progress_bar, tag, attribute, class_name)
        
        # Display and save the scraped data
        if scraped_data:
            df = pd.DataFrame(scraped_data, columns=["Scraped Data"])
            df.to_csv(f"scraped_data_{website_category.replace(' ', '_')}.csv", index=False)
            st.success(f"Scraping completed successfully from {base_url}!")
            st.write(df)
            
            # Download as CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f'{base_url.lower().replace(" ", "_")}_scraped_data.csv',
                mime='text/csv'
            )
        else:
            st.warning("No data scraped. Please check the URL or website structure.")
    
    # Adding some instructions for users
    st.write("""
    **Instructions:**
    1. Select the category of the website you want to scrape.
    2. Input the number of pages to scrape.
    3. For custom URLs, specify the HTML tag, attribute, and class or ID to locate the textual data.
    4. Click 'Scrape Data' to start scraping.
    5. The scraped data will be saved in a CSV file and displayed below.
    6. Ensure the file contains a 'content' column for analyzing 
    """)


