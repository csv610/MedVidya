import streamlit as st
import re
import os
import requests
import threading
from io import BytesIO

from PIL import Image
from duckduckgo_search import DDGS

# Use full page width
st.set_page_config(layout="wide")

class DuckDuckImages:

    def fetch_image_size(self, url):
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content))
            return img.size  # Returns a tuple (width, height)
        except Exception:
            return (0, 0)  # Return a tuple indicating an error

    def get_urls(self, query, size, max_results):
        image_urls = set()  
        with DDGS() as ddgs:
            for r in ddgs.images(keywords=query, max_results=max_results, size=size if size else None):
                url = r['image']
                if self.is_valid_image_url(url):  
                    image_urls.add(url)  
                    if len(image_urls) >= max_results:
                        break
        return list(image_urls)  

    def download_image(self, url, save_dir, query, index):
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content))
            ext = img.format.lower()
            safe_query = "_".join(query.lower().split())
            filename = f"{safe_query}_{index+1}.{ext}"
            filepath = os.path.join(save_dir, filename)
            img.save(filepath)
            return filename
        except Exception as e:
            return None

    def is_valid_image_url(self, url):
        try:
            response = requests.head(url, allow_redirects=True)
            return response.headers['Content-Type'].startswith('image/')
        except:
            return False

class RenderImages:
    def __init__(self):
        self.image_searcher = DuckDuckImages()
        if "last_query" not in st.session_state:
            st.session_state.last_query = ""
        # Initialize object_images in session state
        if "object_images" not in st.session_state:
            st.session_state.object_images = {}

    def display_image(self, query, url):
        width, height = self.image_searcher.fetch_image_size(url)
        if width == 0 and height == 0:
            st.write(f"Could not retrieve image size for {url}.")
            return
        st.image(url, caption=f"Image - Size: {width}x{height}", use_container_width=self.fit_image)
        if st.button(f"Remove Image", key=query+url):
            for query, urls in st.session_state.object_images.items():
                if url in urls:
                    st.session_state.object_images[query].remove(url)
            st.rerun()
        st.divider()
            
    def display_all_images(self):
        for query, urls in st.session_state.object_images.items():
            for url in urls:
                self.display_image(query, url)
            
    def fetch_object_images(self, query, image_size, num_images):
        st.write(f"Downloading images for {query} ")
        with st.spinner(f"Fetching images for '{query}'..."):
            new_urls = self.image_searcher.get_urls(query, image_size, num_images)
            for url in new_urls:
                self.display_image(query, url)
            st.session_state.object_images[query] = new_urls

    def fetch_all_images(self, query_input, image_size, num_images):
        queries = [q.strip() for q in query_input.split(",") if q.strip()]
        if query_input != st.session_state.last_query:
            st.session_state.last_query = query_input
            st.session_state.object_images = {}

        for q in queries:
            self.fetch_object_images(q, image_size, num_images)
        
        st.write("All images feteched")
        st.rerun()  

    def save_images(self):
        # Create directory for saving images
        os.makedirs("downloaded_images", exist_ok=True)

        # Check if there are any images to save
        if not st.session_state.object_images:
            st.write("No images to save.")
            return

        for query, urls in st.session_state.object_images.items():
            for i, url in enumerate(urls):
                filename = self.image_searcher.download_image(url, "downloaded_images", query, i)
                if filename:
                    st.write(f"Saved: {filename}")

    def get_sidebar_options(self):
        size = st.sidebar.selectbox(
            "Select image size:",
            options=["Large", "Medium", "Small",  "Wallpaper"],
            index=0
        )
        max_results = st.sidebar.number_input("images per item:", min_value=1, value=5, step=1)
        self.fit_image = st.sidebar.checkbox("Fit Image", value=False)

        # Store the selected size in session state
        st.session_state.image_size = size
        return size, max_results

    def render(self, med_image):
        image_size, max_images_per_item = self.get_sidebar_options()

        # Check if the image size has changed
        if 'image_size' not in st.session_state or st.session_state.image_size != image_size:
            st.session_state.object_images = {}  # Clear previous images if size changes
            st.session_state.image_size = image_size  # Update the stored image size

        #Image Search
        if st.button("Image Search"):
            st.session_state.query_input =  med_image
            self.fetch_all_images(med_image, image_size, max_images_per_item)
                    
        self.display_all_images()

        #Images download and save
        if st.session_state.object_images and st.sidebar.button("Save Images"):
            self.save_images()

if __name__ == "__main__":
    st.title("DuckDuckGo Image Search")
    med_image = st.text_input("Search Medical Images: ", "Give an image of lazy eye")
    app = RenderImages()
    app.render(med_image)
