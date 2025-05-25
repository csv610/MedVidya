import streamlit as st
from duckduckgo_search import DDGS
import re

# Configure Streamlit page
st.set_page_config(page_title="DuckDuckGo Video Finder", layout="wide")

class DuckDuckVideos:
    def get_urls(self, query, max_results=10): 
        video_urls = []
        try:
            with DDGS() as ddgs:
                for result in ddgs.videos(keywords=query, max_results=max_results):
                    url = result.get('url') or result.get('content') or result.get('image')
                    if not url:
                        continue
                    video_urls.append({
                        'url': url,
                        'title': result.get('title', 'No title'),
                        'duration': result.get('duration', 'N/A')
                    })
                    if len(video_urls) >= max_results:
                        break
        except Exception as e:
            st.error(f"Error during video search: {e}")
        return self.sort_by_duration(video_urls)

    def _duration_to_seconds(self, duration_str):
        if duration_str == 'N/A':
            return float('inf')
        match = re.match(r'(?:(\d+):)?(\d+):?(\d+)?', duration_str)
        if not match:
            return float('inf')
        parts = [int(p) if p else 0 for p in match.groups()]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return float('inf')

    def sort_by_duration(self, videos):
        return sorted(videos, key=lambda x: self._duration_to_seconds(x['duration']))

class StVideoSearch:
    def __init__(self):
        self.video_searcher = DuckDuckVideos()
        st.session_state.setdefault("videos", [])
        st.session_state.setdefault("selected_title", "")
        st.session_state.setdefault("max_results", 10)

    def search_videos(self, title, max_results):
        if title.strip():
            st.session_state.selected_title = title.strip()
            st.session_state.max_results = max_results
            with st.spinner("Searching for videos..."):
                st.session_state.videos = self.video_searcher.get_urls(st.session_state.selected_title, max_results=st.session_state.max_results)
        else:
            st.warning("Please enter a valid search query.")

    def show_videos(self):
        if st.session_state.videos:
            updated_videos = st.session_state.videos.copy()
            for i, video in enumerate(st.session_state.videos):
                if i >= st.session_state.max_results:
                    break
                with st.container():
                    st.markdown(f"### VideoID: {i}")
                    st.markdown(f"### Title: {video['title']}")
                    st.markdown(f"### Duration: {video['duration']}")
                    try:
                        st.video(video['url'])
                    except Exception:
                        st.warning("This video cannot be embedded. Please use the link above.")

                    st.divider()

                    if st.button(f"Remove Video {i}"):
                        updated_videos.pop(i)
                        st.session_state.videos = updated_videos
                        st.rerun()

class UIVideoApp:
    def __init__(self):
        self.app = StVideoSearch()

    def run(self):
        st.title("🔎 DuckDuckGo Video Finder")

        # Step 1: Input title
        title = st.text_input("Enter a video title to search:", value=st.session_state.selected_title)

        # Step 2: Select number of results
        max_results = st.slider("Select number of videos to retrieve:", min_value=1, max_value=50, value=st.session_state.max_results)

        # Step 3: Search for videos
        if st.button("Search"):
           self.app.search_videos(title, max_results)

        self.app.show_videos()

if __name__ == "__main__":
    app = UIVideoApp()
    app.run()
