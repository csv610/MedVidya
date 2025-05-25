import os
import time
import json
from datetime import datetime

from duckduckgo_search import DDGS
from typing import Dict, List, Optional
import streamlit as st
import google.generativeai as genai

@st.cache_data # Cache the loaded data
def load_clinical_procedures(filepath="medscape_clinical_procedures.json"):
    if not os.path.exists(filepath):
        st.error(f"Clinical procedures file not found at {filepath}")
        return {}
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            st.error(f"Invalid format in {filepath}. Expected a dictionary at the root.")
            return {}

        # Validate and clean the data
        valid_data = {}
        for key, topics in data.items():
            if not isinstance(topics, list):
                st.warning(f"Skipping invalid key {key}: expected list of topics, got {type(topics)}")
                continue
            
            # Convert simple strings to topic dictionaries
            valid_topics = [{"topic": topic, "see": []} for topic in topics if isinstance(topic, str)]
            
            if valid_topics:
                valid_data[key] = valid_topics

        return valid_data
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from {filepath}. Please check the file format.")
        return {}
    except Exception as e:
        st.error(f"An unexpected error occurred while loading clinical_procedures: {e}")
        return {}

def get_available_generation_models():
    """Return the specific generation models we want to use."""
    if not os.getenv('GEMINI_API_KEY'):
        return []
    
    # List of specific models we want to use
    TARGET_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.5-flash-preview-05-20"
    ]
        
    return TARGET_MODELS

def generate_text_from_model(model_name, prompt):
    """Generic function to generate text using a given model and prompt."""
    try:
        model = genai.GenerativeModel(model_name)
        start_time = time.time()
        response = model.generate_content(prompt)
        end_time = time.time()

        generated_text = ""
        if hasattr(response, 'text'):
            # Ensure markdown formatting is preserved
            generated_text = response.text.replace('#', '\n#').strip()  # Add newlines before headers
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            generated_text = f"Content generation was blocked due to: {response.prompt_feedback.block_reason}"
        elif response.candidates and response.candidates[0].finish_reason: # Check if candidates exist
            generated_text = f"Content generation finished with reason: {response.candidates[0].finish_reason}"
        else:
            generated_text = "Unable to generate content or content was empty." # Fallback for unexpected cases

        # Add newlines before markdown headers if they don't exist
        if generated_text:
            generated_text = '\n'.join(['\n# ' + line if line.startswith('#') else line 
                                     for line in generated_text.split('\n')])

        return {
            "text": generated_text,
            "generation_time": end_time - start_time,
            "text_length": len(generated_text)
        }
    except Exception as e:
        st.error(f"An error occurred during generation with model `{model_name}`: {e}")
        return None

def generate_clinical_procedure_info(model_name, clinical_procedure):
    """Generate detailed information about a specific clinical procedure."""
    prompt = f"""You are an expert medical assistant providing clear and concise clinical procedure information.

Clinical Procedure: {clinical_procedure}

Please provide detailed information for each section below. If a section is not applicable, clearly state why:

# Summary
Provide a concise overview of the procedure, including its purpose and main benefits.

# Indications
List specific medical conditions or situations where this procedure is indicated.

# Urgency of the Procedure
Describe the time sensitivity of this procedure and when it should be performed.

# Contraindications
List any medical conditions or circumstances that would make this procedure unsafe.

# Preoperative Investigations
List all necessary diagnostic tests and evaluations required before the procedure.

# Patient Preparation
Describe patient preparation steps, including fasting requirements and medication adjustments.

# Surgical Positioning
Describe the optimal patient positioning for this procedure.

# Surgical Steps
Provide a detailed, numbered list of surgical steps.

# Special Considerations
Describe any special considerations or precautions to be aware of.

# Common Complications
List and describe the most frequent complications and their management.

# Rare Complications
List and describe less common but serious complications.

# Average Time in Operation Theatre
Provide an estimated time range for the procedure.

# Success Rate
Provide statistics or percentages for successful outcomes.

# Monitoring and Follow-up
Describe the post-procedure monitoring requirements and follow-up schedule.

# Alternatives
List alternative treatment options and their relative advantages/disadvantages.

# Surgical Tools and Equipment
List all necessary surgical instruments and equipment.

# New Advancements
Describe recent technological or procedural advancements in this field.

Format your response using proper markdown headings (# for main sections, ## for subsections).
Ensure each section contains specific, actionable information.
If a section is not applicable, state 'Not applicable: [reason]' rather than leaving it empty.
"""
    
    # Generate and process the text
    result = generate_text_from_model(model_name, prompt)
    if result and 'text' in result:
        # Find the position of the Summary section
        summary_start = result['text'].find('# Summary')
        if summary_start != -1:
            # Only keep content from Summary section onwards
            return {'text': result['text'][summary_start:]}  # Slice from Summary section to end
    return {'text': ''}

def generate_answer_for_question(model_name, clinical_procedure, question):
    """Generate an answer to a specific question about a health topic."""
    prompt = f"""You are an expert medical assistant focused on clinical procedures.

Clinical Procedure: {clinical_procedure}

Question: {question}

Provide a detailed and accurate answer based on your medical knowledge."""
    return generate_text_from_model(model_name, prompt)


def get_user_answer(model_name, clinical_procedure, question):
    friendly_model_name = model_name.split('models/')[-1]
    with st.spinner(f'Finding answer about "{clinical_procedure}" using {friendly_model_name}...'):
        generation_result = generate_answer_for_question(model_name, clinical_procedure, question)
        if generation_result:
            st.subheader(f"Answer to: \"{question}\"")
            display_generation_result(generation_result['text'])
            save_qa_to_queue(
                f"Question about {clinical_procedure}: {question}",
                generation_result['text'],
                friendly_model_name
            )
            return True
    return False

def clinical_procedure_question_answer(model_name, clinical_procedure):
    """Handle the process of asking a question about a health and displaying the answer.
    
    Args:
        model_name (str): The name of the model to use for generation
        clinical_procedure (str): The name of the medical topic being asked about
    """
    # Check if we should disable the input
    button_disabled = model_name is None or clinical_procedure is None
    
    # Get user question
    user_question = get_user_question(button_disabled, clinical_procedure)
    
    if st.button("Ask Question", 
                 disabled=button_disabled or not (user_question and user_question.strip()), 
                 help="Submit your question about the selected health topic."):
        if not user_question.strip():
            st.warning("Please enter a question.")
            return
            
        if not model_name or not clinical_procedure:
            st.error("Please select both a medical topic and a model first.")
            return
            
        get_user_answer(model_name, clinical_procedure, user_question)

def display_generation_result(generated_text):
    if generated_text:
        st.markdown(generated_text)
    else:
        st.warning("Generation returned empty or no content.")


def get_user_question(button_disabled, selected_clinical_procedure):
    st.subheader("Ask a Specific Question")

    return st.text_input(
        "Enter your question about the selected medical topic:",
        key="question_input",
        disabled=button_disabled,
        label_visibility="collapsed", 
        placeholder="e.g., What are the common treatments?" if selected_clinical_procedure else "Select a medical topic first"
    )

def configure_api():
    """Configure the Google Generative AI API."""
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        st.error("API key is not configured. Set the GEMINI_API_KEY environment variable.")
    else:
        try:
            genai.configure(api_key=gemini_api_key)
        except Exception as e:
            st.error(f"Failed to configure Generative AI API: {e}")

def initialize_qa_queue():
    """Initialize the queue in session state if it doesn't exist."""
    if 'qa_queue' not in st.session_state:
        st.session_state.qa_queue = []

def save_qa_to_queue(question, answer, model_used):
    """Store the Q&A pair, and the model used."""
    initialize_qa_queue() # Ensure queue exists
    st.session_state.qa_queue.insert(0, {
        "question": question,
        "answer": answer,
        "model_used": model_used
    })

def display_qa_history():
    """Display the session's history of questions and model answers."""
    initialize_qa_queue() # Ensure queue exists
    if not st.session_state.qa_queue:
        st.info("No history yet. Select a health topic and get an answer or ask a question!")
        return

    for idx, qa in enumerate(st.session_state.qa_queue):
        st.write(f"**Entry {len(st.session_state.qa_queue) - idx}:**")
        st.write(f"   **Input/Question:** {qa['question']}")
        st.write(f"   **Model Used:** `{qa['model_used']}`")
        st.write(f"   **Answer:**")
        st.markdown(f"   {qa['answer']}") # Use markdown for better rendering of model's output
        st.divider()

def select_llm_model():
    available_models = get_available_generation_models()

    if available_models:
        # Extract the user-friendly part of the model name for display
        # e.g., "models/gemini-1.5-pro-latest" -> "gemini-1.5-pro-latest"
        display_model_names = [name.split('models/')[-1] for name in available_models]
        
        # If you want to map the display name back to the full name for genai.GenerativeModel
        model_name_map = {name.split('models/')[-1]: name for name in available_models}

        selected_display_name = st.sidebar.selectbox(
            "Choose a Gemini Model:",
            display_model_names, # Show user-friendly names
            help="Select the model you want to use for text generation."
        )
        if selected_display_name:
            llm_model = model_name_map[selected_display_name] # Get full model name for internal use
            return llm_model
    else:
        st.sidebar.warning("No text generation models available. Check API key and configuration, or model filtering.")
        return None

def select_clinical_procedure():
    # Load the clinical procedures data
    all_clinical_procedures = load_clinical_procedures()
    
    selected_topic = None
    selected_key = None
    selected_topic_name = None

    if all_clinical_procedures:
        # First select the key/category
        keys = list(all_clinical_procedures.keys())
        if keys:
            selected_key = st.sidebar.selectbox("Select a category:", ["Select a category..."] + keys)
            
            if selected_key != "Select a category...":
                topics = all_clinical_procedures.get(selected_key, [])
                
                if topics:
                    # Create list of topic names
                    topic_names = ["Select a topic..."] + [t["topic"] for t in topics]
                    selected_topic_name = st.sidebar.selectbox("Select a topic:", topic_names)
                    
                    if selected_topic_name != "Select a topic...":
                        # Find the selected topic dictionary
                        selected_topic = next((t for t in topics if t["topic"] == selected_topic_name), None)
                        
                        # If the selected topic exists
                        if selected_topic:
                            # Return all three values when a topic is selected
                            return selected_topic, selected_key, selected_topic_name
                        else:
                            # Fallback in case selected_topic is None for some reason
                            selected_topic = selected_topic_name
                else:
                    st.sidebar.info(f"No topics found in category '{selected_key}'.")
        else:
            st.sidebar.info("No categories found in the data.")
    else:
        st.sidebar.info("No clinical procedures data loaded.")
    
    # Return all three values, even if None
    return selected_topic, selected_key, selected_topic_name

def configure_sidebar():
    """Configure and display the sidebar elements, returning user selections."""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Model selection
        selected_model = select_llm_model()
        
        # Topic selection
        selected_topic, selected_key, selected_clinical_procedure = select_clinical_procedure()
        
        # Video count selection
        video_count = st.slider(
            "Number of videos to show",
            min_value=1,
            max_value=100,
            value=5,
            step=1
        )
        
        # History toggle
        show_history = st.checkbox("Show History", value=False)
        
        return {
            "model": selected_model,
            "clinical_procedure": selected_clinical_procedure,
            "show_history": show_history,
            "video_count": video_count
        }

def get_clinical_procedure_info(model_name, clinical_procedure):
    """Generate detailed information about a specific clinical procedure.
    
    Args:
        model_name (str): The name of the model to use for generation
        clinical_procedure (str): The name of the clinical procedure
        
    Returns:
        str: The generated markdown text containing detailed information
    """
    try:
        # Generate clinical procedure information
        info = generate_clinical_procedure_info(model_name, clinical_procedure)
        if info:
            return info
    except Exception as e:
        st.error(f"Error getting clinical procedure information: {str(e)}")
    return ""

def search_clinical_videos(procedure_name, max_results):
    try:
        video_urls = []
        with DDGS() as ddgs:
            # Search using the procedure name
            search_query = f"{procedure_name}"
            st.write(f"Searching for videos with keywords: {search_query}")
            
            for r in ddgs.videos(keywords=search_query, max_results=max_results):
                video_url = r.get('url')
                if video_url:
                    # Validate URL
                    if not video_url.startswith(('http://', 'https://')):
                        continue
                    
                    video_urls.append({
                        'url': video_url,
                        'title': r.get('title', 'No title'),
                        'duration': r.get('duration', 'N/A')
                    })
                    
                    # Break if we have enough results
                    if len(video_urls) >= max_results:
                        break
        
        if not video_urls:
            st.warning(f"No video results found for '{procedure_name}'. Try searching with different keywords.")
        else:
            st.success(f"Found {len(video_urls)} video(s) related to '{procedure_name}'")
        
        return video_urls
    except Exception as e:
        st.error(f"Error searching for videos: {str(e)}")
        return []

def convert_duration_to_seconds(duration_str):
    """Convert duration string (e.g., '1:23' or '2:30:45') to seconds."""
    try:
        if ':' in duration_str:
            parts = duration_str.split(':')
            if len(parts) == 2:  # MM:SS format
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            elif len(parts) == 3:  # HH:MM:SS format
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
        return 0  # Return 0 if unable to parse
    except:
        return 0

def display_video_results(video_results):
    if not video_results:
        st.warning("No video results to display")
        return
    
    st.header("Related Videos")
    # Sort videos by duration (shortest first)
    sorted_videos = sorted(video_results, key=lambda v: convert_duration_to_seconds(v['duration']))
    
    for idx, video in enumerate(sorted_videos, 1):
        st.write(f"**Video {idx}:**")
        try:
            st.video(video['url'])
        except Exception as e:
            st.error(f"Error displaying video {idx}: {str(e)}")
            continue
        
        st.write(f"**Title:** {video['title']}")
        st.write(f"**Duration:** {video['duration']}")
        st.markdown(f"**[Watch Video]({video['url']})**", unsafe_allow_html=True)
        st.divider()

def display_main_content_area(selected_model_name, selected_clinical_procedure, show_history, video_count):
    """Handles the display and interactions in the main application area."""
    # Clear stored information and videos if any selection changes
    if selected_model_name != st.session_state.get('previous_model_name') or \
       selected_clinical_procedure != st.session_state.get('previous_procedure'):
        # Clear procedure information
        st.session_state.pop('procedure_info', None)
        st.session_state.pop('procedure_info_displayed', None)
        # Clear video results
        st.session_state.pop('video_results', None)
        st.session_state.pop('videos_displayed', None)
        # Update previous selections
        st.session_state['previous_model_name'] = selected_model_name
        st.session_state['previous_procedure'] = selected_clinical_procedure

    if selected_clinical_procedure:
        st.write(f"# Clinical Procedure: {selected_clinical_procedure}")
    else:
        st.info("Please select a clinical procedure from the sidebar to get information or ask questions.")
        return

    # Create a fixed header with buttons
    with st.container():
        if st.button("Get Clinical Procedure Info", key="get_info_btn", 
                        help="Get videos and information about the selected clinical procedure."):
            friendly_model_name = selected_model_name.split('models/')[-1]
            with st.spinner(f"Getting information about '{selected_clinical_procedure}'..."):
                generation_result = get_clinical_procedure_info(selected_model_name, selected_clinical_procedure)
                if generation_result:
                    # Store the information in session state
                    st.session_state['procedure_info'] = generation_result['text']
                    st.session_state['procedure_info_displayed'] = True
                    save_qa_to_queue(
                        f"Information about {selected_clinical_procedure}", 
                        "Videos and information retrieved successfully",
                        friendly_model_name
                    )
        
        if st.button("Search Videos", key="search_videos_btn_main", 
                    help="Search for videos related to the selected clinical procedure."):
            with st.spinner(f"Searching for videos related to '{selected_clinical_procedure}'..."):
                video_results = search_clinical_videos(selected_clinical_procedure, video_count)
                if video_results:
                    # Store video results in session state
                    st.session_state['video_results'] = video_results
                    st.session_state['videos_displayed'] = True

    # Add a divider after the buttons
    st.divider()

    # Display stored procedure information if available
    if 'procedure_info' in st.session_state and st.session_state.get('procedure_info_displayed', False):
        markdown_text = st.session_state['procedure_info'].strip()
        sections = markdown_text.split('#')
        for section in sections:
            if section.strip():
                section = '#' + section
                heading, content = section.split('\n', 1)
                
                if heading.startswith('##'):
                    st.subheader(heading.strip().replace('##', '').strip())
                else:
                    st.header(heading.strip().replace('#', '').strip())
                
                if content.strip():
                    st.markdown(content.strip())

        # Display stored video results if available
        if 'video_results' in st.session_state and st.session_state.get('videos_displayed', False):
            display_video_results(st.session_state['video_results'])

    # Display existing content
    if selected_clinical_procedure:
        clinical_procedure_question_answer(selected_model_name, selected_clinical_procedure)

    # Display History (if toggled)
    if show_history:
        st.write("---")
        st.subheader("Interaction History")
        display_qa_history()

def app_clinical_procedure():
    st.set_page_config(page_title="Clinical Procedure Explorer", layout="wide")
    st.title("⚕️ Clinical Procedure Explorer")

    # Initial configurations
    configure_api()
    initialize_qa_queue()

    sidebar_selections = configure_sidebar()
    
    if not isinstance(sidebar_selections, dict):
        st.error("Error: Sidebar selections not returned as dictionary")
        return
        
    selected_model= sidebar_selections.get("model")
    selected_procedure = sidebar_selections.get("clinical_procedure")
    show_history = sidebar_selections.get("show_history", False)
    video_count = sidebar_selections.get("video_count", 5)
    
    if not selected_model or not selected_procedure:
        st.error("Please select both model and clinical procedure")
        return
    
    display_main_content_area(selected_model,  selected_procedure, show_history, video_count)

if __name__ == "__main__":
     app_clinical_procedure()
