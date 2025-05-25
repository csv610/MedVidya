import streamlit as st
import google.generativeai as genai
import os
import time
import json

@st.cache_data # Cache the loaded data
def load_health_topics(filepath="medlineplus_health_topics.json"):
    """
    Load health topics from a JSON file with the structure:
    [
        {"topic": "Topic Name", "see": ["Related Topic 1", "Related Topic 2"]},
        ...
    ]
    """
    if not os.path.exists(filepath):
        st.error(f"Health topics file not found at {filepath}")
        return []
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            st.error(f"Invalid format in {filepath}. Expected a list at the root.")
            return []

        # Validate each item in the list
        valid_topics = []
        for item in data:
            if (isinstance(item, dict) and 
                "topic" in item and 
                isinstance(item["topic"], str) and
                "see" in item and 
                (isinstance(item["see"], list) or isinstance(item["see"], str))):
                # Convert single string to list for consistency
                if isinstance(item["see"], str):
                    item["see"] = [item["see"]]
                valid_topics.append(item)
            else:
                st.warning(f"Skipping invalid entry in JSON: {item}")

        # Sort topics alphabetically
        valid_topics.sort(key=lambda x: x["topic"].lower())
        return valid_topics
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from {filepath}. Please check the file format.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading health_topics: {e}")
        return []

def get_available_generation_models():
    """Fetch and filter models available for text generation, focusing on current/valid ones."""
    if not os.getenv('GEMINI_API_KEY'): # Check API key before trying to list models
        return []
    try:
        all_models_info = genai.list_models()

        # Define a set of model name suffixes (part after "models/") for filtering.
        # This list is curated based on Google's documentation for generally available,
        # stable, and recommended models for generation, avoiding explicitly legacy,
        # preview, or experimental models unless they are the primary way to access a stable line.
        # As of early-mid 2025, these are strong candidates based on documentation.
        CURATED_VALID_MODEL_SUFFIXES = {
            "gemini-2.0-flash",         # Latest stable Flash (likely an alias to a specific version like -001)
            "gemini-2.5-flash",         # Latest stable Flash (likely an alias to a specific version like -001)
        }

        generation_models = []
        for m in all_models_info:
            if 'generateContent' in m.supported_generation_methods:
                model_suffix = m.name.split('models/')[-1]
                if model_suffix in CURATED_VALID_MODEL_SUFFIXES:
                    # Additional check to avoid preview/experimental if the alias itself doesn't imply it
                    if "preview" not in model_suffix and "exp" not in model_suffix:
                         generation_models.append(m.name)

        generation_models = list(set(generation_models)) # Ensure uniqueness
        generation_models.sort()

        # Fallback if the curated list results in no models
        if not generation_models:
            st.warning(
                "Curated list of valid models did not match any available models. "
                "Falling back to a broader filter for 'gemini-*latest' or main 'gemini-2.0-flash/lite' models."
            )
            broader_list = []
            for m in all_models_info:
                if 'generateContent' in m.supported_generation_methods:
                    model_suffix = m.name.split('models/')[-1]
                    is_latest_alias = "latest" in model_suffix and model_suffix.startswith("gemini-")
                    is_v2_flash_family = model_suffix in ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
                    
                    if (is_latest_alias or is_v2_flash_family) and \
                       "preview" not in model_suffix and \
                       "experimental" not in model_suffix: # Note: "exp" is sometimes used instead of "experimental"
                        broader_list.append(m.name)
            
            generation_models = list(set(broader_list))
            generation_models.sort()

        # Final fallback: if still no models, list all compatible ones (original behavior)
        if not generation_models:
            st.warning(
                "No specific current models found after filtering. Listing all available text generation models. "
                "Please check model availability and naming conventions if this list is too broad or misses expected models."
            )
            all_gen_models = [
                m.name for m in all_models_info if 'generateContent' in m.supported_generation_methods
            ]
            all_gen_models.sort()
            return all_gen_models # Return directly to avoid empty list if this is the only way
            
        return generation_models

    except Exception as e:
        st.error(f"Error fetching or filtering available models: {e}")
        return []

def generate_text_from_model(model_name, prompt):
    """Generic function to generate text using a given model and prompt."""
    try:
        model = genai.GenerativeModel(model_name)
        start_time = time.time()
        response = model.generate_content(prompt)
        end_time = time.time()

        generated_text = ""
        if hasattr(response, 'text'):
            generated_text = response.text
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            generated_text = f"Content generation was blocked due to: {response.prompt_feedback.block_reason}"
        elif response.candidates and response.candidates[0].finish_reason: # Check if candidates exist
            generated_text = f"Content generation finished with reason: {response.candidates[0].finish_reason}"
        else:
            generated_text = "Unable to generate content or content was empty." # Fallback for unexpected cases


        return {
            "text": generated_text,
            "generation_time": end_time - start_time,
            "text_length": len(generated_text)
        }
    except Exception as e:
        st.error(f"An error occurred during generation with model `{model_name}`: {e}")
        return None

def generate_health_topic_info(model_name, health_topic):
    """Generate detailed information about a specific health topic."""
    prompt = f"""You are a helpful and informative AI assistant specializing in providing clear and concise medical information.

Please provide detailed information about the following health topic based on your training data and general health knowledge. Focus on describing:
# Summary
# Symptoms
# Which Doctor(s) Should I Go to Consult.
# Causes and Risk Factors
# Diagnosis
# Future Complications
# Treatment and Theapies 
# Prevention 
# Natural/Ayurvedic Cure
# Related Issues
# New Advancements

Present the information in a clear, well-structured format using headings or bullet points.

Health Topic: {health_topic}

Information:"""
    return generate_text_from_model(model_name, prompt)

def generate_answer_for_question(model_name, health_topic, question):
    """Generate an answer to a specific question about a health topic."""
    prompt = f"""You are a helpful and informative AI assistant specializing in providing clear and concise medical information.
You are currently focused on the medical topic: {health_topic}.

Please provide a detailed and accurate answer to the following question based on your knowledge of {health_topic}:

Question: {question}

Answer:"""
    return generate_text_from_model(model_name, prompt)


def get_user_answer(model_name, health_topic, question):
    """Generate and display an answer to a specific question about a health topic.
    
    Args:
        model_name (str): The name of the model to use for generation
        health topic (str): The name of the health topic being asked about
        question (str): The user's question
        
    Returns:
        bool: True if answer was successfully generated and displayed, False otherwise
    """
    friendly_model_name = model_name.split('models/')[-1]
    with st.spinner(f'Finding answer about "{health_topic}" using {friendly_model_name}...'):
        generation_result = generate_answer_for_question(model_name, health_topic, question)
        if generation_result:
            st.subheader(f"Answer to: \"{question}\"")
            display_generation_result(generation_result['text'])
            save_qa_to_queue(
                f"Question about {health_topic}: {question}",
                generation_result['text'],
                friendly_model_name
            )
            return True
    return False

def health_question_answer(model_name, health_topic):
    """Handle the process of asking a question about a health and displaying the answer.
    
    Args:
        model_name (str): The name of the model to use for generation
        health_topic (str): The name of the medical topic being asked about
    """
    # Check if we should disable the input
    button_disabled = model_name is None or health_topic is None
    
    # Get user question
    user_question = get_user_question(button_disabled, health_topic)
    
    if st.button("Ask Question", 
                 disabled=button_disabled or not (user_question and user_question.strip()), 
                 help="Submit your question about the selected health topic."):
        if not user_question.strip():
            st.warning("Please enter a question.")
            return
            
        if not model_name or not health_topic:
            st.error("Please select both a medical topic and a model first.")
            return
            
        get_user_answer(model_name, health_topic, user_question)

# --- UI Display Functions ---
def display_generation_result(generated_text):
    """Display the generated text using markdown."""
    if generated_text:
        st.markdown(generated_text)
    else:
        st.warning("Generation returned empty or no content.")


def get_user_question(button_disabled, selected_health_topic):
    """Display the question input UI and return the user's question.
    
    Args:
        button_disabled (bool): Whether the input should be disabled
        selected_health_topic (str): The currently selected medical topic, if any
        
    Returns:
        str: The user's question or None if not provided
    """
    st.subheader("Ask a Specific Question")

    return st.text_input(
        "Enter your question about the selected medical topic:",
        key="question_input",
        disabled=button_disabled,
        label_visibility="collapsed", 
        placeholder="e.g., What are the common treatments?" if selected_health_topic else "Select a medical topic first"
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

def select_health_topic():

    all_health_topic_names = load_health_topics() 

    selected_topic = ""

    if all_health_topic_names:
        letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

        selected_letter = st.sidebar.selectbox("Topics starting letter:", ["All"] + letters)

        # Filter topics based on selected letter
        if selected_letter == "All":
            filtered_health_topics = all_health_topic_names
        else:
            filtered_health_topics = [
                d for d in all_health_topic_names 
                if d["topic"].lower().startswith(selected_letter.lower())
            ]
        
        if filtered_health_topics:
            # Create a list of just topic names for the selectbox
            topic_names = ["Select a health topic..."] + [d["topic"] for d in filtered_health_topics]
            selected_topic_name = st.sidebar.selectbox("Select a medical topic:", topic_names)
            
            if selected_topic_name != "Select a health topic...":
                # Find the selected topic dictionary
                selected_topic = next((t for t in filtered_health_topics if t["topic"] == selected_topic_name), None)
                
                # If the selected topic exists
                if selected_topic:
                    # Clean up the 'see' list - remove any empty strings
                    if selected_topic.get("see"):
                        selected_topic["see"] = [item for item in selected_topic["see"] if item.strip()]
                        
                    # If the selected topic has valid related topics (see also), handle them
                    if selected_topic.get("see") and selected_topic["see"]:  # Check if see exists and is not empty
                        see_list = selected_topic["see"]
                        if len(see_list) == 1:
                            # If there's only one related topic, use it automatically
                            selected_topic = see_list[0]
                        else:
                            # If multiple related topics, show selection
                            see_options = ["Select a related topic..."] + see_list
                            selected_related = st.sidebar.selectbox("See also:", see_options)
                            if selected_related != "Select a related topic...":
                                selected_topic = selected_related
                            else:
                                # If user didn't select a related topic, use the main topic
                                selected_topic = selected_topic_name
                    else:
                        # If no related topics or empty see list, use the main topic
                        selected_topic  = selected_topic_name
                else:
                    # Fallback in case selected_topic is None for some reason
                    selected_topic  = selected_topic_name
        else:
            st.sidebar.info(f"No health topics found starting with '{selected_letter}'.")
    else:
        st.sidebar.info("No health topic loaded to select from.")

    return selected_topic

def configure_sidebar():
    """Configure and display the sidebar elements, returning user selections."""
    st.sidebar.header("Select Options")
    selections = {
        "model": None,
        "health_topic": None,
        "show_history": False
    }

    selections["model"] = select_llm_model()
    selections["health_topic"] = select_health_topic()
    selections["show_history"] = st.sidebar.checkbox("Show History", value=False)

    return selections

def get_health_topic_info(model_name, health_topic):
    """Display a button to get and show information about a specific health topic.
    
    Args:
        model_name (str): The name of the model to use for generation
        health_topic (str): The name of the health topic to get information about
        
    Returns:
        bool: True if information was successfully retrieved and displayed, False otherwise
    """
    if st.button("Get General Information", 
                 help="Get a general overview of the selected health topic."):
        friendly_model_name = model_name.split('models/')[-1]
        with st.spinner(f"Getting information for '{health_topic}' using {friendly_model_name}..."):
            generation_result = generate_health_topic_info(model_name, health_topic)
            if generation_result:
                display_generation_result(generation_result['text'])
                save_qa_to_queue(
                    f"General information about {health_topic}", 
                    generation_result['text'], 
                    friendly_model_name
                )
                return True
    return False

def display_main_content_area(selected_model_name, selected_health_topic, show_history):
    """Handles the display and interactions in the main application area."""
    if selected_health_topic:
        st.write(f"# Health Topic: {selected_health_topic}")
    else:
        st.info("Please select a health topic from the sidebar to get information or ask questions.")
        return

    get_health_topic_info(selected_model_name, selected_health_topic)

    health_question_answer(selected_model_name, selected_health_topic)

    # Display History (if toggled)
    if show_history:
        st.write("---")
        st.subheader("Interaction History")
        display_qa_history()

def app_health_topic():
    st.set_page_config(page_title="Health Topic Explorer", layout="wide")
    st.title("⚕️ Health Topic Explorer")

    # Initial configurations
    configure_api()
    initialize_qa_queue()

    sidebar_selections = configure_sidebar()
    
    selected_model_name = sidebar_selections["model"]
    selected_topic = sidebar_selections["health_topic"]
    show_history = sidebar_selections["show_history"]

    display_main_content_area(selected_model_name, selected_topic, show_history)

if __name__ == "__main__":
     app_health_topic()
