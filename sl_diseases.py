import streamlit as st
import google.generativeai as genai
import os
import time
import json

# --- Disease Data Loading ---
@st.cache_data # Cache the loaded data
def load_diseases(filepath="diseases.json"):
    """Load disease names from a JSON file with object structure."""
    if not os.path.exists(filepath):
        st.error(f"Disease data file not found at {filepath}")
        return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            st.error(f"Invalid format in {filepath}. Expected a list at the root.")
            return []

        diseases = []
        for item in data:
            if isinstance(item, dict) and "Name" in item and isinstance(item["Name"], str):
                diseases.append(item["Name"])
            else:
                st.warning(f"Skipping invalid entry in JSON: {item}")

        diseases.sort()
        return diseases
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from {filepath}. Please check the file format.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading diseases: {e}")
        return []

# --- API Configuration and Model Handling ---
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
            "gemini-2.0-flash-lite",    # Latest stable Flash Lite (similarly, an alias)
            "gemini-1.5-pro",           # Primary alias for the 1.5 Pro series (may point to legacy, but often listed)
            "gemini-1.5-flash",         # Primary alias for the 1.5 Flash series (similarly)
            "gemini-1.5-pro-latest",    # Explicit "latest" pointer for 1.5 Pro
            "gemini-1.5-flash-latest",  # Explicit "latest" pointer for 1.5 Flash
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


# --- QA History Management ---
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
        st.info("No history yet. Select a disease and get an answer or ask a question!")
        return

    for idx, qa in enumerate(st.session_state.qa_queue):
        st.write(f"**Entry {len(st.session_state.qa_queue) - idx}:**")
        st.write(f"   **Input/Question:** {qa['question']}")
        st.write(f"   **Model Used:** `{qa['model_used']}`")
        st.write(f"   **Answer:**")
        st.markdown(f"   {qa['answer']}") # Use markdown for better rendering of model's output
        st.divider()

# --- Generative AI Interaction ---
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

def generate_disease_info(model_name, disease_name):
    """Generate detailed information about a specific disease."""
    prompt = f"""You are a helpful and informative AI assistant specializing in providing clear and concise medical information.

Please provide detailed information about the following medical condition based on your training data and general medical knowledge. Focus on describing:
1.  Symptoms
2.  Causes and Risk Factors
3.  Diagnosis
4.  Treatment Options
5.  Prevention (if applicable)

Present the information in a clear, well-structured format using headings or bullet points.

IMPORTANT INSTRUCTION: If the provided term "{disease_name}" is NOT a known medical condition or disease according to your knowledge base, please respond simply by saying: "I can only provide information about known medical conditions. '{disease_name}' does not appear to be a medical condition I can provide information on. Please select a different medical condition from the list." Do not attempt to answer or provide any other information if it's not a recognized medical condition.

Medical Condition: {disease_name}

Information:"""
    return generate_text_from_model(model_name, prompt)

def generate_answer_for_question(model_name, disease_name, question):
    """Generate an answer to a specific question about a disease."""
    prompt = f"""You are a helpful and informative AI assistant specializing in providing clear and concise medical information.
You are currently focused on the medical condition: {disease_name}.

Please provide a detailed and accurate answer to the following question based on your knowledge of {disease_name}:

Question: {question}

Answer:"""
    return generate_text_from_model(model_name, prompt)


def get_user_answer(model_name, disease_name, question):
    """Generate and display an answer to a specific question about a disease.
    
    Args:
        model_name (str): The name of the model to use for generation
        disease_name (str): The name of the disease being asked about
        question (str): The user's question
        
    Returns:
        bool: True if answer was successfully generated and displayed, False otherwise
    """
    friendly_model_name = model_name.split('models/')[-1]
    with st.spinner(f'Finding answer about "{disease_name}" using {friendly_model_name}...'):
        generation_result = generate_answer_for_question(model_name, disease_name, question)
        if generation_result:
            st.subheader(f"Answer to: \"{question}\"")
            display_generation_result(generation_result['text'])
            save_qa_to_queue(
                f"Question about {disease_name}: {question}",
                generation_result['text'],
                friendly_model_name
            )
            return True
    return False

def disease_question_answer(model_name, disease_name):
    """Handle the process of asking a question about a disease and displaying the answer.
    
    Args:
        model_name (str): The name of the model to use for generation
        disease_name (str): The name of the disease being asked about
    """
    # Check if we should disable the input
    button_disabled = model_name is None or disease_name is None
    
    # Get user question
    user_question = get_user_question(button_disabled, disease_name)
    
    if st.button("Ask Question", 
                 disabled=button_disabled or not (user_question and user_question.strip()), 
                 help="Submit your question about the selected disease."):
        if not user_question.strip():
            st.warning("Please enter a question.")
            return
            
        if not model_name or not disease_name:
            st.error("Please select both a disease and a model first.")
            return
            
        get_user_answer(model_name, disease_name, user_question)

# --- UI Display Functions ---
def display_generation_result(generated_text):
    """Display the generated text using markdown."""
    if generated_text:
        st.markdown(generated_text)
    else:
        st.warning("Generation returned empty or no content.")


def get_user_question(button_disabled, selected_disease):
    """Display the question input UI and return the user's question.
    
    Args:
        button_disabled (bool): Whether the input should be disabled
        selected_disease (str): The currently selected disease, if any
        
    Returns:
        str: The user's question or None if not provided
    """
    st.subheader("Ask a Specific Question")
    return st.text_input(
        "Enter your question about the selected disease:",
        key="question_input",
        disabled=button_disabled,
        label_visibility="collapsed", 
        placeholder="e.g., What are the common treatments?" if selected_disease else "Select a disease first"
    )

# --- Modular UI Sections ---
def configure_sidebar(available_models, all_diseases_names):
    """Configure and display the sidebar elements, returning user selections."""
    st.sidebar.header("Select Options")
    selections = {
        "model": None,
        "disease": None,
        "show_history": False
    }

    # Model selection
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
            selections["model"] = model_name_map[selected_display_name] # Get full model name for internal use
    else:
        st.sidebar.warning("No text generation models available. Check API key and configuration, or model filtering.")


    # Disease selection based on letter
    if all_diseases_names:
        letters = sorted(list(set(d[0].upper() for d in all_diseases_names if d))) 
        if not letters: # Fallback if diseases have no typical starting letters
             letters = [chr(i) for i in range(65, 91)] 

        selected_letter = st.sidebar.selectbox("Filter diseases by starting letter:", ["All"] + letters)

        if selected_letter == "All":
            filtered_diseases = all_diseases_names
        else:
            filtered_diseases = [d for d in all_diseases_names if d.lower().startswith(selected_letter.lower())]
        
        filtered_diseases.sort()

        if filtered_diseases:
            display_list = ["Select a disease..."] + filtered_diseases
            selected_disease_with_placeholder = st.sidebar.selectbox("Select a disease:", display_list)
            if selected_disease_with_placeholder != "Select a disease...":
                selections["disease"] = selected_disease_with_placeholder
        else:
            st.sidebar.info(f"No diseases found starting with '{selected_letter}'.")
    else:
        st.sidebar.info("No diseases loaded to select from.")

    # History Toggle
    selections["show_history"] = st.sidebar.checkbox("Show History", value=False)

    return selections

def get_disease_info(model_name, disease_name):
    """Display a button to get and show information about a specific disease.
    
    Args:
        model_name (str): The name of the model to use for generation
        disease_name (str): The name of the disease to get information about
        
    Returns:
        bool: True if information was successfully retrieved and displayed, False otherwise
    """
    if st.button("Get General Information", 
                 help="Get a general overview of the selected disease."):
        friendly_model_name = model_name.split('models/')[-1]
        with st.spinner(f"Getting information for '{disease_name}' using {friendly_model_name}..."):
            generation_result = generate_disease_info(model_name, disease_name)
            if generation_result:
                display_generation_result(generation_result['text'])
                save_qa_to_queue(
                    f"General information about {disease_name}", 
                    generation_result['text'], 
                    friendly_model_name
                )
                return True
    return False


def display_main_content_area(selected_model_name, selected_disease, show_history, all_diseases_names_present):
    """Handles the display and interactions in the main application area."""
    if selected_disease:
        st.write(f"# Disease: {selected_disease}")
    else:
        st.info("Please select a disease from the sidebar to get information or ask questions.")

    # Display disease information section
    get_disease_info(selected_model_name, selected_disease)

    st.write("---") 

    disease_question_answer(selected_model_name, selected_disease)

    # Display History (if toggled)
    if show_history:
        st.write("---")
        st.subheader("Interaction History")
        display_qa_history()

# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Disease Explorer", layout="wide")
    st.title("⚕️ Disease Explorer")

    # Initial configurations
    configure_api()
    initialize_qa_queue()

    # Load data and models
    available_models = get_available_generation_models()
    all_diseases_names = load_diseases() 

    # Configure sidebar and get selections
    sidebar_selections = configure_sidebar(available_models, all_diseases_names)
    selected_model_name = sidebar_selections["model"]
    selected_disease = sidebar_selections["disease"]
    show_history = sidebar_selections["show_history"]

    # Display main content area based on selections
    display_main_content_area(selected_model_name, selected_disease, show_history, bool(all_diseases_names))


if __name__ == "__main__":
    main()
