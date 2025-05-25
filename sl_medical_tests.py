import streamlit as st
import google.generativeai as genai
import os
import time
import json


@st.cache_data # Cache the loaded data
def load_medical_tests(filepath="redcliffe_labtests.json"):
    """
    Load medical tests from a JSON file with the structure:
    ["test_1", "test_2", ...]
    """
    if not os.path.exists(filepath):
        st.error(f"Medical tests file not found at {filepath}")
        return []
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            st.error(f"Invalid format in {filepath}. Expected a list at the root.")
            return []

        # Return the list of medical tests
        return data
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from {filepath}. Please check the file format.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading medical_tests: {e}")
        return []
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
        st.error(f"An unexpected error occurred while loading medical_tests: {e}")
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
        st.info("No history yet. Select a medical_test and get an answer or ask a question!")
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

def generate_medical_test_info(model_name, medical_test):
    """Generate detailed information about a specific medical_test."""
    prompt = f"""You are a helpful and informative AI assistant specializing in providing clear and concise medical information.

Please provide detailed information about the following medical_test based on your training data and general health knowledge. Focus on describing:
1.  Definition
2.  Alternative Names
3.  Brief history of this test.
4.  Why do I need this test?
5.  Is this test critical for life-saving? 
6.  Is this test prerequiste for other tests?
7.  What this test is trying to diagnose?
8.  What symptoms or concerns require this test?
9.  What types of tests are available?
10.  How the test is performed?
11.  How to prepare for the test?
12. How the test will feel?
13. What are normal results?
14. What abnormal results mean?
15. What risks or side effects are there from the test?
16. Must I do anything special after the test is over?
17. How long is it before the result of the test is known?
18. Considerations
19. Cost Range in India (In Indian Currency)
20. Do I need some assistant after the test? 
21. How accurate and reliable is this test?
22. Is this a one-time test of will it need to be repeated?
23. Will I need more tests or treatment based on the results?
24. Latest Developments about the test.
25. References

Present the information in a clear, well-structured format using headings or bullet points.

Medical Test: {medical_test}

Information:"""
    return generate_text_from_model(model_name, prompt)

def generate_answer_for_question(model_name, medical_test, question):
    """Generate an answer to a specific question about a medical_test."""
    prompt = f"""You are a helpful and informative AI assistant specializing in providing clear and concise medical information.
You are currently focused on the medical topic: {medical_test}.

Please provide a detailed and accurate answer to the following question based on your knowledge of {medical_test}:

Question: {question}

Answer:"""
    return generate_text_from_model(model_name, prompt)


def get_user_answer(model_name, medical_test, question):
    """Generate and display an answer to a specific question about a medical_test.
    
    Args:
        model_name (str): The name of the model to use for generation
        medical_test (str): The name of the medical_test being asked about
        question (str): The user's question
        
    Returns:
        bool: True if answer was successfully generated and displayed, False otherwise
    """
    friendly_model_name = model_name.split('models/')[-1]
    with st.spinner(f'Finding answer about "{medical_test}" using {friendly_model_name}...'):
        generation_result = generate_answer_for_question(model_name, medical_test, question)
        if generation_result:
            st.subheader(f"Answer to: \"{question}\"")
            display_generation_result(generation_result['text'])
            save_qa_to_queue(
                f"Question about {medical_test}: {question}",
                generation_result['text'],
                friendly_model_name
            )
            return True
    return False

def health_question_answer(model_name, medical_test):
    """Handle the process of asking a question about a medical_test and displaying the answer.
    
    Args:
        model_name (str): The name of the model to use for generation
        medical_test (str): The name of the medical_test being asked about
    """
    # Check if we should disable the input
    button_disabled = model_name is None or medical_test is None
    
    # Get user question
    user_question = get_user_question(button_disabled, medical_test)
    
    if st.button("Ask Question", 
                 disabled=button_disabled or not (user_question and user_question.strip()), 
                 help="Submit your question about the selected medical_test."):
        if not user_question.strip():
            st.warning("Please enter a question.")
            return
            
        if not model_name or not medical_test:
            st.error("Please select both a medical_test and a model first.")
            return
            
        get_user_answer(model_name, medical_test, user_question)

# --- UI Display Functions ---
def display_generation_result(generated_text):
    """Display the generated text using markdown."""
    if generated_text:
        st.markdown(generated_text)
    else:
        st.warning("Generation returned empty or no content.")


def get_user_question(button_disabled, selected_medical_test):
    """Display the question input UI and return the user's question.
    
    Args:
        button_disabled (bool): Whether the input should be disabled
        selected_medical_test (str): The currently selected medical_test, if any
        
    Returns:
        str: The user's question or None if not provided
    """
    st.subheader("Ask a Specific Question")
    return st.text_input(
        "Enter your question about the selected medical_test:",
        key="question_input",
        disabled=button_disabled,
        label_visibility="collapsed", 
        placeholder="e.g., What are the common treatments?" if selected_medical_test else "Select a medical_test first"
    )

# --- Modular UI Sections ---
def configure_sidebar(available_models, all_medical_test_names):
    """Configure and display the sidebar elements, returning user selections."""
    st.sidebar.header("Select Options")
    selections = {
        "model": None,
        "medical_test": None,
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


    if all_medical_test_names:
        letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

        selected_letter = st.sidebar.selectbox("Filter medical_tests by starting letter:", ["All"] + letters)

        # Filter topics based on selected letter
        if selected_letter == "All":
            filtered_medical_tests = all_medical_test_names
        else:
            filtered_medical_tests = [d for d in all_medical_test_names if d.lower().startswith(selected_letter.lower())]
        
        if filtered_medical_tests:
            # Create a list of just topic names for the selectbox
            topic_names = ["Select a medical_test..."] + filtered_medical_tests
            selected_topic_name = st.sidebar.selectbox("Select a medical_test:", topic_names)
            
            if selected_topic_name != "Select a medical_test...":
                selections["medical_test"] = selected_topic_name
        else:
            st.sidebar.info(f"No medical_tests found starting with '{selected_letter}'.")
    else:
        st.sidebar.info("No medical_test loaded to select from.")

    # History Toggle
    selections["show_history"] = st.sidebar.checkbox("Show History", value=False)

    return selections

def get_medical_test_info(model_name, medical_test):
    """Display a button to get and show information about a specific medical_test.
    
    Args:
        model_name (str): The name of the model to use for generation
        medical_test (str): The name of the medical_test to get information about
        
    Returns:
        bool: True if information was successfully retrieved and displayed, False otherwise
    """
    if st.button("Get General Information", 
                 help="Get a general overview of the selected medical_test."):
        friendly_model_name = model_name.split('models/')[-1]
        with st.spinner(f"Getting information for '{medical_test}' using {friendly_model_name}..."):
            generation_result = generate_medical_test_info(model_name, medical_test)
            if generation_result:
                display_generation_result(generation_result['text'])
                save_qa_to_queue(
                    f"General information about {medical_test}", 
                    generation_result['text'], 
                    friendly_model_name
                )
                return True
    return False


def display_main_content_area(selected_model_name, selected_medical_test, show_history, all_medical_test_names_present):
    """Handles the display and interactions in the main application area."""
    if selected_medical_test:
        st.write(f"# Medical Test: {selected_medical_test}")
    else:
        st.info("Please select a medical_test from the sidebar to get information or ask questions.")
        return

    get_medical_test_info(selected_model_name, selected_medical_test)

    st.write("---") 

    health_question_answer(selected_model_name, selected_medical_test)

    # Display History (if toggled)
    if show_history:
        st.write("---")
        st.subheader("Interaction History")
        display_qa_history()

# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Medical Test Explorer", layout="wide")
    st.title("⚕️ Medical Test Explorer")

    # Initial configurations
    configure_api()
    initialize_qa_queue()

    # Load data and models
    available_models = get_available_generation_models()
    all_medical_test_names = load_medical_tests() 

    # Configure sidebar and get selections
    
    sidebar_selections = configure_sidebar(available_models, all_medical_test_names)
    
    selected_model_name = sidebar_selections["model"]
    selected_topic = sidebar_selections["medical_test"]
    show_history = sidebar_selections["show_history"]

    display_main_content_area(selected_model_name, selected_topic, show_history, bool(all_medical_test_names))


if __name__ == "__main__":
    main()
