import streamlit as st
import google.generativeai as genai
import os
import time
import json

@st.cache_data # Cache the loaded data
def load_medicines(filepath="drugscom_drugs.json"):
    """
    Load medicines from a JSON file with the structure:
    ["medicine_1", "medicine_2", ...]
    """
    if not os.path.exists(filepath):
        st.error(f"Medicines file not found at {filepath}")
        return []
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            st.error(f"Invalid format in {filepath}. Expected a list at the root.")
            return []

        # Return the list of medicines
        return data
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from {filepath}. Please check the file format.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading medicines: {e}")
        return []

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
            "gemini-2.0-flash",           # Latest stable Flash (likely an alias to a specific version like -001)
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
        st.info("No history yet. Select a medicine and get an answer or ask a question!")
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

def generate_medicine_info(model_name, medicine):
    """Generate detailed and user-friendly information about a specific medicine."""
    prompt = f"""You are a helpful, accurate, and responsible AI assistant specializing in providing clear, concise, and easy-to-understand medical information for the general public.

**IMPORTANT MEDICAL DISCLAIMER:**
The information provided is for general knowledge and informational purposes only, and does not constitute professional medical advice. It is not a substitute for a consultation with a qualified healthcare professional, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Do not disregard professional medical advice or delay in seeking it because of something you have read here.

Please provide comprehensive and structured information about the following medicine, drawing from your training data and general health knowledge. Focus on delivering information that is most relevant and crucial for safe and effective use by a layperson.

**Medicine Name:** {medicine}

---

Present the information under the following main headings. If specific information is not available or applicable, state "Information not available" or "Not applicable" under that subheading rather than omitting the heading entirely.

## Overview
Give detail information about the medicine.

## Brief History 
Provide a concise historical context, including its discovery, company name, development, and first approval 

## Is Rx required?
Is the prescription only medicine.

## How does it works?
Give a concise description on how does the medicine works?

## Classification
Give the classification of the medicine.

## Generic Names
Give the list of generic names of the medicine in India

## Brand Names
Give the list of brand names of the medicine in India

## Physical Form and strenghts
In which physical forms(s) and strengths the medicine is available in India

## Life span
What are the life of the unopened and opened medicine. 

## Active Ingredient(s)
Give the list of active ingradients along with percentage composition

## Indications
Explain why this medicine is prescribed and its main therapeutic uses, including any other significant uses.

## Contradictions
Who should not be given this medicine?

## How and When to Take/Use
Detailed instructions on administration (e.g., with/without food, time of day, route of administration).

## Recommended Dosage
Typical dosage for common indications. Specify if dosage varies by age, weight, or condition.

## Duration of Use
For how long is this medicine typically required or recommended?

## Storage at Home
Instructions on how to properly store the medicine (temperature, light, moisture, etc.).

## Missed Dose 
Clear instructions for handling a forgotten dose.

## Overdose
Guidance on immediate actions for an overdose situation.

## Special Precautions to Follow
Any particular warnings or considerations (e.g., for pregnant/breastfeeding individuals, elderly, individuals with specific health conditions, driving, etc.).

## How to Know if it's Working for me?
Signs or indicators that the medicine is having its intended effect.

## Implications of Skipping Doses/Not Completing Course
What might happen if the medicine isn't taken as prescribed.

## Common Side Effects
List frequently occurring, generally mild side effects.

## Serious Side Effects 
List severe or urgent side effects that require immediate medical consultation.

## Drug Interactions
How this medicine interacts with other prescription drugs and over-the-counter (OTC) medicines. List specific OTC medicines to avoid if applicable.

## Food and Drink Interactions
Any specific foods or beverages to avoid while taking this medicine.

## Other Interactions
 How does this drug interacts with with alcohol, certain medical conditions, lab tests.

## Similar Drugs
List chemically similar drugs or common therapeutic alternatives available in India.

## Alternatives
List all the alternatives to this medicine.

## Major Manufacturers
List key pharmaceutical companies that manufacture this drug in India.

## Current Status
Check if the medicine or its ingradients have been banned anywhere in the world.

## Evidence of Effectiveness
Present evidence of the effectiveness of this medicine in the treatment. It may include
research papers from medical literature or articles from medical hospitals/companies.

---

**Output Format:** Please present the information clearly and concisely using Markdown headings (##) for main sections and bullet points (*) or numbered lists for sub-points as structured above. Maintain a neutral, informative, and easy-to-understand tone suitable for the general public.
"""
    return generate_text_from_model(model_name, prompt)

def generate_answer_for_question(model_name, medicine, question):
    """Generate an answer to a specific question about a medicine """
    prompt = f"""You are a helpful and informative AI assistant specializing in providing clear and concise information on the medical drug.
You are currently focused on the medicine : {medicine}.

Please provide a detailed and accurate answer to the following question based on your knowledge of {medicine}:

Question: {question}

Answer:"""
    return generate_text_from_model(model_name, prompt)


def get_user_answer(model_name, medicine, question):
    """Generate and display an answer to a specific question about a medicine.
    
    Args:
        model_name (str): The name of the model to use for generation
        medicine (str): The name of the medicine being asked about
        question (str): The user's question
        
    Returns:
        bool: True if answer was successfully generated and displayed, False otherwise
    """
    friendly_model_name = model_name.split('models/')[-1]
    with st.spinner(f'Finding answer about "{medicine}" using {friendly_model_name}...'):
        generation_result = generate_answer_for_question(model_name, medicine, question)
        if generation_result:
            st.subheader(f"Answer to: \"{question}\"")
            display_generation_result(generation_result['text'])
            save_qa_to_queue(
                f"Question about {medicine}: {question}",
                generation_result['text'],
                friendly_model_name
            )
            return True
    return False

def medicine_user_question(model_name, medicine): # Renamed function
    """Handle the process of asking a question about a medicine and displaying the answer.
    
    Args:
        model_name (str): The name of the model to use for generation
        medicine (str): The name of the medicine being asked about
    """
    # Check if we should disable the input
    button_disabled = model_name is None or medicine is None
    
    # Get user question
    user_question = get_user_question(button_disabled, medicine)
    
    if st.button("Ask Question", 
                  disabled=button_disabled or not (user_question and user_question.strip()), 
                  help="Submit your question about the selected medicine."):
        if not user_question.strip():
            st.warning("Please enter a question.")
            return
            
        if not model_name or not medicine:
            st.error("Please select both a medicine and a model first.")
            return
            
        get_user_answer(model_name, medicine, user_question)

# --- UI Display Functions ---
def display_generation_result(generated_text):
    """Display the generated text using markdown."""
    if generated_text:
        st.markdown(generated_text)
    else:
        st.warning("Generation returned empty or no content.")


def get_user_question(button_disabled, selected_medicine):
    """Display the question input UI and return the user's question.
    
    Args:
        button_disabled (bool): Whether the input should be disabled
        selected_medicine (str): The currently selected medicine, if any
        
    Returns:
        str: The user's question or None if not provided
    """
    st.subheader("Ask a Specific Question")
    return st.text_input(
        "Enter your question about the selected medicine:",
        key="question_input",
        disabled=button_disabled,
        label_visibility="collapsed", 
        placeholder="e.g., What are the common treatments?" if selected_medicine else "Select a medicine first"
    )

# --- Modular UI Sections ---
def configure_sidebar(available_models, all_medicines_names):
    """Configure and display the sidebar elements, returning user selections."""
    st.sidebar.header("Select Options")
    selections = {
        "model": None,
        "medicine": None,
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


    if all_medicines_names:
        letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

        selected_letter = st.sidebar.selectbox("Filter medicine by starting letter:", ["All"] + letters)

        # Filter topics based on selected letter
        if selected_letter == "All":
            filtered_medicines = all_medicines_names
        else:
            filtered_medicines = [d for d in all_medicines_names if d.lower().startswith(selected_letter.lower())]
        
        if filtered_medicines:
            # Create a list of just topic names for the selectbox
            topic_names = ["Select a medicine..."] + filtered_medicines
            selected_medicine = st.sidebar.selectbox("Select a medicine:", topic_names)
            
            if selected_medicine != "Select a medicine...":
                selections["medicine"] = selected_medicine
        else:
            st.sidebar.info(f"No medicine found starting with '{selected_letter}'.")
    else:
        st.sidebar.info("No medicine loaded to select from.")

    # History Toggle
    selections["show_history"] = st.sidebar.checkbox("Show History", value=False)

    return selections

def get_medicine_info(model_name, medicine):
    """Display a button to get and show information about a specific medicine.
    
    Args:
        model_name (str): The name of the model to use for generation
        medicine (str): The name of the medicine to get information about
        
    Returns:
        bool: True if information was successfully retrieved and displayed, False otherwise
    """
    if st.button("Get General Information", 
                  help="Get a general overview of the selected medicine."):
        friendly_model_name = model_name.split('models/')[-1]
        with st.spinner(f"Getting information for '{medicine}' using {friendly_model_name}..."):
            generation_result = generate_medicine_info(model_name, medicine)
            if generation_result:
                display_generation_result(generation_result['text'])
                save_qa_to_queue(
                    f"General information about {medicine}", 
                    generation_result['text'], 
                    friendly_model_name
                )
                return True
    return False


def display_main_content_area(selected_model_name, selected_medicine, show_history, all_medicines_names_present):
    """Handles the display and interactions in the main application area."""
    if selected_medicine:
        st.write(f"# {selected_medicine}")
    else:
        st.info("Please select a medicine from the sidebar to get information or ask questions.")
        return

    get_medicine_info(selected_model_name, selected_medicine)

    st.write("---") 

    medicine_user_question(selected_model_name, selected_medicine)

    # Display History (if toggled)
    if show_history:
        st.write("---")
        st.subheader("Interaction History")
        display_qa_history()

# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Medicine Explorer", layout="wide")
    st.title("⚕️ Medicine Explorer")

    # Initial configurations
    configure_api()
    initialize_qa_queue()

    # Load data and models
    available_models = get_available_generation_models()
    all_medicines_names = load_medicines() # Renamed function call and variable

    # Configure sidebar and get selections
    
    sidebar_selections = configure_sidebar(available_models, all_medicines_names) # Renamed variable
    
    selected_model_name = sidebar_selections["model"]
    selected_medicine = sidebar_selections["medicine"]
    show_history = sidebar_selections["show_history"]

    display_main_content_area(selected_model_name, selected_medicine, show_history, bool(all_medicines_names)) # Renamed variable


if __name__ == "__main__":
    main()
