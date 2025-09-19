import streamlit as st
import pandas as pd
import openai
import json
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


# --- 1. SETUP AND DATA PREPARATION ---

# Set your OpenAI API key from Streamlit's secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_data(show_spinner="Loading and preparing data...")
def load_and_prepare_data(filepath):
    """
    Loads data from the Excel file, merges the sheets into a single DataFrame,
    and performs feature engineering. This is the foundation of our app.
    """
    trips_df = pd.read_excel(filepath, sheet_name='Trip Data')
    checkins_df = pd.read_excel(filepath, sheet_name='Checked in User ID\'s')
    demographics_df = pd.read_excel(filepath, sheet_name='Customer Demographics')

    user_ages_df = pd.merge(checkins_df, demographics_df, on='User ID', how='left')
    trip_age_stats_df = user_ages_df.groupby('Trip ID')['Age'].agg(['mean', 'min', 'max']).reset_index()
    trip_age_stats_df.rename(columns={'mean': 'avg_age', 'min': 'min_age', 'max': 'max_age'}, inplace=True)
    
    master_df = pd.merge(trips_df, trip_age_stats_df, on='Trip ID', how='left')
    
    master_df['Trip Date and Time'] = pd.to_datetime(master_df['Trip Date and Time'])
    master_df['day_of_week'] = master_df['Trip Date and Time'].dt.day_name()
    master_df['hour_of_day'] = master_df['Trip Date and Time'].dt.hour
    
    # Clean up address for better matching
    master_df['Drop Off Address'] = master_df['Drop Off Address'].str.lower()
    
    return master_df

# --- 2. RELIABLE "INTENT" FUNCTIONS (PATH A) ---

def handle_count_trips(df, filters):
    """
    Handles the 'count_trips' intent. This is a reliable, hard-coded function.
    Filters the DataFrame based on a destination keyword.
    """
    destination_keyword = filters.get("destination", "").lower()
    if not destination_keyword:
        return "You need to specify a destination to count trips."
    
    filtered_df = df[df['Drop Off Address'].str.contains(destination_keyword, na=False)]
    count = len(filtered_df)
    
    return f"Based on the data, there were {count} trips to locations containing '{filters.get('destination')}'. 🚐"

def handle_top_destinations(df, filters):
    """
    Handles the 'top_destinations' intent.
    Finds the most frequent drop-off locations.
    """
    top_n = filters.get("limit", 5)
    top_destinations = df['Drop Off Address'].value_counts().nlargest(top_n)
    
    response = f"Here are the top {top_n} destinations:\n"
    for dest, count in top_destinations.items():
        # Capitalize for display
        response += f"- {dest.title()}: {count} trips\n"
    return response

# --- 3. THE AI ROUTER ---

def route_question(question):
    """
    Uses GPT-4o to classify the user's question into a known intent or
    routes it to the agent as a fallback.
    """
    prompt = f"""
    You are an AI router. Your job is to classify the user's question and extract any necessary parameters.
    The available intents are:
    - "count_trips": For questions about counting trips to a specific place. Requires a "destination" parameter.
    - "top_destinations": For questions asking for the most popular destinations. Can take an optional "limit" parameter.
    - "agent_fallback": For any other, more complex or general question that the other intents can't handle.

    Analyze the user's question: "{question}"

    Return your response ONLY as a JSON object with two keys: "intent" and "filters".
    "filters" should be an object containing any extracted parameters.
    If no parameters are found, "filters" should be an empty object.
    
    Example for "How many groups went to the Moody Center?":
    {{"intent": "count_trips", "filters": {{"destination": "Moody Center"}}}}
    
    Example for "What are the top 5 destinations?":
    {{"intent": "top_destinations", "filters": {{"limit": 5}}}}

    Example for "What's the average age of riders on Saturday nights?":
    {{"intent": "agent_fallback", "filters": {{}}}}
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, IndexError) as e:
        # If the LLM fails to return valid JSON, fall back to the agent
        st.warning("Router failed to produce valid JSON, falling back to agent.")
        return {"intent": "agent_fallback", "filters": {}}


# --- 4. AI AGENT "SAFETY NET" (PATH B) ---

def run_agent_query(df, question):
    """
    Initializes and runs the LangChain Pandas Agent for complex queries.
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=st.secrets["OPENAI_API_KEY"])
    # ✅ Correct (new code)
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
        allow_dangerous_code=True  # <--- Add this line
    )
    
    # Add a prefix to the prompt to guide the agent
    prompt_prefix = """
    You are a data analyst for Fetii, a group rideshare company.
    Please answer the user's question based on the provided DataFrame.
    When calculating statistics like averages, if data is missing, state that the calculation
    is based only on the trips with available data.
    """
    full_question = prompt_prefix + "\nUser question: " + question

    try:
        response = agent.invoke(full_question)
        return response.get('output', "I couldn't process that question with the agent.")
    except Exception as e:
        return f"An error occurred with the agent: {e}"

# --- 5. STREAMLIT UI AND MAIN APP LOGIC ---

st.set_page_config(layout="wide")
st.title("🚀 FetiiAI Hybrid Analytics")
st.markdown("Ask a question about Fetii's Austin rideshare data. Try one of the examples or ask your own!")

# Load the data once
master_df = load_and_prepare_data('fetii_data.xlsx')

# Example questions for users to click
example_questions = [
    "How many trips went to the Moody Center?",
    "What are the top 3 most popular destinations?",
    "What time do large groups (10+) usually ride in Austin on Fridays?",
    "What is the average age of passengers going to The Domain?"
]

# Create a row of buttons for example questions
cols = st.columns(len(example_questions))
for i, question in enumerate(example_questions):
    if cols[i].button(question):
        st.session_state.user_question = question
    
# Use session state to hold the input value
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""

user_question = st.text_input("Your Question:", value=st.session_state.user_question, key="question_input")

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Analyzing your question..."):
            # 1. Route the question
            routed_intent = route_question(user_question)
            intent = routed_intent.get("intent")
            filters = routed_intent.get("filters", {})

            st.info(f"**Selected Path:** `{intent}`") # Show the user which path was taken

            # 2. Execute the appropriate path
            if intent == "count_trips":
                response = handle_count_trips(master_df, filters)
            elif intent == "top_destinations":
                response = handle_top_destinations(master_df, filters)
            else: # Fallback to the agent
                with st.spinner("The question is complex. Engaging the AI Data Analyst Agent... Please wait."):
                    response = run_agent_query(master_df, user_question)
            
            # 3. Display the response
            st.success(response)
    else:
        st.warning("Please ask a question.")