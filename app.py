import streamlit as st
import pandas as pd
import openai
import json
import plotly.express as px
import plotly.graph_objects as go
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import re


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

# --- PERFORMANCE OPTIMIZATIONS: PRE-COMPUTED STATS ---

@st.cache_data
def get_dashboard_metrics(df):
    """Pre-compute key metrics for dashboard display"""
    return {
        'total_trips': len(df),
        'total_destinations': df['Drop Off Address'].nunique(),
        'peak_hour': df.groupby('hour_of_day').size().idxmax(),
        'peak_day': df['day_of_week'].value_counts().index[0],
        'avg_group_size': df.groupby('Trip ID').size().mean(),
        'popular_dest': df['Drop Off Address'].value_counts().index[0]
    }

@st.cache_data
def get_destination_stats(df):
    """Cache destination statistics"""
    return df['Drop Off Address'].value_counts()

@st.cache_data
def get_time_patterns(df):
    """Cache time-based patterns"""
    hourly = df.groupby('hour_of_day').size()
    daily = df.groupby('day_of_week').size()
    return {'hourly': hourly, 'daily': daily}

@st.cache_data
def get_age_demographics(df):
    """Cache age-related statistics"""
    return {
        'avg_age': df['avg_age'].mean(),
        'age_by_dest': df.groupby('Drop Off Address')['avg_age'].mean().sort_values(ascending=False)
    }

# --- 2. RELIABLE "INTENT" FUNCTIONS (PATH A) ---

def handle_count_trips(df, filters):
    """
    Handles the 'count_trips' intent. This is a reliable, hard-coded function.
    Filters the DataFrame based on a destination keyword.
    """
    destination_keyword = filters.get("destination", "").lower()
    if not destination_keyword:
        return "You need to specify a destination to count trips."

    # Strip common articles for better matching
    destination_keyword = re.sub(r'\b(the|a|an)\s+', '', destination_keyword).strip()

    filtered_df = df[df['Drop Off Address'].str.contains(destination_keyword, na=False)]
    count = len(filtered_df)

    return f"Based on the data, there were {count} trips to locations containing '{filters.get('destination')}'. 🚐"

def handle_top_destinations(df, filters):
    """
    Handles the 'top_destinations' intent.
    Finds the most frequent drop-off locations.
    """
    top_n = filters.get("limit", 5)
    dest_stats = get_destination_stats(df)  # Use cached stats
    top_destinations = dest_stats.nlargest(top_n)

    response = f"Here are the top {top_n} destinations:\n"
    for dest, count in top_destinations.items():
        response += f"- {dest.title()}: {count} trips\n"
    return response

def handle_time_analysis(df, filters):
    """
    Handles time-based questions about ride patterns.
    """
    time_patterns = get_time_patterns(df)
    peak_hour = time_patterns['hourly'].idxmax()
    peak_count = time_patterns['hourly'].max()

    response = f"Peak riding time is {peak_hour}:00 with {peak_count} trips. "
    response += f"Most popular day is {time_patterns['daily'].idxmax()} with {time_patterns['daily'].max()} trips."
    return response

def handle_age_insights(df, filters):
    """
    Handles age-related demographic questions.
    """
    destination = filters.get("destination", "").lower()
    age_stats = get_age_demographics(df)

    if destination:
        filtered_df = df[df['Drop Off Address'].str.contains(destination, na=False)]
        if len(filtered_df) > 0:
            avg_age = filtered_df['avg_age'].mean()
            return f"Average age of riders going to locations containing '{filters.get('destination')}' is {avg_age:.1f} years."
        else:
            return f"No trips found to locations containing '{filters.get('destination')}'."
    else:
        return f"Overall average age of riders is {age_stats['avg_age']:.1f} years."

def handle_group_size(df, filters):
    """
    Handles questions about group sizes and ride capacity.
    """
    trip_sizes = df.groupby('Trip ID').size()
    large_groups = trip_sizes[trip_sizes >= 8].count()
    avg_size = trip_sizes.mean()

    return f"Average group size is {avg_size:.1f} people. {large_groups} trips had large groups (8+ people)."

# --- 3. SMART ROUTING WITH PATTERN MATCHING ---

def quick_pattern_match(question):
    """
    Check for obvious patterns before using AI router to save API calls.
    """
    question_lower = question.lower()

    # Simple pattern matching for common questions
    if re.search(r'\bhow many.*trips?.*to\b', question_lower):
        destination_match = re.search(r'to\s+([^?]+)', question_lower)
        if destination_match:
            return {"intent": "count_trips", "filters": {"destination": destination_match.group(1).strip()}}

    if re.search(r'\btop\s*\d*\s*destinations?\b', question_lower):
        limit_match = re.search(r'top\s*(\d+)', question_lower)
        limit = int(limit_match.group(1)) if limit_match else 5
        return {"intent": "top_destinations", "filters": {"limit": limit}}

    if re.search(r'\btime\b|\bhour\b|\bwhen\b|\bpeak\b', question_lower):
        return {"intent": "time_analysis", "filters": {}}

    if re.search(r'\bage\b|\bold\b|\byoung\b', question_lower):
        destination_match = re.search(r'(going to|to)\s+([^?]+)', question_lower)
        filters = {"destination": destination_match.group(2).strip()} if destination_match else {}
        return {"intent": "age_insights", "filters": filters}

    if re.search(r'\bgroup size\b|\blarge group\b|\bhow many people\b', question_lower):
        return {"intent": "group_size", "filters": {}}

    return None  # No pattern match, use AI router

@st.cache_data
def route_question_cached(question):
    """
    Cached version of AI routing to save API calls for similar questions.
    """
    return route_question(question)

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
    - "time_analysis": For questions about when people ride, peak hours, popular days.
    - "age_insights": For questions about rider demographics and age patterns. Can take optional "destination" parameter.
    - "group_size": For questions about group sizes, large groups, capacity.
    - "agent_fallback": For any other, more complex or general question that the other intents can't handle.

    Analyze the user's question: "{question}"

    Return your response ONLY as a JSON object with two keys: "intent" and "filters".
    "filters" should be an object containing any extracted parameters.
    If no parameters are found, "filters" should be an empty object.
    
    Example for "How many groups went to the Moody Center?":
    {{"intent": "count_trips", "filters": {{"destination": "Moody Center"}}}}
    
    Example for "What are the top 5 destinations?":
    {{"intent": "top_destinations", "filters": {{"limit": 5}}}}

    Example for "What time do most people ride?":
    {{"intent": "time_analysis", "filters": {{}}}}

    Example for "What's the average age going to UT?":
    {{"intent": "age_insights", "filters": {{"destination": "UT"}}}}

    Example for "How many large groups book rides?":
    {{"intent": "group_size", "filters": {{}}}}

    Example for "What's the correlation between weather and ridership?":
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

st.set_page_config(
    page_title="FetiiAI Analytics",
    page_icon="🚐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
}
.metric-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin-bottom: 1rem;
}
.stButton > button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 20px;
    border: none;
}
.example-button {
    margin: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('''
<div class="main-header">
    <h1 style="color: white; margin: 0;">🚐 FetiiAI Analytics Dashboard</h1>
    <p style="color: #f0f0f0; margin: 0.5rem 0 0 0;">Intelligent insights for Austin rideshare data</p>
</div>
''', unsafe_allow_html=True)

# Load the data and compute metrics once
master_df = load_and_prepare_data('fetii_data.xlsx')
metrics = get_dashboard_metrics(master_df)
time_patterns = get_time_patterns(master_df)

# Sidebar with key metrics
with st.sidebar:
    st.markdown("### 📊 Quick Stats")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Trips", f"{metrics['total_trips']:,}")
        st.metric("Peak Hour", f"{metrics['peak_hour']}:00")
    with col2:
        st.metric("Destinations", metrics['total_destinations'])
        st.metric("Avg Group Size", f"{metrics['avg_group_size']:.1f}")

    st.markdown(f"**Most Popular Day:** {metrics['peak_day']}")
    st.markdown(f"**Top Destination:** {metrics['popular_dest'].title()}")

    # Simple hourly chart
    st.markdown("### 📈 Trips by Hour")
    hourly_chart = px.bar(
        x=time_patterns['hourly'].index,
        y=time_patterns['hourly'].values,
        labels={'x': 'Hour of Day', 'y': 'Number of Trips'}
    )
    hourly_chart.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(hourly_chart, use_container_width=True)

# Enhanced example questions organized by category
st.markdown("### 💡 Try These Example Questions")

example_categories = {
    "📍 Destinations": [
        "How many trips went to the Moody Center?",
        "What are the top 5 most popular destinations?"
    ],
    "⏰ Time Patterns": [
        "What time do most people ride?",
        "When is peak riding time?"
    ],
    "👥 Demographics": [
        "What is the average age of passengers?",
        "How many large groups book rides?"
    ]
}

for category, questions in example_categories.items():
    st.markdown(f"**{category}**")
    cols = st.columns(len(questions))
    for i, question in enumerate(questions):
        if cols[i].button(question, key=f"{category}_{i}"):
            st.session_state.user_question = question

# This section is now handled above in the categorized examples
    
# Use session state to hold the input value
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""

user_question = st.text_input("Your Question:", value=st.session_state.user_question, key="question_input")

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Analyzing your question..."):
            # 1. Try pattern matching first, then AI routing
            routed_intent = quick_pattern_match(user_question)
            if routed_intent is None:
                # No pattern match, use cached AI router
                routed_intent = route_question_cached(user_question)

            intent = routed_intent.get("intent")
            filters = routed_intent.get("filters", {})

            st.info(f"**Selected Path:** `{intent}`") # Show the user which path was taken

            # 2. Execute the appropriate path
            if intent == "count_trips":
                response = handle_count_trips(master_df, filters)
            elif intent == "top_destinations":
                response = handle_top_destinations(master_df, filters)
            elif intent == "time_analysis":
                response = handle_time_analysis(master_df, filters)
            elif intent == "age_insights":
                response = handle_age_insights(master_df, filters)
            elif intent == "group_size":
                response = handle_group_size(master_df, filters)
            else: # Fallback to the agent
                with st.spinner("Analyzing complex question with AI agent..."):
                    response = run_agent_query(master_df, user_question)
            
            # 3. Display the response
            st.success(response)
    else:
        st.warning("Please ask a question.")