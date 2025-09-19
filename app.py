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
def get_grouped_destination_stats(df):
    """Cache grouped destination statistics with venue variations"""
    import re

    def normalize_venue_name(address):
        """Extract and normalize the main venue name from an address"""
        if pd.isna(address):
            return "unknown"

        address = str(address).lower()

        # Extract the part before the first comma (usually the venue name)
        venue_part = address.split(',')[0].strip()

        # Remove common suffixes and normalize
        venue_part = re.sub(r"['']s?\s*(pub|bar|restaurant|grill|cafe|coffee|shop|store|center|building)$", "'s", venue_part)
        venue_part = re.sub(r"\s*(pub|bar|restaurant|grill|cafe|coffee|shop|store|center|building)$", "", venue_part)
        venue_part = re.sub(r"['']s?$", "'s", venue_part)  # Standardize possessives

        return venue_part.strip()

    # Get all destinations with their normalized names
    df_with_normalized = df.copy()
    df_with_normalized['normalized_venue'] = df_with_normalized['Drop Off Address'].apply(normalize_venue_name)

    # Group by normalized venue name
    grouped_stats = {}
    for normalized_name in df_with_normalized['normalized_venue'].unique():
        if normalized_name == "unknown":
            continue

        # Get all variations for this normalized venue
        variations = df_with_normalized[df_with_normalized['normalized_venue'] == normalized_name]['Drop Off Address'].value_counts()
        total_count = variations.sum()

        grouped_stats[normalized_name] = {
            'total_count': total_count,
            'variations': variations.to_dict()
        }

    # Sort by total count
    sorted_venues = sorted(grouped_stats.items(), key=lambda x: x[1]['total_count'], reverse=True)

    return dict(sorted_venues)

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
    Handles the 'count_trips' intent with enhanced time and day filtering.
    """
    destination_keyword = filters.get("destination", "").lower()
    specific_day = filters.get("specific_day", "").lower()
    time_comparison = filters.get("time_comparison")

    if not destination_keyword:
        return "**Error:** You need to specify a destination to count trips."

    # Strip common articles for better matching
    destination_keyword = re.sub(r'\b(the|a|an)\s+', '', destination_keyword).strip()

    # Start with destination filtering
    working_df = df[df['Drop Off Address'].str.contains(destination_keyword, na=False)]
    original_dest = filters.get('destination', '').title()
    filter_description = [f"to {original_dest}"]

    # Apply time filters
    if specific_day:
        working_df = working_df[working_df['day_of_week'].str.lower() == specific_day]
        filter_description.append(f"on {specific_day.title()}s")

    if time_comparison == "day_vs_night":
        day_df = working_df[(working_df['hour_of_day'] >= 6) & (working_df['hour_of_day'] < 18)]
        night_df = working_df[(working_df['hour_of_day'] >= 18) | (working_df['hour_of_day'] < 6)]

        response = f"## Trip Count Results - Day vs Night\n\n"
        response += f"**Destination:** *{original_dest}*\n\n"

        day_count = len(day_df)
        night_count = len(night_df)
        total_count = day_count + night_count

        response += f"### **Day Trips (6 AM - 6 PM)**\n"
        response += f"**{day_count:,} trips** to {original_dest}\n\n"

        response += f"### **Night Trips (6 PM - 6 AM)**\n"
        response += f"**{night_count:,} trips** to {original_dest}\n\n"

        response += f"### **Total & Comparison**\n"
        response += f"**{total_count:,} total trips** to {original_dest}\n"

        if day_count > night_count:
            diff = day_count - night_count
            pct = (day_count / total_count) * 100 if total_count > 0 else 0
            response += f"**More day trips** (+{diff:,} trips, {pct:.1f}% during day)\n"
        elif night_count > day_count:
            diff = night_count - day_count
            pct = (night_count / total_count) * 100 if total_count > 0 else 0
            response += f"**More night trips** (+{diff:,} trips, {pct:.1f}% during night)\n"
        else:
            response += f"**Equal day and night trips**\n"

        return response

    elif time_comparison == "weekend_vs_weekday":
        weekend_df = working_df[working_df['day_of_week'].isin(['Saturday', 'Sunday'])]
        weekday_df = working_df[~working_df['day_of_week'].isin(['Saturday', 'Sunday'])]

        response = f"## Trip Count Results - Weekend vs Weekday\n\n"
        response += f"**Destination:** *{original_dest}*\n\n"

        weekend_count = len(weekend_df)
        weekday_count = len(weekday_df)
        total_count = weekend_count + weekday_count

        response += f"### **Weekend Trips**\n"
        response += f"**{weekend_count:,} trips** to {original_dest}\n\n"

        response += f"### **Weekday Trips**\n"
        response += f"**{weekday_count:,} trips** to {original_dest}\n\n"

        response += f"### **Total & Comparison**\n"
        response += f"**{total_count:,} total trips** to {original_dest}\n"

        if weekend_count > weekday_count:
            diff = weekend_count - weekday_count
            pct = (weekend_count / total_count) * 100 if total_count > 0 else 0
            response += f"**More weekend trips** (+{diff:,} trips, {pct:.1f}% on weekends)\n"
        elif weekday_count > weekend_count:
            diff = weekday_count - weekend_count
            pct = (weekday_count / total_count) * 100 if total_count > 0 else 0
            response += f"**More weekday trips** (+{diff:,} trips, {pct:.1f}% on weekdays)\n"
        else:
            response += f"**Equal weekend and weekday trips**\n"

        return response

    else:
        # Standard filtering with optional day filter
        count = len(working_df)

        if count == 0:
            response = f"## Trip Search Results\n\n"
            response += f"**Search criteria:** *{', '.join(filter_description)}*\n\n"
            response += f"**No trips found** matching the specified criteria\n\n"
            response += "*Try searching with different keywords or check the spelling*"
        else:
            response = f"## Trip Count Results\n\n"
            response += f"**Search criteria:** *{', '.join(filter_description)}*\n\n"
            response += f"**{count:,} trips** found matching the criteria\n\n"

            # Show a few examples if there are matches
            if count > 0:
                sample_destinations = working_df['Drop Off Address'].value_counts().head(3)
                response += "**Sample Locations:**\n"
                for dest, dest_count in sample_destinations.items():
                    response += f"- *{dest.title()}* ({dest_count} trips)\n"

                if len(sample_destinations) < count:
                    remaining = count - sample_destinations.sum()
                    response += f"- *...and {remaining} more trips*\n"

                # Add day breakdown if no specific day was requested
                if not specific_day and count > 10:
                    day_breakdown = working_df['day_of_week'].value_counts().head(3)
                    response += f"\n**Popular Days:**\n"
                    for day, day_count in day_breakdown.items():
                        response += f"- *{day}* ({day_count} trips)\n"

        return response

def handle_top_destinations(df, filters):
    """
    Handles the 'top_destinations' intent.
    Finds the most frequent drop-off locations with grouped venue variations.
    """
    top_n = filters.get("limit", 5)
    min_age = filters.get("min_age")
    max_age = filters.get("max_age")

    # Apply age filtering if specified
    if min_age is not None and max_age is not None:
        # Filter trips where average age is in the specified range
        age_filtered_df = df[(df['avg_age'] >= min_age) & (df['avg_age'] <= max_age)]

        if len(age_filtered_df) == 0:
            return f"## Top {top_n} Destinations - Ages {min_age}-{max_age}\n\n**No trips found for riders with average age between {min_age} and {max_age} years.**"

        grouped_stats = get_grouped_destination_stats(age_filtered_df)
        age_suffix = f" - Ages {min_age}-{max_age}"
        total_filtered_trips = len(age_filtered_df)
        avg_age_in_range = age_filtered_df['avg_age'].mean()
    else:
        grouped_stats = get_grouped_destination_stats(df)
        age_suffix = ""
        total_filtered_trips = len(df)
        avg_age_in_range = None

    # Get top N venues by total count
    top_venues = list(grouped_stats.items())[:top_n]

    response = f"## Top {top_n} Most Popular Destinations{age_suffix}\n\n"

    if min_age is not None and max_age is not None:
        response += f"**Age Range:** {min_age}-{max_age} years  \n"
        response += f"**Trips in range:** {total_filtered_trips:,} trips  \n"
        response += f"**Average age:** {avg_age_in_range:.1f} years\n\n"

    for i, (venue_name, stats) in enumerate(top_venues, 1):
        total_count = stats['total_count']
        variations = stats['variations']

        # Format the main venue name nicely
        display_name = venue_name.replace("'s", "'s").title()

        if len(variations) == 1:
            # Single variation - show normally with clean formatting
            address = list(variations.keys())[0]
            location = ', '.join(address.split(',')[1:]).strip().title()
            response += f"### {i}. **{display_name}**\n"
            response += f"*Location:* {location}  \n"
            response += f"**{total_count:,} trips**\n\n"
        else:
            # Multiple variations - show grouped with breakdown
            response += f"### {i}. **{display_name}**\n"
            response += f"**{total_count:,} trips total**\n\n"
            response += "**Address Variations:**\n"
            for j, (address, count) in enumerate(sorted(variations.items(), key=lambda x: x[1], reverse=True), 1):
                # Format address nicely
                venue_detail = address.split(',')[0].strip().title()
                location = ', '.join(address.split(',')[1:]).strip().title()
                response += f"- **{venue_detail}** - *{location}* ({count:,} trips)\n"
            response += "\n"

    response += "---\n"
    if min_age is not None and max_age is not None:
        response += f"*Showing destinations popular with {min_age}-{max_age} year old riders*"
    else:
        response += f"*Showing destinations with the highest ridership volume*"

    return response

def handle_time_analysis(df, filters):
    """
    Handles time-based questions about ride patterns.
    """
    specific_day = filters.get("day", "").lower()

    if specific_day:
        # Filter data for specific day
        day_df = df[df['day_of_week'].str.lower() == specific_day]
        if len(day_df) == 0:
            return f"## Time Analysis - {specific_day.title()}\n\n**No data found for {specific_day.title()}**"

        # Get hourly patterns for this specific day
        hourly_pattern = day_df.groupby('hour_of_day').size()
        peak_hour = hourly_pattern.idxmax() if len(hourly_pattern) > 0 else 0
        peak_count = hourly_pattern.max() if len(hourly_pattern) > 0 else 0
        total_day_trips = len(day_df)

        # Format hour display
        hour_display = f"{peak_hour:02d}:00"
        if peak_hour == 0:
            hour_display = "12:00 AM"
        elif peak_hour < 12:
            hour_display = f"{peak_hour}:00 AM"
        elif peak_hour == 12:
            hour_display = "12:00 PM"
        else:
            hour_display = f"{peak_hour-12}:00 PM"

        response = f"## Time Analysis - {specific_day.title()}\n\n"
        response += f"### **Peak Hour on {specific_day.title()}**\n"
        response += f"**{hour_display}** - {peak_count:,} trips\n\n"
        response += f"### **Total {specific_day.title()} Trips**\n"
        response += f"**{total_day_trips:,} trips** across all hours\n\n"

        # Show top 3 hours for this day
        top_hours = hourly_pattern.nlargest(3)
        response += f"### **Top Hours on {specific_day.title()}**\n"
        for hour, count in top_hours.items():
            if hour == 0:
                hour_display = "12:00 AM"
            elif hour < 12:
                hour_display = f"{hour}:00 AM"
            elif hour == 12:
                hour_display = "12:00 PM"
            else:
                hour_display = f"{hour-12}:00 PM"
            response += f"- **{hour_display}** - {count:,} trips\n"

        return response

    else:
        # Original logic for overall patterns
        time_patterns = get_time_patterns(df)
        peak_hour = time_patterns['hourly'].idxmax()
        peak_count = time_patterns['hourly'].max()
        peak_day = time_patterns['daily'].idxmax()
        peak_day_count = time_patterns['daily'].max()

        # Format hour display
        hour_display = f"{peak_hour:02d}:00"
        if peak_hour == 0:
            hour_display = "12:00 AM"
        elif peak_hour < 12:
            hour_display = f"{peak_hour}:00 AM"
        elif peak_hour == 12:
            hour_display = "12:00 PM"
        else:
            hour_display = f"{peak_hour-12}:00 PM"

        response = f"## Ride Time Analysis\n\n"

        response += f"### **Peak Hour**\n"
        response += f"**{hour_display}** - {peak_count:,} trips\n\n"

        response += f"### **Busiest Day**\n"
        response += f"**{peak_day}** - {peak_day_count:,} trips\n\n"

        # Add some insights
        response += "### **Insights**\n"

        # Time of day insights
        if peak_hour >= 17 and peak_hour <= 22:
            response += "*Evening hours are most popular (likely dinner/nightlife)*\n"
        elif peak_hour >= 11 and peak_hour <= 14:
            response += "*Lunch hours see peak ridership*\n"
        elif peak_hour >= 7 and peak_hour <= 10:
            response += "*Morning commute time is busiest*\n"
        else:
            response += "*Peak time is outside typical patterns*\n"

        # Weekend vs weekday
        if peak_day in ['Saturday', 'Sunday']:
            response += "*Weekend days are most popular*\n"
        else:
            response += "*Weekdays see the highest demand*\n"

        return response

def handle_age_insights(df, filters):
    """
    Handles age-related demographic questions with enhanced filtering.
    """
    destination = filters.get("destination", "").lower()
    specific_day = filters.get("specific_day", "").lower()
    time_comparison = filters.get("time_comparison")
    age_comparison = filters.get("age_comparison")

    # Start with the full dataset
    working_df = df.copy()
    filter_description = []

    # Apply destination filter
    if destination:
        destination = re.sub(r'\b(the|a|an)\s+', '', destination).strip()
        working_df = working_df[working_df['Drop Off Address'].str.contains(destination, na=False)]
        filter_description.append(f"to {filters.get('destination', '').title()}")

    # Apply day filter
    if specific_day:
        working_df = working_df[working_df['day_of_week'].str.lower() == specific_day]
        filter_description.append(f"on {specific_day.title()}s")

    # Handle time comparisons
    if time_comparison == "day_vs_night":
        day_df = working_df[(working_df['hour_of_day'] >= 6) & (working_df['hour_of_day'] < 18)]
        night_df = working_df[(working_df['hour_of_day'] >= 18) | (working_df['hour_of_day'] < 6)]

        response = f"## Age Demographics - Day vs Night"
        if filter_description:
            response += f" ({', '.join(filter_description)})"
        response += "\n\n"

        if len(day_df) > 0:
            day_avg = day_df['avg_age'].mean()
            day_min = day_df['min_age'].min()
            day_max = day_df['max_age'].max()
        else:
            day_avg = day_min = day_max = 0

        if len(night_df) > 0:
            night_avg = night_df['avg_age'].mean()
            night_min = night_df['min_age'].min()
            night_max = night_df['max_age'].max()
        else:
            night_avg = night_min = night_max = 0

        response += f"### **Day Trips (6 AM - 6 PM)**\n"
        response += f"**Average age:** {day_avg:.1f} years\n"
        response += f"**Age range:** {day_min:.0f} - {day_max:.0f} years\n"
        response += f"**Total trips:** {len(day_df):,}\n\n"

        response += f"### **Night Trips (6 PM - 6 AM)**\n"
        response += f"**Average age:** {night_avg:.1f} years\n"
        response += f"**Age range:** {night_min:.0f} - {night_max:.0f} years\n"
        response += f"**Total trips:** {len(night_df):,}\n\n"

        # Comparison
        response += f"### **Comparison**\n"
        if day_avg > night_avg:
            diff = day_avg - night_avg
            response += f"**Day riders are older** on average (+{diff:.1f} years)\n"
        elif night_avg > day_avg:
            diff = night_avg - day_avg
            response += f"**Night riders are older** on average (+{diff:.1f} years)\n"
        else:
            response += f"**Similar ages** between day and night riders\n"

        return response

    elif age_comparison == "young_vs_old":
        # Split into young (under 25) and older (25+) demographics
        young_df = working_df[working_df['avg_age'] < 25]
        older_df = working_df[working_df['avg_age'] >= 25]

        response = f"## Age Demographics - Young vs Older Riders"
        if filter_description:
            response += f" ({', '.join(filter_description)})"
        response += "\n\n"

        response += f"### **Young Riders (Under 25)**\n"
        response += f"**Average age:** {young_df['avg_age'].mean():.1f} years\n"
        response += f"**Total trips:** {len(young_df):,}\n\n"

        response += f"### **Older Riders (25+)**\n"
        response += f"**Average age:** {older_df['avg_age'].mean():.1f} years\n"
        response += f"**Total trips:** {len(older_df):,}\n\n"

        # Show top destinations for each group
        if len(young_df) > 0:
            young_destinations = young_df['Drop Off Address'].value_counts().head(3)
            response += f"### **Popular with Young Riders**\n"
            for dest, count in young_destinations.items():
                response += f"- *{dest.title()}* ({count} trips)\n"
            response += "\n"

        if len(older_df) > 0:
            older_destinations = older_df['Drop Off Address'].value_counts().head(3)
            response += f"### **Popular with Older Riders**\n"
            for dest, count in older_destinations.items():
                response += f"- *{dest.title()}* ({count} trips)\n"

        return response

    else:
        # Standard single-filter analysis
        if len(working_df) == 0:
            return f"## No Data Found\n\n*No trips found matching the specified criteria*"

        avg_age = working_df['avg_age'].mean()
        min_age = working_df['min_age'].min()
        max_age = working_df['max_age'].max()

        response = f"## Age Demographics Analysis"
        if filter_description:
            response += f" ({', '.join(filter_description)})"
        response += "\n\n"

        response += f"### **Average Age**\n"
        response += f"**{avg_age:.1f} years**\n\n"
        response += f"### **Age Range**\n"
        response += f"**Youngest group:** {min_age:.0f} years\n"
        response += f"**Oldest group:** {max_age:.0f} years\n"
        response += f"**Total trips:** {len(working_df):,}\n\n"

        # Age category insight
        if avg_age < 25:
            response += "### **Insight**\n*Attracts a younger crowd (college-age)*"
        elif avg_age < 35:
            response += "### **Insight**\n*Popular with young professionals*"
        elif avg_age < 50:
            response += "### **Insight**\n*Appeals to middle-aged demographics*"
        else:
            response += "### **Insight**\n*Attracts a more mature clientele*"

        return response

def handle_group_size(df, filters):
    """
    Handles questions about group sizes and ride capacity.
    """
    compare_time = filters.get("compare_time")

    if compare_time == "day_vs_night":
        # Load the raw checkins data to get actual passenger counts per trip
        import pandas as pd
        checkins_df = pd.read_excel('fetii_data.xlsx', sheet_name='Checked in User ID\'s')

        # Split trips into day (6 AM - 6 PM) and night (6 PM - 6 AM)
        day_trips = df[(df['hour_of_day'] >= 6) & (df['hour_of_day'] < 18)]
        night_trips = df[(df['hour_of_day'] >= 18) | (df['hour_of_day'] < 6)]

        # Get group sizes for day trips
        day_trip_ids = day_trips['Trip ID'].unique()
        day_checkins = checkins_df[checkins_df['Trip ID'].isin(day_trip_ids)]
        day_trip_sizes = day_checkins.groupby('Trip ID').size()

        # Get group sizes for night trips
        night_trip_ids = night_trips['Trip ID'].unique()
        night_checkins = checkins_df[checkins_df['Trip ID'].isin(night_trip_ids)]
        night_trip_sizes = night_checkins.groupby('Trip ID').size()

        response = f"## Group Size Analysis - Day vs Night\n\n"

        # Day analysis
        if len(day_trip_sizes) > 0:
            day_avg = day_trip_sizes.mean()
            day_large = (day_trip_sizes >= 8).sum()
            day_large_pct = (day_large / len(day_trip_sizes)) * 100
            day_max = day_trip_sizes.max()
        else:
            day_avg = 0
            day_large = 0
            day_large_pct = 0
            day_max = 0

        # Night analysis
        if len(night_trip_sizes) > 0:
            night_avg = night_trip_sizes.mean()
            night_large = (night_trip_sizes >= 8).sum()
            night_large_pct = (night_large / len(night_trip_sizes)) * 100
            night_max = night_trip_sizes.max()
        else:
            night_avg = 0
            night_large = 0
            night_large_pct = 0
            night_max = 0

        response += f"### **Day Trips (6 AM - 6 PM)**\n"
        response += f"**Average group size:** {day_avg:.1f} people\n"
        response += f"**Total trips:** {len(day_trip_sizes):,}\n"
        response += f"**Large groups (8+):** {day_large:,} ({day_large_pct:.1f}%)\n"
        response += f"**Largest group:** {day_max} people\n\n"

        response += f"### **Night Trips (6 PM - 6 AM)**\n"
        response += f"**Average group size:** {night_avg:.1f} people\n"
        response += f"**Total trips:** {len(night_trip_sizes):,}\n"
        response += f"**Large groups (8+):** {night_large:,} ({night_large_pct:.1f}%)\n"
        response += f"**Largest group:** {night_max} people\n\n"

        # Comparison insights
        response += f"### **Comparison**\n"
        if day_avg > night_avg:
            diff = day_avg - night_avg
            response += f"**Day trips have larger groups** on average (+{diff:.1f} people)\n"
        elif night_avg > day_avg:
            diff = night_avg - day_avg
            response += f"**Night trips have larger groups** on average (+{diff:.1f} people)\n"
        else:
            response += f"**Similar group sizes** between day and night\n"

        if day_large_pct > night_large_pct:
            response += f"**Day trips more likely to be large groups** ({day_large_pct:.1f}% vs {night_large_pct:.1f}%)\n"
        elif night_large_pct > day_large_pct:
            response += f"**Night trips more likely to be large groups** ({night_large_pct:.1f}% vs {day_large_pct:.1f}%)\n"

        return response

    else:
        # Original logic for overall group size analysis
        import pandas as pd
        checkins_df = pd.read_excel('fetii_data.xlsx', sheet_name='Checked in User ID\'s')
        trip_sizes = checkins_df.groupby('Trip ID').size()

        large_groups = trip_sizes[trip_sizes >= 8].count()
        medium_groups = trip_sizes[(trip_sizes >= 4) & (trip_sizes < 8)].count()
        small_groups = trip_sizes[trip_sizes < 4].count()
        avg_size = trip_sizes.mean()
        max_size = trip_sizes.max()

        response = f"## Group Size Analysis\n\n"
        response += f"### **Average Group Size**\n"
        response += f"**{avg_size:.1f} people** per trip\n\n"

        response += f"### **Group Distribution**\n"
        response += f"**Small groups (1-3 people):** {small_groups:,} trips\n"
        response += f"**Medium groups (4-7 people):** {medium_groups:,} trips\n"
        response += f"**Large groups (8+ people):** {large_groups:,} trips\n\n"

        response += f"### **Capacity Insights**\n"
        response += f"**Largest group:** {max_size} people\n"

        # Calculate percentages
        total_trips = len(trip_sizes)
        large_pct = (large_groups / total_trips) * 100

        if large_pct > 20:
            response += f"*High demand for large capacity vehicles ({large_pct:.1f}% are large groups)*"
        elif large_pct > 10:
            response += f"*Moderate large group usage ({large_pct:.1f}% are large groups)*"
        else:
            response += f"*Most trips are smaller groups ({large_pct:.1f}% are large groups)*"

        return response

def handle_distance_query(df, filters):
    """
    Handles distance/route-related questions with helpful explanation.
    """
    response = f"## Distance Analysis Not Available\n\n"
    response += f"**Data Available:**\n"
    response += f"- Pickup coordinates (latitude/longitude)\n"
    response += f"- Drop-off coordinates (latitude/longitude)\n"
    response += f"- **{len(df):,} total trips** with coordinate data\n\n"

    response += f"**What's Missing:**\n"
    response += f"- Pre-calculated distances are not included in the dataset\n"
    response += f"- Route information and travel times are not available\n\n"

    response += f"**Alternative Analysis:**\n"
    response += f"Instead of distance-based analysis, I can help with:\n"
    response += f"- **Popular destinations** by trip volume\n"
    response += f"- **Time patterns** (peak hours, busy days)\n"
    response += f"- **Group size analysis** and capacity insights\n"
    response += f"- **Age demographics** by destination or time\n\n"

    response += f"*Try asking: 'What are the most popular destinations?' or 'What time do most people ride?'*"

    return response

def handle_predictive_query(df, filters):
    """
    Handles future/predictive questions that cannot be answered with historical data.
    """
    response = f"## Predictive Analysis Not Available\n\n"
    response += f"**Data Type:** Historical rideshare data\n"
    response += f"**Time Period:** Past trips only (no future data)\n"
    response += f"**Total Records:** {len(df):,} completed trips\n\n"

    response += f"**What I Cannot Predict:**\n"
    response += f"- Future ridership or demand\n"
    response += f"- Next week/weekend trip volumes\n"
    response += f"- Upcoming passenger counts\n"
    response += f"- Future destination popularity\n\n"

    response += f"**What I Can Analyze:**\n"
    response += f"Instead of predicting the future, I can show you **historical patterns**:\n"
    response += f"- **Past weekend trends** - 'How many trips happened on weekends?'\n"
    response += f"- **Historical patterns** - 'What was the busiest time last month?'\n"
    response += f"- **Destination history** - 'How popular was Rainey Street historically?'\n"
    response += f"- **Group size patterns** - 'What was the average group size on weekends?'\n\n"

    response += f"*Try asking about historical data: 'How many people went to Rainey Street on past weekends?'*"

    return response

# --- 3. SMART ROUTING WITH PATTERN MATCHING ---

def extract_universal_filters(question):
    """Extract common filters that can be applied across different intents"""
    question_lower = question.lower()
    filters = {}

    # Time-based filters
    day_match = re.search(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', question_lower)
    if day_match:
        filters["specific_day"] = day_match.group(1)

    # Day vs night comparison
    if re.search(r'(day\s+vs\s+night|night\s+vs\s+day|day.*night.*difference|night.*day.*difference)', question_lower):
        filters["time_comparison"] = "day_vs_night"

    # Weekend vs weekday
    if re.search(r'(weekend.*weekday|weekday.*weekend|weekend.*vs.*weekday|weekday.*vs.*weekend)', question_lower):
        filters["time_comparison"] = "weekend_vs_weekday"

    # Age ranges - multiple patterns to catch different formats
    age_patterns = [
        r'age.*between\s+(\d+)\s+and\s+(\d+)',  # "age between 18 and 21"
        r'for\s+(\d+)-(\d+)\s+year',            # "for 18-21 year olds"
        r'ages?\s+(\d+)-(\d+)',                 # "ages 18-21"
        r'(\d+)\s+to\s+(\d+)\s+year',          # "18 to 21 year olds"
        r'(\d+)-(\d+)\s+year\s+olds?'          # "18-21 year olds"
    ]

    for pattern in age_patterns:
        age_match = re.search(pattern, question_lower)
        if age_match:
            filters["min_age"] = int(age_match.group(1))
            filters["max_age"] = int(age_match.group(2))
            break

    # Young vs old comparison
    if re.search(r'(young.*old|old.*young|young.*vs.*old|old.*vs.*young)', question_lower):
        filters["age_comparison"] = "young_vs_old"

    # Destination filtering - be more specific to avoid false matches
    destination_patterns = [
        r'\btrips?\s+to\s+([^?]+?)(?:\s+(?:on|during|for)|\s*\?|$)',
        r'\bgoing\s+to\s+([^?]+?)(?:\s+(?:on|during|for)|\s*\?|$)',
        r'\bat\s+([a-zA-Z][^?]+?)(?:\s+(?:on|during|for)|\s*\?|$)',
        r'\bhow many.*to\s+([^?]+?)(?:\s+(?:on|during|for)|\s*\?|$)'
    ]
    for pattern in destination_patterns:
        dest_match = re.search(pattern, question_lower)
        if dest_match:
            destination = dest_match.group(1).strip()
            # Clean up common words that aren't part of destination
            destination = re.sub(r'\b(the|a|an)\s+', '', destination).strip()
            # More restrictive validation
            if len(destination) > 2 and not re.search(r'\b(average|age|what|how|when|where|why)\b', destination):
                filters["destination"] = destination
                break

    return filters

def assess_query_complexity(question, filters):
    """Determine if query should go to agent or direct path"""
    question_lower = question.lower()

    # Count distinct filter types
    filter_types = len([k for k in filters.keys() if k not in ['limit']])

    # Complex analytical language
    complex_keywords = [
        r'\bcorrelation\b|\brelationship\b|\bcompare.*between\b',
        r'\bpercentage\b|\bincrease\b|\bdecrease\b|\btrend\b',
        r'\bpredict\b|\bforecast\b|\bif.*then\b',
        r'\banalyz\w+\b|\bstatistic\w+\b|\bcalculat\w+\b',
        r'\bwhy\b|\bhow.*affect\b|\bimpact\b|\bcause\b'
    ]

    # Distance/route questions that should be handled specially
    distance_keywords = [
        r'\bdistance\b|\bmiles?\b|\bkilometer\b|\bkm\b',
        r'\blonger than\b|\bshorter than\b|\bfurther than\b',
        r'\broute\b|\btravel time\b|\bduration\b'
    ]

    # Predictive/future questions that cannot be answered
    predictive_keywords = [
        r'\bwill\b|\bwould\b|\bshall\b|\bgoing to\b',
        r'\bnext\s+(week|month|year|weekend|friday|saturday|sunday)\b',
        r'\btomorrow\b|\blater\b|\bfuture\b|\bupcoming\b',
        r'\bpredict\b|\bforecast\b|\bexpect\b|\bestimate\b'
    ]

    has_distance_query = any(re.search(pattern, question_lower) for pattern in distance_keywords)
    has_predictive_query = any(re.search(pattern, question_lower) for pattern in predictive_keywords)

    has_complex_language = any(re.search(pattern, question_lower) for pattern in complex_keywords)

    # Route distance queries to special handler instead of agent
    if has_distance_query:
        return "distance_query"

    # Route predictive queries to special handler
    if has_predictive_query:
        return "predictive_query"

    # Route to agent if:
    # - 3+ filter types OR
    # - Complex analytical language OR
    # - Very specific multi-dimensional questions
    if filter_types >= 3 or has_complex_language:
        return "agent_fallback"

    return "direct_path"

def quick_pattern_match(question):
    """
    Enhanced pattern matching with universal filter support and complexity assessment.
    """
    question_lower = question.lower()

    # Extract universal filters first
    universal_filters = extract_universal_filters(question)

    # Assess complexity - route to agent if too complex
    complexity = assess_query_complexity(question, universal_filters)
    if complexity == "agent_fallback":
        return {"intent": "agent_fallback", "filters": {}}
    elif complexity == "distance_query":
        return {"intent": "distance_query", "filters": {}}
    elif complexity == "predictive_query":
        return {"intent": "predictive_query", "filters": {}}

    # Intent-specific pattern matching with enhanced filters
    if re.search(r'\bhow many.*trips?.*to\b', question_lower):
        filters = {"destination": universal_filters.get("destination", "")}
        # Add time-based filters
        if "specific_day" in universal_filters:
            filters["specific_day"] = universal_filters["specific_day"]
        if "time_comparison" in universal_filters:
            filters["time_comparison"] = universal_filters["time_comparison"]

        if filters["destination"]:  # Only proceed if we found a destination
            return {"intent": "count_trips", "filters": filters}

    if re.search(r'\b(top\s*\d*\s*|most\s+popular\s*|popular\s*)\s*destinations?\b', question_lower):
        limit_match = re.search(r'top\s*(\d+)', question_lower)
        limit = int(limit_match.group(1)) if limit_match else 5

        filters = {"limit": limit}
        # Add age filtering from universal filters
        if "min_age" in universal_filters and "max_age" in universal_filters:
            filters["min_age"] = universal_filters["min_age"]
            filters["max_age"] = universal_filters["max_age"]

        return {"intent": "top_destinations", "filters": filters}

    if re.search(r'\btime\b|\bhour\b|\bwhen\b|\bpeak\b', question_lower):
        filters = {}
        # Use universal filter for specific day
        if "specific_day" in universal_filters:
            filters["day"] = universal_filters["specific_day"]

        return {"intent": "time_analysis", "filters": filters}

    if re.search(r'\bage\b|\bold\b|\byoung\b', question_lower):
        filters = {}
        # Add all relevant universal filters
        if "destination" in universal_filters:
            filters["destination"] = universal_filters["destination"]
        if "specific_day" in universal_filters:
            filters["specific_day"] = universal_filters["specific_day"]
        if "time_comparison" in universal_filters:
            filters["time_comparison"] = universal_filters["time_comparison"]
        if "age_comparison" in universal_filters:
            filters["age_comparison"] = universal_filters["age_comparison"]

        return {"intent": "age_insights", "filters": filters}

    if re.search(r'\bgroup size\b|\blarge group\b|\bhow many people\b', question_lower):
        filters = {}
        # Add time comparison from universal filters
        if "time_comparison" in universal_filters:
            filters["compare_time"] = universal_filters["time_comparison"]

        return {"intent": "group_size", "filters": filters}

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

# Basic example questions
with st.expander("💡 **Basic Example Questions**", expanded=False):
    example_categories = {
        "📍 Destinations": [
            "How many trips went to the Moody Center?",
            "What are the top 5 most popular destinations?"
        ],
        "⏰ Time Patterns": [
            "What time do most people ride?",
            "What are the busiest days of the week?"
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

with st.expander("🎯 **Advanced Question Examples**", expanded=False):
    st.markdown("*These demonstrate the enhanced filtering and intelligence capabilities*")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⚡ **Simple Enhanced Queries**")
        st.markdown("*Fast responses with enhanced filtering*")

        simple_questions = [
            "What's the average age for day vs night trips?",
            "Top 3 destinations for 18-21 year olds",
            "Group sizes on weekends vs weekdays",
            "How many trips to UT on Fridays?"
        ]

        for i, question in enumerate(simple_questions):
            if st.button(question, key=f"simple_{i}"):
                st.session_state.user_question = question

    with col2:
        st.markdown("#### 🚫 **Handled Edge Cases**")
        st.markdown("*Questions that get helpful explanations*")

        edge_questions = [
            "How many trips were longer than 5 miles?",
            "How many people will go to Rainey Street next weekend?",
            "What's the average trip distance?",
            "Will there be more riders next month?"
        ]

        for i, question in enumerate(edge_questions):
            if st.button(question, key=f"edge_{i}"):
                st.session_state.user_question = question

    st.markdown("---")
    st.markdown("#### 🤖 **Complex Analysis (AI Agent)**")
    st.markdown("*These automatically route to the AI agent for sophisticated analysis*")

    complex_cols = st.columns(2)
    complex_questions = [
        "What's the correlation between age and group size?",
        "Analyze the relationship between destination and time patterns",
        "Calculate the percentage increase in ridership trends",
        "Why do certain destinations attract younger riders?"
    ]

    for i, question in enumerate(complex_questions):
        col_idx = i % 2
        if complex_cols[col_idx].button(question, key=f"complex_{i}"):
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
            elif intent == "distance_query":
                response = handle_distance_query(master_df, filters)
            elif intent == "predictive_query":
                response = handle_predictive_query(master_df, filters)
            else: # Fallback to the agent
                with st.spinner("Analyzing complex question with AI agent..."):
                    response = run_agent_query(master_df, user_question)
            
            # 3. Display the response
            st.success(response)
    else:
        st.warning("Please ask a question.")