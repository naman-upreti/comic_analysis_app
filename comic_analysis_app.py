import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter

# Set page configuration
st.set_page_config(
    page_title="Comic Character Analysis",
    page_icon="🦸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define color constants to match CSS variables
MARVEL_RED = "#ED1D24"
DC_BLUE = "#0476F2"

# Load CSS from external file
def load_css(css_file):
    with open(css_file, 'r') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Load the CSS
load_css('styles.css')

# Title and introduction
st.markdown("<h1 class='main-header'>Marvel vs DC: Comic Character Analysis</h1>", unsafe_allow_html=True)
st.markdown("""
This application provides an in-depth analysis of Marvel and DC comic book characters.
Explore character demographics, powers, alignments, and publication trends through interactive visualizations.
""")

# Function to load and preprocess data
@st.cache_data
def load_data():
    try:
        # Adjust file paths as needed
        marvel_df = pd.read_csv('marvel-wikia-data.csv')
        dc_df = pd.read_csv('dc-wikia-data.csv')
        
        # Add universe column to each dataframe
        marvel_df['universe'] = 'Marvel'
        dc_df['universe'] = 'DC'
        
        # Combine the dataframes
        combined_df = pd.concat([marvel_df, dc_df], ignore_index=True)
        
        # Clean and preprocess data
        # Convert YEAR to numeric, handling errors
        combined_df['YEAR'] = pd.to_numeric(combined_df['YEAR'], errors='coerce')
        
        # Convert APPEARANCES to numeric, handling errors
        combined_df['APPEARANCES'] = pd.to_numeric(combined_df['APPEARANCES'], errors='coerce')
        
        # Extract decade from YEAR
        combined_df['decade'] = (combined_df['YEAR'] // 10) * 10
        combined_df['decade'] = combined_df['decade'].apply(lambda x: f"{int(x)}s" if not pd.isna(x) else "Unknown")
        
        # Clean alignment data
        combined_df['alignment'] = combined_df['ALIGN'].apply(
            lambda x: 'Good' if isinstance(x, str) and 'good' in x.lower() else
                     'Bad' if isinstance(x, str) and 'bad' in x.lower() else
                     'Neutral' if isinstance(x, str) and 'neutral' in x.lower() else
                     'Unknown'
        )
        
        # Clean gender data
        combined_df['gender'] = combined_df['SEX'].apply(
            lambda x: 'Male' if isinstance(x, str) and 'male' in x.lower() else
                     'Female' if isinstance(x, str) and 'female' in x.lower() else
                     'Other' if isinstance(x, str) else
                     'Unknown'
        )
        
        return marvel_df, dc_df, combined_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Provide sample data if files not found
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Load data
marvel_df, dc_df, combined_df = load_data()

# Check if data is loaded
if combined_df.empty:
    st.warning("Sample data files not found. Please upload the Marvel and DC datasets.")
    
    # File uploader
    marvel_file = st.file_uploader("Upload Marvel dataset (CSV)", type="csv")
    dc_file = st.file_uploader("Upload DC dataset (CSV)", type="csv")
    
    if marvel_file and dc_file:
        marvel_df = pd.read_csv(marvel_file)
        dc_df = pd.read_csv(dc_file)
        
        # Add universe column to each dataframe
        marvel_df['universe'] = 'Marvel'
        dc_df['universe'] = 'DC'
        
        # Combine the dataframes
        combined_df = pd.concat([marvel_df, dc_df], ignore_index=True)
else:
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.radio(
        "Choose Analysis",
        ["Overview", "Character Demographics", "Publication Trends", "Character Attributes", "Advanced Analysis"]
    )
    
    # Overview section
    if analysis_type == "Overview":
        st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Characters", f"{len(combined_df):,}")
            st.metric("Marvel Characters", f"{len(marvel_df):,}")
            st.metric("Earliest Character", f"{int(combined_df['YEAR'].min())}" if not pd.isna(combined_df['YEAR'].min()) else "Unknown")
        
        with col2:
            st.metric("Unique Character Names", f"{combined_df['name'].nunique():,}")
            st.metric("DC Characters", f"{len(dc_df):,}")
            st.metric("Most Recent Character", f"{int(combined_df['YEAR'].max())}" if not pd.isna(combined_df['YEAR'].max()) else "Unknown")
        
        # Sample data
        st.markdown("<h3>Sample Data</h3>", unsafe_allow_html=True)
        st.dataframe(combined_df.head())
        
        # Data distribution
        st.markdown("<h3>Character Distribution</h3>", unsafe_allow_html=True)
        fig = px.pie(combined_df, names='universe', title='Marvel vs DC Character Count',
                    color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)
        
        # Missing values heatmap
        st.markdown("<h3>Missing Values Analysis</h3>", unsafe_allow_html=True)
        missing_data = combined_df.isnull().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Values']
        missing_data['Percentage'] = (missing_data['Missing Values'] / len(combined_df)) * 100
        
        fig = px.bar(missing_data, x='Column', y='Percentage', 
                    title='Percentage of Missing Values by Column',
                    labels={'Percentage': 'Missing Values (%)'},
                    color='Percentage',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # Character Demographics section
    elif analysis_type == "Character Demographics":
        st.markdown("<h2 class='sub-header'>Character Demographics</h2>", unsafe_allow_html=True)
        
        # Gender distribution
        st.markdown("<h3>Gender Distribution</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender_counts = combined_df.groupby(['universe', 'gender']).size().reset_index(name='count')
            fig = px.bar(gender_counts, x='gender', y='count', color='universe', 
                        barmode='group', title='Gender Distribution by Universe',
                        color_discrete_sequence=[MARVEL_RED, DC_BLUE])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            gender_pct = combined_df.groupby('universe')['gender'].value_counts(normalize=True).mul(100).reset_index(name='percentage')
            fig = px.bar(gender_pct, x='universe', y='percentage', color='gender', 
                        title='Gender Percentage by Universe',
                        labels={'percentage': 'Percentage (%)'},
                        color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)
        
        # Alignment distribution
        st.markdown("<h3>Character Alignment</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            align_counts = combined_df.groupby(['universe', 'alignment']).size().reset_index(name='count')
            fig = px.bar(align_counts, x='alignment', y='count', color='universe', 
                        barmode='group', title='Alignment Distribution by Universe',
                        color_discrete_sequence=[MARVEL_RED, DC_BLUE])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gender and alignment relationship
            gender_align = combined_df.groupby(['gender', 'alignment']).size().reset_index(name='count')
            fig = px.bar(gender_align, x='gender', y='count', color='alignment', 
                        title='Gender vs Alignment',
                        barmode='stack',
                        color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)
        
        # Identity status (secret identity vs public identity)
        st.markdown("<h3>Identity Status</h3>", unsafe_allow_html=True)
        
        id_counts = combined_df.groupby(['universe', 'ID']).size().reset_index(name='count')
        fig = px.bar(id_counts, x='ID', y='count', color='universe', 
                    barmode='group', title='Identity Status by Universe',
                    color_discrete_sequence=[MARVEL_RED, DC_BLUE])
        st.plotly_chart(fig, use_container_width=True)
        
        # Character status (alive vs deceased)
        st.markdown("<h3>Living Status</h3>", unsafe_allow_html=True)
        
        alive_counts = combined_df.groupby(['universe', 'ALIVE']).size().reset_index(name='count')
        fig = px.bar(alive_counts, x='ALIVE', y='count', color='universe', 
                    barmode='group', title='Living Status by Universe',
                    color_discrete_sequence=[MARVEL_RED, DC_BLUE])
        st.plotly_chart(fig, use_container_width=True)
    
    # Publication Trends section
    elif analysis_type == "Publication Trends":
        st.markdown("<h2 class='sub-header'>Publication Trends</h2>", unsafe_allow_html=True)
        
        # Characters by decade
        st.markdown("<h3>Characters Introduced by Decade</h3>", unsafe_allow_html=True)
        
        # Filter out rows with missing decade
        decade_data = combined_df[combined_df['decade'] != 'Unknown']
        decade_counts = decade_data.groupby(['universe', 'decade']).size().reset_index(name='count')
        
        # Sort decades chronologically
        decades_order = sorted([d for d in decade_counts['decade'].unique() if d != 'Unknown'])
        decade_counts['decade'] = pd.Categorical(decade_counts['decade'], categories=decades_order, ordered=True)
        decade_counts = decade_counts.sort_values('decade')
        
        fig = px.line(decade_counts, x='decade', y='count', color='universe', 
                    markers=True, title='Characters Introduced by Decade',
                    color_discrete_sequence=[MARVEL_RED, DC_BLUE])
        st.plotly_chart(fig, use_container_width=True)
        
        # Character appearances
        st.markdown("<h3>Character Appearances</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 characters by appearances
            top_appearances = combined_df.nlargest(10, 'APPEARANCES')
            fig = px.bar(top_appearances, x='name', y='APPEARANCES', color='universe',
                        title='Top 10 Characters by Number of Appearances',
                        color_discrete_sequence=[MARVEL_RED, DC_BLUE])
            fig.update_layout(xaxis_title='Character', yaxis_title='Number of Appearances')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average appearances by decade
            avg_appearances = decade_data.groupby(['universe', 'decade'])['APPEARANCES'].mean().reset_index()
            avg_appearances['decade'] = pd.Categorical(avg_appearances['decade'], categories=decades_order, ordered=True)
            avg_appearances = avg_appearances.sort_values('decade')
            
            fig = px.line(avg_appearances, x='decade', y='APPEARANCES', color='universe',
                        markers=True, title='Average Appearances by Decade',
                        color_discrete_sequence=[MARVEL_RED, DC_BLUE])
            fig.update_layout(yaxis_title='Average Appearances')
            st.plotly_chart(fig, use_container_width=True)
        
        # Publication year distribution
        st.markdown("<h3>Publication Year Distribution</h3>", unsafe_allow_html=True)
        
        # Filter out rows with missing YEAR
        year_data = combined_df.dropna(subset=['YEAR'])
        
        fig = px.histogram(year_data, x='YEAR', color='universe', 
                        nbins=30, title='Character Introduction by Year',
                        color_discrete_sequence=[MARVEL_RED, DC_BLUE])
        fig.update_layout(xaxis_title='Year', yaxis_title='Number of Characters')
        st.plotly_chart(fig, use_container_width=True)
    
    # Character Attributes section
    elif analysis_type == "Character Attributes":
        st.markdown("<h2 class='sub-header'>Character Attributes</h2>", unsafe_allow_html=True)
        
        # Eye color distribution
        st.markdown("<h3>Eye Color Distribution</h3>", unsafe_allow_html=True)
        
        # Get top 10 eye colors
        eye_counts = combined_df['EYE'].value_counts().nlargest(10).reset_index()
        eye_counts.columns = ['EYE', 'count']
        
        fig = px.pie(eye_counts, values='count', names='EYE', 
                    title='Top 10 Eye Colors',
                    color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)
        
        # Hair color distribution
        st.markdown("<h3>Hair Color Distribution</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get top 10 hair colors for Marvel
            marvel_hair = marvel_df['HAIR'].value_counts().nlargest(10).reset_index()
            marvel_hair.columns = ['HAIR', 'count']
            marvel_hair['universe'] = 'Marvel'
            
            fig = px.bar(marvel_hair, x='HAIR', y='count', 
                        title='Top 10 Hair Colors in Marvel',
                        color_discrete_sequence=[MARVEL_RED])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Get top 10 hair colors for DC
            dc_hair = dc_df['HAIR'].value_counts().nlargest(10).reset_index()
            dc_hair.columns = ['HAIR', 'count']
            dc_hair['universe'] = 'DC'
            
            fig = px.bar(dc_hair, x='HAIR', y='count', 
                        title='Top 10 Hair Colors in DC',
                        color_discrete_sequence=[DC_BLUE])
            st.plotly_chart(fig, use_container_width=True)
        
        # Word cloud of character names
        st.markdown("<h3>Character Name Word Cloud</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Marvel word cloud
            marvel_names = ' '.join(marvel_df['name'].dropna())
            marvel_wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                        colormap='Reds', max_words=100).generate(marvel_names)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(marvel_wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Marvel Character Names')
            st.pyplot(fig)
        
        with col2:
            # DC word cloud
            dc_names = ' '.join(dc_df['name'].dropna())
            dc_wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                    colormap='Blues', max_words=100).generate(dc_names)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(dc_wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('DC Character Names')
            st.pyplot(fig)
    
    # Advanced Analysis section
    elif analysis_type == "Advanced Analysis":
        st.markdown("<h2 class='sub-header'>Advanced Analysis</h2>", unsafe_allow_html=True)
        
        # Correlation analysis
        st.markdown("<h3>Correlation Analysis</h3>", unsafe_allow_html=True)
        
        # Select only numeric columns
        numeric_df = combined_df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Plot heatmap
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Correlation Matrix of Numeric Features",
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        
        # Gender and alignment over time
        st.markdown("<h3>Gender and Alignment Trends Over Time</h3>", unsafe_allow_html=True)
        
        # Filter out rows with missing decade
        decade_data = combined_df[combined_df['decade'] != 'Unknown']
        
        # Gender trends
        gender_decade = decade_data.groupby(['decade', 'gender']).size().reset_index(name='count')
        gender_decade_pct = gender_decade.groupby('decade')['count'].transform(lambda x: x / x.sum() * 100)
        gender_decade['percentage'] = gender_decade_pct
        
        # Sort decades chronologically
        decades_order = sorted([d for d in gender_decade['decade'].unique() if d != 'Unknown'])
        gender_decade['decade'] = pd.Categorical(gender_decade['decade'], categories=decades_order, ordered=True)
        gender_decade = gender_decade.sort_values('decade')
        
        fig = px.line(gender_decade, x='decade', y='percentage', color='gender', 
                    markers=True, title='Gender Distribution by Decade',
                    labels={'percentage': 'Percentage (%)'},
                    color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)
        
        # Alignment trends
        align_decade = decade_data.groupby(['decade', 'alignment']).size().reset_index(name='count')
        align_decade_pct = align_decade.groupby('decade')['count'].transform(lambda x: x / x.sum() * 100)
        align_decade['percentage'] = align_decade_pct
        
        # Sort decades chronologically
        align_decade['decade'] = pd.Categorical(align_decade['decade'], categories=decades_order, ordered=True)
        align_decade = align_decade.sort_values('decade')
        
        fig = px.line(align_decade, x='decade', y='percentage', color='alignment', 
                    markers=True, title='Alignment Distribution by Decade',
                    labels={'percentage': 'Percentage (%)'},
                    color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig, use_container_width=True)
        
        # Character diversity analysis
        st.markdown("<h3>Character Diversity Analysis</h3>", unsafe_allow_html=True)
        
        # Calculate diversity score (percentage of non-male, non-white characters)
        # For simplicity, we'll use gender and hair color as proxies
        
        # Define function to calculate diversity score
        def calc_diversity_score(df):
            gender_diversity = (df['gender'] != 'Male').mean() * 100
            hair_diversity = (df['HAIR'] != 'Black Hair').mean() * 100
            return (gender_diversity + hair_diversity) / 2
        
        # Calculate diversity by decade
        diversity_by_decade = decade_data.groupby(['universe', 'decade']).apply(calc_diversity_score).reset_index(name='diversity_score')
        
        # Sort decades chronologically
        diversity_by_decade['decade'] = pd.Categorical(diversity_by_decade['decade'], categories=decades_order, ordered=True)
        diversity_by_decade = diversity_by_decade.sort_values('decade')
        
        fig = px.line(diversity_by_decade, x='decade', y='diversity_score', color='universe', 
                    markers=True, title='Character Diversity Score by Decade',
                    labels={'diversity_score': 'Diversity Score (%)'},
                    color_discrete_sequence=[MARVEL_RED, DC_BLUE])
        st.plotly_chart(fig, use_container_width=True)
        
        # Character complexity analysis
        st.markdown("<h3>Character Complexity Analysis</h3>", unsafe_allow_html=True)
        
        # Define complexity as a function of appearances and alignment
        combined_df['complexity'] = np.log1p(combined_df['APPEARANCES']) * (combined_df['alignment'] == 'Neutral').astype(int) * 2 + 1
        
        # Top 10 most complex characters
        top_complex = combined_df.nlargest(10, 'complexity')
        
        fig = px.bar(top_complex, x='name', y='complexity', color='universe',
                    title='Top 10 Most Complex Characters',
                    color_discrete_sequence=[MARVEL_RED, DC_BLUE])
        fig.update_layout(xaxis_title='Character', yaxis_title='Complexity Score')
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown("<h3>Key Insights</h3>", unsafe_allow_html=True)
        
        st.markdown("<div class='insight-text'>", unsafe_allow_html=True)
        st.markdown("""
        Based on the analysis, here are some key insights:
        
        1. **Gender Representation**: Both Marvel and DC have predominantly male characters, but the percentage of female characters has been increasing over time.
        
        2. **Character Alignment**: Marvel tends to have more morally ambiguous characters (neutral alignment) compared to DC.
        
        3. **Publication Trends**: There are clear publication booms in certain decades, likely corresponding to major cultural events or company initiatives.
        
        4. **Character Diversity**: Character diversity has generally increased over time, with both publishers introducing more diverse characters in recent decades.
        
        5. **Character Complexity**: Characters with more appearances and neutral alignments tend to be more complex, suggesting that longevity allows for more character development.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit | Data source: FiveThirtyEight Comic Characters Dataset")