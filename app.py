import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.cluster import KMeans
import folium

# Load the dataset with a specified encoding
data = pd.read_csv('kijiji_cleaned.csv', encoding='latin1')
# Convert the 'Date Posted' column to datetime
data['Date Posted'] = pd.to_datetime(data['Date Posted'])


# Extract additional components if needed
data['Year'] = data['Date Posted'].dt.year
data['Month'] = data['Date Posted'].dt.month
data['Day'] = data['Date Posted'].dt.day


# Define functions for each page
def dashboard():
    st.image('Logo.PNG.png', use_column_width=True)
    st.subheader("💡 Agba Baller: Luxury Or Nothing")
    inspiration = '''
    The Luxury Property Listing App (aka Agba Baller) is a sophisticated data-driven platform designed to provide detailed market analysis, property valuations, and investment opportunities in the luxury real estate market. Specifically focused on property listings in Ontario with rental prices exceeding 4000 CAD, this app leverages advanced data analytics and machine learning techniques to offer valuable insights to investors, property managers, and prospective tenants.
    '''
    st.write(inspiration)
    st.subheader("👨🏻‍💻 App Features")
    what_it_does = '''
    The app features several key functionalities:

Dashboard: An intuitive interface that provides an overview of the app's capabilities, objectives, and key metrics. Users can quickly understand the scope of the app and its benefits for luxury real estate analysis.

Exploratory Data Analysis (EDA): This section offers a comprehensive exploration of the dataset, showcasing various trends and distributions within the luxury rental market. Users can view histograms, box plots, and other visualizations to gain insights into rental prices, property types, and geographical distributions.

Market Trends: Users can track the evolution of luxury rental prices over time. This feature presents time-series analyses and trend lines, helping investors understand market dynamics and predict future price movements.

Property Value Prediction: Leveraging machine learning models, this feature allows users to predict the rental value of luxury properties based on specific attributes such as location, property type, size, and amenities. This tool is invaluable for both property owners looking to price their listings accurately and potential tenants seeking to understand the market value.

Investment Opportunities: This section highlights potential investment opportunities within the luxury rental market. By identifying areas with high rental yields and emerging trends, the app aids investors in making informed decisions about where to allocate their resources.

Geographical Analysis: Using interactive maps, this feature provides a visual representation of luxury property listings across Ontario. Users can explore the spatial distribution of high-end rentals, identify clusters of luxury properties, and analyze geographical trends.
    '''
    st.write(what_it_does)

def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    # Price Distribution
    fig = px.histogram(data, x='Price', nbins=20, title='Distribution of Rental Prices')
    st.plotly_chart(fig)
    # Boxplot for Price by Property Type
    fig = px.box(data, x='Type', y='Price', title='Price Distribution by Property Type')
    st.plotly_chart(fig)

def market_trends():
    st.title("Market Trends")
    # Extract YearMonth for aggregation
    data['YearMonth'] = data['Date Posted'].dt.to_period('M').astype(str)
    monthly_trend = data.groupby('YearMonth')['Price'].mean().reset_index()
    fig = px.line(monthly_trend, x='YearMonth', y='Price', title='Average Rental Price Over Time')
    st.plotly_chart(fig)

def property_value_prediction():
    st.title("Property Value Prediction")
    st.write("Enter the details of the property to predict its rental price:")
    property_type = st.selectbox("Type of Property", ['Apartment', 'House', 'Condo', 'Townhouse'])
    bedrooms = st.slider("Number of Bedrooms", 1, 5, 2)
    bathrooms = st.slider("Number of Bathrooms", 1, 3, 1)
    size = st.slider("Size (sqft)", 300, 5000, 1000)
    unique_locations = data['CSDNAME'].unique()
    location = st.selectbox("Location", unique_locations)

    if st.button("Predict"):
        model = joblib.load('random_forest_regressor_model.pkl')
        input_df = pd.DataFrame({
            'Type': [property_type],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Size': [size],
            'CSDNAME': [location]
        })
        prediction = model.predict(input_df)
        st.success(f"Predicted Rental Price: ${prediction[0]:,.2f}")

def investment_opportunities():
    st.title("Investment Opportunities")
    # Highlight top 10 most expensive regions
    top_10 = data.groupby('CSDNAME')['Price'].mean().nlargest(10).reset_index()
    fig = px.bar(top_10, x='CSDNAME', y='Price', title='Top 10 Most Expensive Regions')
    st.plotly_chart(fig)

def geographical_analysis():
    st.title("Geographical Analysis")
    kmeans = KMeans(n_clusters=5, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[['Latitude', 'Longitude', 'Price']])
    max_cluster = data['Cluster'].max()
    data['Cluster'] = max_cluster - data['Cluster']
    fig = px.scatter_mapbox(data, lat='Latitude', lon='Longitude', color='Cluster', 
                            size='Price', hover_name='CSDNAME', zoom=10,
                            mapbox_style='open-street-map')
    fig.update_layout(title='Rental Prices Clustered by Price (Higher numbers = Higher Prices)')
    st.plotly_chart(fig)

# Main App Logic
def main():
    st.sidebar.title("Kijiji Community App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "Market Trends", "Property Value Prediction", "Investment Opportunities", "Geographical Analysis"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "Market Trends":
        market_trends()
    elif app_page == "Property Value Prediction":
        property_value_prediction()
    elif app_page == "Investment Opportunities":
        investment_opportunities()
    elif app_page == "Geographical Analysis":
        geographical_analysis()

if __name__ == "__main__":
    main()
