import streamlit as st
import requests
import re
import json
import pandas as pd

# --- App Title and Description ---
st.title("Queue Time Data Extractor")
st.write(
    "This app demonstrates fetching data from a URL, "
    "parsing the HTML response to extract specific data from a JavaScript block, "
    "and then displaying it."
)

# --- Data Fetching and Processing ---

# The target URL
url = "https://queue-times.com/parks/31/rides/2844?given_date=2024-07-02"

# Add a button to trigger the data fetching process
if st.button("Fetch and Extract Wait Times"):
    try:
        # Step 1: Make the HTTP GET request to the URL
        with st.spinner("Fetching data from the URL..."):
            response = requests.get(url)
            # Raise an error if the request was unsuccessful
            response.raise_for_status()
        st.success(f"Successfully fetched content from the URL.")

        # Step 2: Use regular expressions to find the specific JavaScript data
        # Look for the chart-4 data more precisely
        html_content = response.text
        pattern = r'new Chartkick\["LineChart"\]\("chart-4",\s*(\[.*?\]),\s*\{'
        match = re.search(pattern, html_content, re.DOTALL)

        if not match:
            st.error("Could not find the target data in the HTML. The website's structure may have changed.")
        else:
            st.success("Found the data block in the HTML.")

            # Step 3: Extract and parse the data
            chart_data_string = match.group(1)

            # Debug: Show what was captured (first 200 chars)
            st.write("Captured data string (first 200 chars):", chart_data_string[:200])

            try:
                chart_data = json.loads(chart_data_string)

                # The data we want is in the first dictionary of the list, under the 'data' key
                # It's in the format: [[hour, wait_time], [hour, wait_time], ...]
                wait_time_list = chart_data[0]['data']

                # Step 4: Display the data in Streamlit
                st.subheader("Extracted Raw Data")
                st.write(wait_time_list)

                # For a better presentation, convert the data to a Pandas DataFrame
                df = pd.DataFrame(wait_time_list, columns=['Hour', 'Average Wait Time (mins)'])
                # Set the 'Hour' column as the index for proper charting
                df = df.set_index('Hour')

                st.subheader("Wait Times in a Table")
                st.dataframe(df)

                st.subheader("Wait Times as a Line Chart")
                st.line_chart(df)

            except json.JSONDecodeError as e:
                st.error(f"JSON parsing error: {e}")
                st.write("Raw captured string:", chart_data_string)

    except requests.exceptions.RequestException as e:
        st.error(f"Error during request: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")