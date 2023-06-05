import requests
import os

owner = 'MBAP'
repo = 'fastsim'
folder_path = 'python/fastsim/resources/vehdb'

url = f'https://api.github.nrel.gov/repos/{owner}/{repo}/contents/{folder_path}'

# Send the GET request
response = requests.get(url)
print(response.status_code)

try:
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the response content as JSON
        data = response.json()

        # Create a folder to save the downloaded files
        save_folder = 'Vehicles'
        os.makedirs(save_folder, exist_ok=True)

        # Filter and download files
        for item in data:
            if item['type'] == 'file' and (item['name'].endswith('.csv') or item['name'].endswith('.yaml')):
                file_url = item['download_url']
                file_name = item['name']

                # Send a GET request to download the file
                file_response = requests.get(file_url)

                # Save the file in the "Vehicles" folder
                save_path = os.path.join(save_folder, file_name)
                with open(save_path, 'wb') as file:
                    file.write(file_response.content)
                    print(f"File {file_name} downloaded and saved.")

        print("All files downloaded and saved successfully.")
    else:
        print(f"Request failed with status code: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")