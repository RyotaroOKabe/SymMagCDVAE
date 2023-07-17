#%%
import requests
import os

# Define the range of values for i and j
i_values = [1, 2, 3]
j_ranges = {1: range(0, 15), 2: range(0, 4), 3: range(0, 6)}

# Define the directory to save the files
save_directory = '/home/rokabe/data2/generative/database/pbe3d'

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Iterate over the values of i and j
for i in i_values:
    for j in j_ranges[i]:
        # Construct the URL for the database file
        url = f"https://archive.materialscloud.org/record/file?filename=dcgat_{i}_{j:03d}.json.bz2&record_id=1485"

        # Send a GET request to download the file
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save the file with the appropriate filename in the specified directory
            filename = f"dcgat_{i}_{j:03d}.json.bz2"
            file_path = os.path.join(save_directory, filename)
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {url}")

#%%
