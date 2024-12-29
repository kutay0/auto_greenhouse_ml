import pandas as pd
import random
import matplotlib.pyplot as plt

# Function to load data from a text file (temperature, moisture, light)
def load_data_from_file(file_name):
    try:
        with open(file_name, 'r') as file:
            data = [float(line.strip()) for line in file.readlines()]
        return data
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return []

# Function to load plant database and return optimal ranges based on plant
def load_plant_database(file_name):
    plant_db = {}
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip() and not line.startswith("Plant"):  # Avoid empty lines and header
                    parts = line.split()
                    plant_name = parts[0]
                    moisture_range = parts[1]
                    light_range = parts[2]
                    temp_range = parts[3]

                    # Store the optimal ranges as tuples
                    plant_db[plant_name.lower()] = {
                        'moisture': moisture_range,
                        'light': light_range,
                        'temperature': temp_range
                    }
        return plant_db
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return {}

# Function to convert string ranges (like '750–900') to actual range tuples (750, 900)
def convert_range_to_tuple(range_str):
    try:
        # Handle both '–' and '-' as the separator in ranges
        start, end = range_str.replace('–', '-').split('-')
        return int(start), int(end)
    except ValueError:
        print(f"Error converting range: {range_str}")
        return None  # Return None if the conversion fails

# Load the temperature, moisture, and light data
temperature_data = load_data_from_file('temp.txt')
moisture_data = load_data_from_file('moisture.txt')
light_level_data = load_data_from_file('light.txt')

# Check if the data has the expected length (25)
if len(temperature_data) != 25 or len(moisture_data) != 25 or len(light_level_data) != 25:
    print("Error: The data should contain exactly 25 values for each parameter.")
else:
    # Create a DataFrame with the loaded data
    data = {
        'temperature': temperature_data,
        'light_level': light_level_data,
        'moisture': moisture_data
    }

    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Load the plant database
    plant_db = load_plant_database('plantdb.txt')

    if not plant_db:
        print("No plant data available.")
    else:
        # Display the available plants to the user
        print("Available plants:")
        for plant in plant_db:
            print(f"- {plant.capitalize()}")

        # Ask the user for the plant name (case-insensitive)
        plant_name = ''
        while plant_name not in plant_db:
            plant_name = input("\nEnter the plant name (choose from the list above): ").strip().lower()
            if plant_name not in plant_db:
                print("Plant not found. Please choose a valid plant from the list.")

        # Retrieve the optimal ranges for the selected plant
        optimal_plant_data = plant_db[plant_name]

    # Perform K-means clustering manually (without sklearn)
    def euclidean_distance(p1, p2):
        return sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))) ** 0.5

    def kmeans_clustering(df, k, max_iterations=100):
        # Initialize centroids randomly
        centroids = [df.iloc[random.randint(0, len(df) - 1)].values for _ in range(k)]

        for _ in range(max_iterations):
            clusters = [[] for _ in range(k)]

            # Assign points to the nearest centroid
            for i in range(len(df)):
                distances = [euclidean_distance(df.iloc[i].values, centroid) for centroid in centroids]
                cluster_index = distances.index(min(distances))
                clusters[cluster_index].append(i)

            # Update centroids
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    new_centroids.append(df.iloc[cluster].mean().values)
                else:
                    new_centroids.append(centroids[len(new_centroids)])

            if all((new_centroids[i] == centroids[i]).all() for i in range(k)):
                break

            centroids = new_centroids

        return clusters, centroids

    # Apply clustering
    clusters, centroids = kmeans_clustering(df, k=3)

    # Assign cluster labels
    cluster_labels = [0] * len(df)
    for cluster_idx, cluster in enumerate(clusters):
        for index in cluster:
            cluster_labels[index] = cluster_idx

    df['cluster'] = cluster_labels

    # Instead of anomaly detection, now label as Optimal or Non-optimal
    optimal_threshold = 2.0  # Distance threshold for optimal conditions
    condition_labels = []

    for i in range(len(df)):
        point = df.iloc[i].values[:-1]  # Exclude cluster label
        cluster_idx = int(df.iloc[i]['cluster'])  # Ensure the cluster index is an integer
        centroid = centroids[cluster_idx]  # Access the centroid using an integer index
        distance = euclidean_distance(point, centroid)
        
        # Compare distance to threshold and label as Optimal or Non-optimal
        if distance <= optimal_threshold:
            condition_labels.append("Optimal")
        else:
            condition_labels.append("Non-optimal")

    # Add condition label to DataFrame
    df['condition'] = condition_labels

    # Display the dataframe with condition labels before suggestion
    print("\nEnvironmental Conditions with Clusters and Conditions:")
    print(df)

    # Calculate median and standard deviation for temperature, light, and moisture
    median_temperature = df['temperature'].median()
    median_light_level = df['light_level'].median()
    median_moisture = df['moisture'].median()

    std_temperature = df['temperature'].std()
    std_light_level = df['light_level'].std()
    std_moisture = df['moisture'].std()

    print(f"\nMedian and Standard Deviation of Environmental Conditions:")
    print(f"Temperature - Median: {median_temperature}, Std Dev: {std_temperature}")
    print(f"Light Level - Median: {median_light_level}, Std Dev: {std_light_level}")
    print(f"Moisture - Median: {median_moisture}, Std Dev: {std_moisture}")

    # Display optimal conditions for the selected plant
    optimal_selected_plant_data = plant_db[plant_name]
    select_plant_moisture_range = convert_range_to_tuple(optimal_selected_plant_data['moisture'])
    select_plant_light_range = convert_range_to_tuple(optimal_selected_plant_data['light'])
    select_plant_temp_range = convert_range_to_tuple(optimal_selected_plant_data['temperature'])

    print(f"\nOptimal Conditions for Selected Plant ({plant_name.capitalize()}):")
    print(f"Temperature: {select_plant_temp_range[0]}-{select_plant_temp_range[1]}")
    print(f"Light Level: {select_plant_light_range[0]}-{select_plant_light_range[1]}")
    print(f"Moisture: {select_plant_moisture_range[0]}-{select_plant_moisture_range[1]}\n")

    # Suggest the best plant based on median and optimal plant ranges
    best_plant = None
    best_distance = float('inf')

    # Calculate distance to suggest the best plant
    for db_plant_name, ranges in plant_db.items():
        plant_moisture_range = convert_range_to_tuple(ranges['moisture'])
        plant_light_range = convert_range_to_tuple(ranges['light'])
        plant_temp_range = convert_range_to_tuple(ranges['temperature'])

        # Calculate distance from median values to plant's optimal ranges
        distance = 0
        if not (plant_moisture_range[0] <= median_moisture <= plant_moisture_range[1]):
            distance += abs(median_moisture - (plant_moisture_range[0] + plant_moisture_range[1]) / 2)
        if not (plant_light_range[0] <= median_light_level <= plant_light_range[1]):
            distance += abs(median_light_level - (plant_light_range[0] + plant_light_range[1]) / 2)
        if not (plant_temp_range[0] <= median_temperature <= plant_temp_range[1]):
            distance += abs(median_temperature - (plant_temp_range[0] + plant_temp_range[1]) / 2)

        # Update best plant if the distance is lower than the current best
        if distance < best_distance:
            best_distance = distance
            best_plant = db_plant_name  # Assign the best plant name from the loop

    # Suggest the best plant based on the closest match to the median conditions
    print(f"\nThe closest matching plant for the current environmental conditions is: {best_plant.capitalize()}\n")

    # Retrieve optimal conditions for the best plant
    optimal_best_plant_data = plant_db[best_plant]
    best_plant_moisture_range = convert_range_to_tuple(optimal_best_plant_data['moisture'])
    best_plant_light_range = convert_range_to_tuple(optimal_best_plant_data['light'])
    best_plant_temp_range = convert_range_to_tuple(optimal_best_plant_data['temperature'])

    # Display optimal conditions for the best plant
    print(f"Optimal Conditions for Best Plant ({best_plant.capitalize()}):")
    print(f"Temperature: {best_plant_temp_range[0]}-{best_plant_temp_range[1]}")
    print(f"Light Level: {best_plant_light_range[0]}-{best_plant_light_range[1]}")
    print(f"Moisture: {best_plant_moisture_range[0]}-{best_plant_moisture_range[1]}\n")

    # Plotting the current environmental conditions and the optimal plant ranges
    def plot_comparison(median_temperature, median_light_level, median_moisture, 
                        plant_name, plant_moisture_range, plant_light_range, plant_temp_range):
        # Create a figure and axis
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Overall title for the figure
        fig.suptitle(f'Environmental Conditions vs Optimal Ranges for {plant_name.capitalize()}', fontsize=16)

        # Temperature plot
        axes[0].bar(['Current', 'Optimal Min', 'Optimal Max'], 
                    [median_temperature, plant_temp_range[0], plant_temp_range[1]], 
                    color=['blue', 'green', 'green'])
        axes[0].set_title(f'Temperature Comparison - {plant_name.capitalize()}')
        axes[0].set_ylabel('Temperature (°C)')
        
        # Light level plot
        axes[1].bar(['Current', 'Optimal Min', 'Optimal Max'], 
                    [median_light_level, plant_light_range[0], plant_light_range[1]], 
                    color=['blue', 'green', 'green'])
        axes[1].set_title(f'Light Level Comparison - {plant_name.capitalize()}')
        axes[1].set_ylabel('Light Level')
        
        # Moisture plot
        axes[2].bar(['Current', 'Optimal Min', 'Optimal Max'], 
                    [median_moisture, plant_moisture_range[0], plant_moisture_range[1]], 
                    color=['blue', 'green', 'green'])
        axes[2].set_title(f'Moisture Comparison - {plant_name.capitalize()}')
        axes[2].set_ylabel('Moisture')

        # Display the plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust to make space for the title
        plt.show()

    # Call the function to plot the comparison for the selected plant
    plot_comparison(median_temperature, median_light_level, median_moisture, 
                    plant_name, select_plant_moisture_range, select_plant_light_range, select_plant_temp_range)

    # Call the function to plot the comparison for the best plant
    plot_comparison(median_temperature, median_light_level, median_moisture, 
                    best_plant, best_plant_moisture_range, best_plant_light_range, best_plant_temp_range)
