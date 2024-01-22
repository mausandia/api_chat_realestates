import pandas as pd
import json

def calculate_statistics(df):
    # Group by 'property_type' and calculate statistics
    grouped_stats = df.groupby('property_type').agg({
        'last_price': ['mean', 'min', 'max', 'median', 'std'],
        'num_bedrooms': ['mean', 'min', 'max', 'median', 'std'],
        'num_bathrooms': ['mean', 'min', 'max', 'median', 'std'],
        'surface_total': ['mean', 'min', 'max', 'median', 'std']
    }).reset_index()

    # Rename columns for better readability
    grouped_stats.columns = ['property_type'] + [f'{col}_{stat}' for col, stat in grouped_stats.columns[1:]]

    return grouped_stats

def generate_intents(stats):
    # Create intents structure
    intents = []
    stats_name = {'mean':'average', 'min':'minimum', 'max':'maximum', 'median':'median', 'std':'standard deviation'}
    feature_name = ['last_price', 'num_bedrooms', 'num_bathrooms', 'surface_total']
    for index, row in stats.iterrows():
        for feature in feature_name:
            for key, value in stats_name.items():
                tag = f"{row['property_type'].lower()}_{feature}_{key}"  

                patterns = [f"What is the {value} {feature.replace('_',' ')} of {row['property_type']}?",
                            f"Tell me the {value} {row['property_type']} {feature.replace('_',' ')}."]
                responses = [
                    f"The {value} {feature.replace('_',' ')} of {row['property_type']} is {row[f'{feature}_{key}']:.1f}.",
                ]

                intent = {
                    'tag': tag,
                    'patterns': patterns,
                    'responses': responses
                }

                intents.append(intent)

    return intents

def write_to_json(intents, json_file):
    # Write intents to JSON file
    with open(json_file, 'w') as jsonfile:
        json.dump({'intents': intents}, jsonfile, indent=2)

if __name__ == "__main__":
    # Load CSV file
    file_path = "listings.csv"
    df = pd.read_csv(file_path)

    # Calculate statistics
    grouped_statistics = calculate_statistics(df)

    # Generate intents
    intents_list = generate_intents(grouped_statistics)

    # Write to JSON file
    json_file_path = "intents.json"
    write_to_json(intents_list, json_file_path)

    print(f"Intents written to {json_file_path}.")