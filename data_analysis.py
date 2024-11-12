import pandas as pd


# Load the CSV data
data = pd.read_csv('cleaned_wildfire_data.csv')

# Define fire size categories
def categorize_fire(size):
    if size > 100:
        return 'Large'
    elif size > 10:
        return 'Medium'
    else:
        return 'Small'

def main():
    # Apply the categorization
    data['Size Category'] = data['Acres'].apply(categorize_fire)

    # Calculate average fire size
    average_fire_size = data['Acres'].mean()
    print(f'Average Fire Size: {average_fire_size:.2f} acres')

    print()
    # Count the number of large, medium, and small fires
    fire_size_counts = data['Size Category'].value_counts()
    print('Fire Size Counts:')
    for category, count in fire_size_counts.items():
        print(f'{category}: {count}')

    print()
    # Analyze causes of fires
    cause_mapping = {
        1: 'Natural',      # Lightning
        2: 'Human',        # Equipment Use
        3: 'Human',        # Smoking
        4: 'Human',        # Campfire
        5: 'Natural',        # Debris
        6: 'Human',        # Railroad
        7: 'Human',        # Arson
        8: 'Human',        # Playing with fire
        9: 'Human',        # Miscellaneous
        10: 'Human',       # Vehicle
        11: 'Human',       # Powerline
        12: 'Human',       # Firefighter Training
        13: 'Human',       # Non-Firefighter Training
        14: 'Unknown',     # Unknown/Unidentified
        15: 'Human',       # Structure
        16: 'Human',       # Aircraft
        17: 'Natural',     # Volcanic
        18: 'Human'      # Escaped Prescribed Burn
    }

    # Map the causes to human or natural
    data['Cause Type'] = data['Cause'].map(cause_mapping)

    # Count the occurrences of each cause type
    cause_type_counts = data['Cause Type'].value_counts()
    print('Cause Type Counts:')
    for cause_type, count in cause_type_counts.items():
        print(f'{cause_type}: {count}')

main()