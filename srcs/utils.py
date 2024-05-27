import pandas as pd
import os

def splitHouses(dataset):
    houses = {'Ravenclaw': [], 'Gryffindor': [], 'Hufflepuff': [], 'Slytherin': []}
    for student in dataset:
        if student[1] in houses:
            houses[student[1]].append(student)
        else:
            raise AssertionError("Someone isn't in one of the four houses!")

    # Convert each list to a DataFrame and add a column for the house
    columns = ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']

    df_ravenclaw = pd.DataFrame(houses['Ravenclaw'], columns=columns)
    df_gryffindor = pd.DataFrame(houses['Gryffindor'], columns=columns)
    df_hufflepuff = pd.DataFrame(houses['Hufflepuff'], columns=columns)
    df_slytherin = pd.DataFrame(houses['Slytherin'], columns=columns)

    # Combine all DataFrames into one
    df_combined = [df_ravenclaw, df_gryffindor, df_hufflepuff, df_slytherin]
    
    return df_combined

def load(path: str):
    """
Load .csv file with pandas.
Parse error and return dataset.
    """
    try:
        if not os.path.exists(path):
            raise AssertionError("File doesn't exist.")
        if not path.lower().endswith('.csv'):
            raise AssertionError("File isn't .csv")
        data = pd.read_csv(path)
        return data
    except AssertionError as error:
        print(f"AssertionError: {error}")
        exit()