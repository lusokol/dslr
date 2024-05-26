from load_csv import load
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Gryffindor : red
# Ravenclaw : blue
# Hufflepuff : yellow
# Slytherin : green
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
def main():
    dataset = load("../datasets/dataset_train.csv")
    data = splitHouses(dataset.values)
    figure, axis = plt.subplots(4, 4, figsize=(30, 30)) 
    i = 0
    for idx, col in tqdm(enumerate(data[0].columns[6:]), total=13, desc=f'Loading histograms', position=0, leave=True):
        x = int(idx % 4)
        y = int(idx / 4)
        axis[y, x].hist(data[0][col], color="blue", alpha=0.3, edgecolor='black', label="Ravenclaw")
        axis[y, x].hist(data[1][col], color="red", alpha=0.3, edgecolor='black',label="Gryffindor")
        axis[y, x].hist(data[2][col], color="yellow", alpha=0.3, edgecolor='black',label="Hufflepuff")
        axis[y, x].hist(data[3][col], color="green", alpha=0.3, edgecolor='black',label="Slytherin")
        axis[y, x].set_title(col)
        i = idx + 1
    while i < 16:
        figure.delaxes(axis[int(i / 4), int(i % 4)])
        i += 1        
    plt.legend(loc='lower right', frameon=False, bbox_to_anchor=(4.5,0), prop={'size': 30})
    plt.savefig('../img/histograms.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()