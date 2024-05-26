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
    figure, axis = plt.subplots(13, 13, figsize=(40, 40)) 
    i = 0
    for y, col1 in tqdm(enumerate(data[0].columns[6:]), total=13, desc=f'Loading lines', position=0, leave=True):
        for x, col2 in enumerate(data[0].columns[6:]):
            if (x != y):
                axis[y, x].scatter(data[0][col1], data[0][col2], color="blue", alpha=0.5, label="Ravenclaw")
                axis[y, x].scatter(data[1][col1], data[1][col2], color="red", alpha=0.5, label="Gryffindor")
                axis[y, x].scatter(data[2][col1], data[2][col2], color="yellow", alpha=0.5, label="Hufflepuff")
                axis[y, x].scatter(data[3][col1], data[3][col2], color="green", alpha=0.5, label="Slytherin")

                axis[y, x].set_title(f"{col1} - {col2}", fontsize=5)
            
            else:
                axis[y, x].hist(data[0][col1], color="blue", alpha=0.5, label="Ravenclaw")
                axis[y, x].hist(data[1][col1], color="red", alpha=0.5, label="Gryffindor")
                axis[y, x].hist(data[2][col1], color="yellow", alpha=0.5, label="Hufflepuff")
                axis[y, x].hist(data[3][col1], color="green", alpha=0.5, label="Slytherin")

                axis[y, x].set_title(f"{col1}", fontsize=5)

            axis[y, x].set_xticks([])
            axis[y, x].set_yticks([])

            if (y == 12):
                axis[y, x].set_xlabel(col2)
            if (x == 0):
                axis[y, x].set_ylabel(col1)
    plt.legend(loc='upper right', frameon=False, bbox_to_anchor=(1,0), prop={'size': 30})
    print("Saving plot into file ...")
    plt.savefig('../img/scatters.png', dpi=300, bbox_inches='tight')

    plt.figure(123)
    plt.scatter(data[0]["Astronomy"], data[0]["Defense Against the Dark Arts"], color="blue", alpha=0.5, label="Ravenclaw")
    plt.scatter(data[1]["Astronomy"], data[1]["Defense Against the Dark Arts"], color="red", alpha=0.5, label="Gryffindor")
    plt.scatter(data[2]["Astronomy"], data[2]["Defense Against the Dark Arts"], color="yellow", alpha=0.5, label="Hufflepuff")
    plt.scatter(data[3]["Astronomy"], data[3]["Defense Against the Dark Arts"], color="green", alpha=0.5, label="Slytherin")
    plt.savefig('../img/similar_feature.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()