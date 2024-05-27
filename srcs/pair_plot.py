import matplotlib.pyplot as plt
from utils import splitHouses, load
from tqdm import tqdm

# Gryffindor : red
# Ravenclaw : blue
# Hufflepuff : yellow
# Slytherin : green

def main():
    dataset = load("../datasets/dataset_train.csv")
    data = splitHouses(dataset.values)
    figure, axis = plt.subplots(13, 13, figsize=(50, 50)) 
    i = 0
    pbar = tqdm(total=169, desc='Loading plots')
    for y, col1 in enumerate(data[0].columns[6:]):
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

            # axis[y, x].set_xticks([])
            # axis[y, x].set_yticks([])

            if (y == 12):
                axis[y, x].set_xlabel(col2)
            if (x == 0):
                axis[y, x].set_ylabel(col1)
            pbar.update(1)
    pbar.close()
    plt.legend(loc='upper right', frameon=False, bbox_to_anchor=(1,0), prop={'size': 30})
    print("Saving plot into file ...")
    plt.savefig('../img/pair_plot.png', dpi=200)

if __name__ == "__main__":
    main()