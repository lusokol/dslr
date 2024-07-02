from srcs.utils import splitHouses, load
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="input dataset")
    args = parser.parse_args()

    dataset = load(args.dataset)
    try:
        data = splitHouses(dataset.values)
        plt.scatter(data[0]["Astronomy"], data[0]["Defense Against the Dark Arts"], color="blue", alpha=0.5, label="Ravenclaw")
        plt.scatter(data[1]["Astronomy"], data[1]["Defense Against the Dark Arts"], color="red", alpha=0.5, label="Gryffindor")
        plt.scatter(data[2]["Astronomy"], data[2]["Defense Against the Dark Arts"], color="yellow", alpha=0.5, label="Hufflepuff")
        plt.scatter(data[3]["Astronomy"], data[3]["Defense Against the Dark Arts"], color="green", alpha=0.5, label="Slytherin")
    except KeyError as e:
        print(f"KeyError: {e}")
        exit()
    except IndexError as e:
        print(f"IndexError: {e}")
        exit()
    except AssertionError as e:
        print(f"AssertionError: {e}")
        exit()
    
    plt.title("Astronomy - Defense Against the Dark Arts")
    plt.legend()
    plt.savefig('img/scatter_plot.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()