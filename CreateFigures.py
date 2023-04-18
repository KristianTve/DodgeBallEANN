import pickle
import graphviz
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv('data/genome_fitnessANN.csv', sep=" ")
    df_smoothed = df.rolling(window=10).mean()

    print(df_smoothed.keys())
    label ="CTRNN"

    plt.plot(range(len(df_smoothed)), df_smoothed.average, 'b-', label="average")
    plt.plot(range(len(df_smoothed)), df_smoothed.best, 'r-', label="best")

    plt.title("[" + label + "] - Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")

    plt.savefig("data/figure.png")
    plt.show()

    plt.close()


    print(df)


