import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class DataAnalyzer:

    def __init__(self):
        self.df = None

    def read_data(self):
        self.df = pd.read_csv("GameLog.txt")

    def print_data(self):
        print(self.df)

    def print_duration(self):
        seconds = self.df.elapsed_time.max()
        minutes = int(seconds / 60)
        rest_seconds = seconds % 60
        print(str(seconds) + " seconds (" + str(minutes) + " min " + str(round(rest_seconds, 2)) + " seconds)")

    def clean_data(self):
        """
        Remove the logs before S, so S is the starting point.
        Convert from timestamp to elapsed time since start.
        :return:
        """
        # convert the TimeStamp column to a datetime format
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], format='%H.%M.%S.%f')

        # Find S elapsed time
        s_elapsed = self.df.loc[self.df.EventType == "S", "Timestamp"].min()
        
        # Select subset of DataFrame where Timestamp is greater than or equal to s_elapsed
        self.df = self.df[self.df['Timestamp'] >= s_elapsed]

        # Elapsed time column
        self.df['elapsed_time'] = (self.df['Timestamp'] - self.df['Timestamp'].min()).dt.total_seconds()

    def print_corner(self):
        corner_points = self.df[self.df.Corner != 0]
        print("There was " + str(len(corner_points)) + "/" + str(
            len(self.df)) + " entries where the agent was in the corner")
        print(corner_points)

    def plot(self, columns=None):
        if columns is None:
            columns = ['BallsLeft', 'PlayerLives', 'EnemyLives', 'Corner']

        # set the TimeStamp column as the index
        self.df.set_index('elapsed_time', inplace=True)

        # create the figure and axes objects
        fig, ax = plt.subplots()

        # plot the BallsLeft, PlayerLives, EnemyLives, and Corner columns
        self.df[columns].plot(ax=ax)

        # set the title and labels for the plot
        if len(columns) == 1:
            ax.set_title(columns[0])
        else:
            ax.set_title('GameLog')
        ax.set_xlabel('Time (S)')
        ax.set_ylabel('Count')

        # show the plot
        plt.show()


if __name__ == "__main__":
    da = DataAnalyzer()
    da.read_data()
    da.clean_data()
    da.print_duration()
    # da.print_data()
    # da.print_corner()
    # da.plot(columns=['BallsLeft'])
    # da.plot(columns=['PlayerLives'])
    # da.plot(columns=['EnemyLives'])
    da.plot(columns=['Corner'])
