from flask import Flask
from Individual import Individual
import GA

app = Flask(__name__)

#individual = Individual(1, 5)

GA.algorithm()


@app.route('/')
def hello_world():  # put application's code here

    #return str(individual.predict([1]))
    GA.algorithm()

    pass

if __name__ == '__main__':
    app.run()

