from flask import Flask
from Individual import Individual
app = Flask(__name__)

individual = Individual(1, 5)

@app.route('/')
def hello_world():  # put application's code here

    return str(individual.predict([1]))


if __name__ == '__main__':
    app.run()
