from flask import Flask
from Individual import Individual
import GA
import requests
from mlagents_envs.environment import UnityEnvironment

app = Flask(__name__)
env = UnityEnvironment()
print("BEHAVIOR SPECS")
print(env.behavior_specs)
env.step()
print(env)
#GA.algorithm()

@app.route('/')
def hello_world():  # put application's code here
    #return str(individual.predict([1]))
    GA.algorithm()
    pass

if __name__ == '__main__':
    app.run()

from flask import Flask
app = Flask(__name__)

@app.route('/some-url')
def get_data():
    return requests.get('http://example.com').content