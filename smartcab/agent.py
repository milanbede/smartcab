from __future__ import print_function, division
import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.actions = [None, 'forward', 'left', 'right']
        self.learning_rate = 0.3
        self.state = None
        self.q = {}
        self.trips = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # Prepare for a new trip; reset any variables here, if required
        self.trips += 1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        state = (inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'],
self.next_waypoint)
        self.state = state

        # Lazy initialization of Q-values
        for action in self.actions:
            if (state, action) not in self.q:
                self.q[(state, action)] = 0.1
        
        # Select action according to the policy

        probabilities = [self.q[(state, None)], self.q[(state, 'forward')], self.q[(state, 'left')], self.q[(state, 'right')]]
        # Use the softmax funtion so that the values actually behave like probabilities
        probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities), axis=0)
        
        adventurousness = max((100 - self.trips) / 100, 0)
        if random.random() < adventurousness:
            action = np.random.choice(self.actions, p=probabilities)
        else:
            action = self.actions[np.argmax(probabilities)]

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        self.q[(state, action)] = self.learning_rate * reward + (1 - self.learning_rate) * self.q[(state, action)]

        #print("LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward))  # [debug]

    def get_state(self):
        return self.state

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=200)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
