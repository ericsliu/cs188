# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        counter = 0
        for state in self.mdp.getStates():
            self.values[state] = 0
        while counter < iterations:
            newValues = util.Counter()
            for state in self.mdp.getStates():
                if not mdp.isTerminal(state):
                    actions = self.mdp.getPossibleActions(state)
                    maxValue = float("-inf")
                    for action in actions:
                        value = 0
                        stateProbs = self.mdp.getTransitionStatesAndProbs(state, action)
                        for pair in stateProbs:
                            discounted = 0
                            if pair[0] == 'TERMINAL_STATE':
                                discounted = 0 #redundant???
                            else:
                                discounted = discount * self.getValue(pair[0])
                            value += (self.mdp.getReward(state, action, pair[0]) + discounted) * pair[1]
                        if value > maxValue:
                            maxValue = value
                    newValues[state] = maxValue
            for state in newValues:
                self.values[state] = newValues[state]
            counter += 1
        print self.values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        statesProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        value = 0
        for pair in statesProbs:
            discounted = 0
            if pair[0] == 'TERMINAL_STATE':
                discounted = 0
            else:
                discounted = self.discount * self.getValue(pair[0])
            value += (self.mdp.getReward(state, action, pair[0]) + discounted) * pair[1]
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)
        maxVal = float("-inf")
        bestAction = None
        if actions == None:
            return None
        else:
            for action in actions:
                qVal = self.computeQValueFromValues(state, action)
                if qVal > maxVal:
                    maxVal = qVal
                    bestAction = action
            return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
