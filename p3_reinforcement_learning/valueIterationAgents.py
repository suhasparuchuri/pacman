# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            states = self.mdp.getStates()
            temp_vals = util.Counter()
            for state in states:
                maxVal = float("-inf")
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    q_val = self.computeQValueFromValues(state, action)
                    if(q_val > maxVal):
                        maxVal = q_val
                    temp_vals[state] = maxVal
            self.values = temp_vals



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
        "*** YOUR CODE HERE ***"
        tState = self.mdp.getTransitionStatesAndProbs(state, action)
        v = 0
        for nextState, prob  in tState:
            reward = self.mdp.getReward(state, action, nextState)
            v  += prob *(self.values[nextState] * self.discount + reward)
        return v

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        maxVal = float("-inf")
        bestAction = ""
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            q_val = self.computeQValueFromValues(state, action)
            if(q_val > maxVal):
                maxVal = q_val
                bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        statesLen = len(states)
        for i in range(self.iterations):
            state = states[i % statesLen]
            actions = self.mdp.getPossibleActions(state)

            if not self.mdp.isTerminal(state):
                temp_vals = []
                for action in actions:
                    q_val = self.computeQValueFromValues(state, action)
                    temp_vals.append(q_val)
                self.values[state] = max(temp_vals)



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pq = util.PriorityQueue()
        pred = {}
        states = self.mdp.getStates()
        for s in states:
            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                for a in actions:
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(s,a):
                        if nextState in pred:
                            pred[nextState].add(s)
                        else:
                            pred[nextState] = {s}

        for s in states:
            if not self.mdp.isTerminal(s):
                temp_vals = []
                actions = self.mdp.getPossibleActions(s)

                for a in actions:
                    q_val = self.computeQValueFromValues(s, a)
                    temp_vals.append(q_val)

                diff = abs(self.values[s] - max(temp_vals))
                pq.update(s, -diff)

        for iteration in range(self.iterations):
            if pq.isEmpty():
                break
            s = pq.pop()
            if not self.mdp.isTerminal(s):
                temp_vals = []
                actions = self.mdp.getPossibleActions(s)
                for a in actions:
                   q_val = self.computeQValueFromValues(s, a)
                   temp_vals.append(q_val)
                self.values[s] = max(temp_vals)

            for p in pred[s]:
                if not self.mdp.isTerminal(p):
                    temp_vals = []
                    actions = self.mdp.getPossibleActions(p)
                    for a in actions:
                        q_val = self.computeQValueFromValues(p, a)
                        temp_vals.append(q_val)
                    diff = abs(max(temp_vals) - self.values[p])

                    if diff > self.theta:
                        pq.update(p,-diff)
