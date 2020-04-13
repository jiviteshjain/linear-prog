import numpy as np
import cvxpy as cp
import json
import os

HEALTH_RANGE = 5
ARROWS_RANGE = 4
STAMINA_RANGE = 3
NUM_STATES = HEALTH_RANGE * ARROWS_RANGE * STAMINA_RANGE

HEALTH_VALUES = tuple(range(HEALTH_RANGE))
ARROWS_VALUES = tuple(range(ARROWS_RANGE))
STAMINA_VALUES = tuple(range(STAMINA_RANGE))

HEALTH_FACTOR = 25 # 0, 25, 50, 75, 100
ARROWS_FACTOR = 1 # 0, 1, 2, 3
STAMINA_FACTOR = 50 # 0, 50, 100

NUM_ACTIONS = 4
ACT_NOOP = 0
ACT_SHOOT = 1
ACT_DODGE = 2
ACT_RECHARGE = 3
ACTION_NAMES = ['NOOP', 'SHOOT', 'DODGE', 'RECHARGE']

TEAM = 34
Y = [1/2, 1,2]
COST = -10/Y[TEAM%3]

class State:
    def __init__(self, health, arrows, stamina):
        if (health not in HEALTH_VALUES) or (arrows not in ARROWS_VALUES) or (stamina not in STAMINA_VALUES):
            raise ValueError
        
        self.health = health
        self.arrows = arrows
        self.stamina = stamina

    def as_tuple(self):
        return (self.health, self.arrows, self.stamina)

    def as_list(self):
        return [self.health, self.arrows, self.stamina]

    def get_hash(self):
        return (self.health * (ARROWS_RANGE * STAMINA_RANGE) +
                self.arrows * STAMINA_RANGE +
                self.stamina)

    def is_action_valid(self, action):
        if action == ACT_NOOP: # NOOP is valid only for terminal states
            return (self.health == 0)

        if self.health == 0: # for terminal states, only NOOP is valid
            return False
        
        # Now the state is not terminal

        if action == ACT_SHOOT:
            return (self.arrows != 0 and self.stamina != 0)

        if action == ACT_DODGE:
            return (self.stamina != 0)

        if action == ACT_RECHARGE:
            return (self.stamina != STAMINA_VALUES[-1])

    def actions(self):
        actions = []
        for i in range(NUM_ACTIONS):
            if self.is_action_valid(i):
                actions.append(i)
        return actions

    def do(self, action):
        # returns list of (probability, state)

        if action not in self.actions():
            raise ValueError

        # the action is valid

        if action == ACT_NOOP:
            return []

        if action == ACT_SHOOT:

            s_a = State(self.health - 1, self.arrows - 1, self.stamina - 1) # Because action is valid,
            s_b = State(self.health, self.arrows - 1, self.stamina - 1)     # these states exist

            return [
                (0.5, s_a),
                (0.5, s_b),
            ]

        if action == ACT_DODGE:

            if self.stamina == STAMINA_VALUES[1]:
                s_b = State(self.health, self.arrows, self.stamina - 1)
                if self.arrows < ARROWS_VALUES[-1]:
                    s_a = State(self.health, self.arrows + 1, self.stamina - 1)
                    return [
                        (0.8, s_a),
                        (0.2, s_b),
                    ]
                else:
                    return [
                        (1, s_b),
                    ]
            else:
                s_c = State(self.health, self.arrows, self.stamina - 1)
                s_d = State(self.health, self.arrows, self.stamina - 2)
                if self.arrows < ARROWS_VALUES[-1]:
                    s_a = State(self.health, self.arrows + 1, self.stamina - 1)
                    s_b = State(self.health, self.arrows + 1, self.stamina - 2)
                    return [
                        (0.64, s_a),
                        (0.16, s_b),
                        (0.16, s_c),
                        (0.04, s_d),
                    ]
                else:
                    return [
                        (0.8, s_c),
                        (0.2, s_d),
                    ]
        
        if action == ACT_RECHARGE:
            s_a = State(self.health, self.arrows, self.stamina + 1)
            s_b = State(*self.as_tuple())

            return [
                (0.8, s_a),
                (0.2, s_b),
            ]

    @classmethod
    def from_hash(self, num):
        if type(num) != int:
            raise ValueError

        if not (0 <= num < NUM_STATES):
            raise ValueError

        health = num // (ARROWS_RANGE * STAMINA_RANGE)
        num = num % (ARROWS_RANGE * STAMINA_RANGE)

        arrows = num // STAMINA_RANGE
        num = num % STAMINA_RANGE

        stamina = num

        return State(health, arrows, stamina)


class Lero:
    def __init__(self):
        self.dim = self.get_dimensions()
        self.r = self.get_r()
        self.a = self.get_a()
        self.alpha = self.get_alpha()
        self.x = self.quest()
        self.policy = []
        self.solution_dict = {}
        self.objective = 0.0
    
    def get_dimensions(self):
        dim = 0
        for i in range(NUM_STATES):
            dim = dim + len(State.from_hash(i).actions())
        return dim

    def get_a(self):
        a = np.zeros((NUM_STATES, self.dim), dtype=np.float64)

        idx = 0
        for i in range(NUM_STATES):
            s = State.from_hash(i)
            actions = s.actions()

            for action in actions:
                a[i][idx] += 1
                next_states = s.do(action)
                
                for next_state in next_states:
                    a[next_state[1].get_hash()][idx] -= next_state[0]

                # increment idx
                idx += 1

        return a

    def get_r(self):
        r = np.full((1, self.dim), COST)

        idx = 0
        for i in range(NUM_STATES):
            actions = State.from_hash(i).actions()

            for action in actions:
                if action == ACT_NOOP:
                    r[0][idx] = 0
                idx += 1
        
        return r

    def get_alpha(self):
        alpha = np.zeros((NUM_STATES, 1))
        s = State(HEALTH_VALUES[-1], ARROWS_VALUES[-1], STAMINA_VALUES[-1]).get_hash()
        alpha[s][0] = 1
        return alpha

    def quest(self):
        x = cp.Variable((self.dim, 1), 'x')
        
        constraints = [
            cp.matmul(self.a, x) == self.alpha,
            x >= 0
        ]

        objective = cp.Maximize(cp.matmul(self.r, x))
        problem = cp.Problem(objective, constraints)

        solution = problem.solve()
        self.objective = solution
        arr = list(x.value)
        l = [ float(val) for val in arr]
        return l

    def get_policy(self):
        idx = 0
        for i in range(NUM_STATES):
            s = State.from_hash(i)
            actions = s.actions()
            temp = np.NINF
            best_action = -1
            for action in actions:
                if(self.x[idx] >= temp):
                    temp = self.x[idx]
                    best_action = action 
                idx += 1
            local = []
            local.append(s.as_list())
            local.append(ACTION_NAMES[best_action])
            self.policy.append(local)

    def generate_dict(self):
        self.solution_dict["a"] = self.a.tolist()
        r = [float(val) for val in np.transpose(self.r)]
        self.solution_dict["r"] = r
        alp = [float(val) for val in self.alpha]
        self.solution_dict["alpha"] = alp
        self.solution_dict["x"] = self.x
        self.solution_dict["policy"] = self.policy
        self.solution_dict["objective"] = float(self.objective)
        
    def write_output(self):
        path = "outputs/output.json"
        json_object = json.dumps(self.solution_dict, indent=4)
        with open(path, 'w+') as f:
          f.write(json_object)

    def execute(self):
        os.makedirs('outputs', exist_ok=True)
        self.quest()
        self.get_policy()
        self.generate_dict()
        self.write_output()    


lero = Lero()
lero.execute()
