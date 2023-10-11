import math
import random
from itertools import combinations
from functools import reduce

POPULATION_SIZE=500
SELECTION_SIZE=25
NUM_GENERATIONS=5000

class Solver():

    def __init__(self):
        self.cities = []
        self.num_cities = 0
        self.population = []
      
    def initial_population(self):
        #return [tuple(random.sample(range(self.num_cities), self.num_cities)) for x in range(POPULATION_SIZE)]
        # pick random point
        population = []
        for _ in range(POPULATION_SIZE):
            mask = [1]*self.num_cities
            member = []
            start_city_idx = random.randrange(self.num_cities)
            member.append(start_city_idx)
            mask[start_city_idx] = 0

            for _ in range(self.num_cities-1):
                last_city_idx = member[-1]
                closest_city_idx = 0
                closest_distance = float('inf')
                for city_idx in range(self.num_cities):
                    if mask[city_idx]:
                        d = self.distance(last_city_idx, city_idx)
                        if d < closest_distance:
                            closest_city_idx=city_idx
                            closest_distance=d
                member.append(closest_city_idx)
                mask[closest_city_idx] = 0

            population.append(tuple(member))
    
        return population

    def get_input(self):
        with open('input.txt', 'r') as file:
            lines = file.readlines()
        self.num_cities = int(lines[0])
        global POPULATION_SIZE, SELECTION_SIZE, NUM_GENERATIONS
        if self.num_cities == 50:
            POPULATION_SIZE=100
            SELECTION_SIZE=15
            NUM_GENERATIONS=100000
        elif self.num_cities == 100:
            POPULATION_SIZE=150
            SELECTION_SIZE=15
            NUM_GENERATIONS=100000
        line_to_coord = lambda line: tuple([int(x) for x in (line.replace('\n', '')).split(' ')])
        for line in lines[1:]:
            self.cities.append(line_to_coord(line))
        self.population = self.initial_population()

    def distance(self, i, j):
        x1, y1, z1 = self.cities[i]
        x2, y2, z2 = self.cities[j] 
        d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return d
    
    def cost(self, tsp_soln):
        cost_value = 0
        for i in range(1, len(tsp_soln)):
            cost_value += self.distance(tsp_soln[i], tsp_soln[i-1])
        cost_value += self.distance(tsp_soln[0], tsp_soln[len(tsp_soln)-1])
        return cost_value
    
    def roulette_wheel(self, total_scores, generation):
        spin = random.randrange(int(total_scores))
        selections = [(total_scores/SELECTION_SIZE*i + spin) % total_scores for i in range(SELECTION_SIZE)]
        selections.sort()
        next_gen = []
        wheel_pointer = 0
        selection_idx = 0
        
        for member in generation:
            while wheel_pointer <= selections[selection_idx] and wheel_pointer + member[1] > selections[selection_idx]:
                next_gen.append(member)
                selection_idx += 1
                if selection_idx >= SELECTION_SIZE:
                    return next_gen
            wheel_pointer += member[1]
                
        return next_gen
    
    def cross(self, soln_a, soln_b, i, j):
        offspring = [-1] * len(soln_a)
        for e in range(i, j):
            offspring[e] = soln_a[e]
        for k in range(len(soln_b)):
            if soln_b[k] not in offspring and offspring[k] == -1:
                offspring[k] = soln_b[k]
        missing_cities = set(range(len(soln_a)))
        for c in offspring:
            missing_cities.discard(c)
        for k in range(len(offspring)):
            if offspring[k] == -1:
                random_element = random.choice(list(missing_cities))
                missing_cities.remove(random_element)
                offspring[k] = random_element
        return tuple(offspring)

    def test_population(self):
        time_steps = []
        rewards = []
        for i in range(NUM_GENERATIONS):
            soln_scores = []
            max_cost = 0
            total_scores = 0
            total_cost_generation = 0
            for member in self.population:
                member_cost = self.cost(member)
                total_cost_generation += member_cost
                soln_scores.append([member, member_cost])
                max_cost = max(member_cost, max_cost)

            for member in soln_scores:
                member[1] = max_cost - member[1]
                total_scores = total_scores + member[1]

            #TODO better check for convergence
            if abs(total_scores) < 10e-5:
                break

            total_cost_generation = total_cost_generation / len(soln_scores)
            time_steps.append(i)
            rewards.append(total_cost_generation)
            
            selected = self.roulette_wheel(total_scores, soln_scores)
            pairs = list(combinations(selected, 2))

            self.population = []
            for pair in pairs:
                i = random.randrange(self.num_cities)
                j = random.randrange(self.num_cities)
                if i > j:
                    temp = i
                    i = j
                    j = temp
                self.population.append(self.cross(pair[0][0], pair[1][0], i, j))

        # TODO fix
        best_soln = [()]
        best_distance = float('inf')
        for soln in self.population:
            if best_distance > self.cost(soln):
                best_soln[0] = soln
                best_distance = self.cost(soln)

        best_soln_first_city=best_soln[0][0]
        with open('output.txt', 'w') as file:
            file.write(f'{best_distance:.3f}\n')
            for point in best_soln[0]:
                file.write(f'{self.cities[point][0]} {self.cities[point][1]} {self.cities[point][2]}\n')
            file.write(f'{self.cities[best_soln_first_city][0]} {self.cities[best_soln_first_city][1]} {self.cities[best_soln_first_city][2]}')

    

solver = Solver()
solver.get_input()
solver.test_population()
