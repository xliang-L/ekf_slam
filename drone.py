import random
import numpy as np


def create_distance_matrix(points):
    n = len(points)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[i][j] = np.linalg.norm(points[i] - points[j])
            distance_matrix[j][i] = distance_matrix[i][j]
    return distance_matrix


def create_population(size, num_cities):
    population = []
    for i in range(size):
        individual = list(range(num_cities))
        random.shuffle(individual)
        population.append(individual)
    return population


def fitness(individual, distance_matrix):
    fitness = 0
    for i in range(len(individual) - 1):
        fitness += distance_matrix[individual[i]][individual[i + 1]]
    fitness += distance_matrix[individual[-1]][individual[0]]
    return fitness


def selection(population, k=3):
    return max(random.choices(population, k=k), key=lambda x: fitness(x))


def crossover(x, y):
    n = len(x)
    c1 = random.randint(0, n - 1)
    c2 = random.randint(c1, n - 1)
    temp_x = x[c1:c2]
    temp_y = [item for item in y if item not in temp_x]
    return temp_x + temp_y


def mutation(x):
    n = len(x)
    c1 = random.randint(0, n - 1)
    c2 = random.randint(0, n - 1)
    x[c1], x[c2] = x[c2], x[c1]


def genetic_algorithm(points, size=100, generations=1000):
    num_cities = len(points)

    # Create initial population
    population = create_population(size, num_cities)

    # Create distance matrix
    distance_matrix = create_distance_matrix(points)

    # Evolution loop
    for i in range(generations):

        # Selection
        parent_1 = selection(population)
        parent_2 = selection(population)

        # Crossover
        child = crossover(parent_1, parent_2)

        # Mutation
        if random.random() < 0.01:
            mutation(child)

        # Add child to population
        population.append(child)

        # Remove worst individual
        population.remove(min(population, key=lambda x: fitness(x, distance_matrix)))

        # Print best individual
        if i % 100 == 0:
            print(
                f"Generation {i}: {min(population, key=lambda x: fitness(x, distance_matrix))} ({fitness(min(population, key=lambda x: fitness(x, distance_matrix)), distance_matrix)})")


points = np.random.rand(20, 2)  # Generate 20 random points
genetic_algorithm(points)  # Solve TSP problem using genetic algorithm