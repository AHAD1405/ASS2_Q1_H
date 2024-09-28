import random
import numpy as np
import pandas as pd

class EP_Individual:
    def __init__(self, dimensions, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dimensions)
        self.strategy = np.random.uniform(0.1, 0.5, dimensions)  # mutation strengths
        self.fitness = None

def objective_fun1(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def objective_fun2(x):
    # Implement Griewanks function
    sum = 0
    prod = 1
    for i in range(len(x)):
        sum += x[i]**2/4000
        prod *= np.cos(x[i]/np.sqrt(i+1))
    return sum - prod + 1

def tournament_selection(population, fitness, tournament_size):
    """
    Perform tournament selection to choose a parent from the population.

    Parameters:
    population (list of numpy.ndarray): The population of individuals.
    fitness (list of float): The fitness values of the individuals in the population.
    tournament_size (int): The number of individuals to be selected for the tournament.

    Returns:
    numpy.ndarray: The selected parent individual.
    """
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    selected_fitness = [fitness[i] for i in selected_indices]
    best_index = selected_indices[np.argmin(selected_fitness)]
    return population[best_index], best_index

def EP(parameters):
    # Unpack parameters
    generations, dim, bounds, mu, lambda_, seed, obj_no = parameters

    # Set random seed
    random.seed(seed)

    # INITIALIZATION: Initialize population
    EP_population = [EP_Individual(dim, bounds) for _ in range(mu)]
    #variance = np.random.uniform(low=0, high=1, size=(mu, dim))

    # EVALUATION: Evaluate population fitness
    for individual in range(EP_population):
        individual.fitness = objective_fun1(individual.position) if obj_no == 1 else objective_fun2(individual.position)
        #fitness[i] = objective_fun1(population[i]) if obj_no == 1 else objective_fun2(population[i])
    
    # Create offspring through mutation
    offspring = []
    for parent in population:
        # Cauchy mutation
        child = EP_Individual(dim, bounds)
        child.position = parent.position.copy()
        child.strategy = parent.strategy.copy()
        #cauchy_mutate(cauchy_child, scale_factor)
        child.fitness = objective_fun1(child.position) if obj_no == 1 else objective_fun2(child.position)
        offspring.append(child)

        # Gaussian mutation
        #gaussian_child = EP_Individual(dim, bounds)
        #gaussian_child.position = parent.position.copy()
        #gaussian_child.strategy = parent.strategy.copy()
        #gaussian_mutate(gaussian_child)
        #gaussian_child.fitness = fitness_function(gaussian_child.position)
        #offspring.append(gaussian_child)

    best_individual = []
    best_fitness = []

    # Evolution loop
    for generation in range(generations):
        # Create offspring
        offspring = np.zeros((lambda_, dim))
        offspring_fitness = np.zeros(lambda_)

        for i in range(lambda_):
            # Select parents
            parent, _ = tournament_selection(population, fitness, 2)

            # Mutation
            offspring[i] = parent + np.random.normal(0, 1, dim)

            # Evaluate offspring
            offspring_fitness[i] = objective_fun1(offspring[i]) if obj_no == 1 else objective_fun2(offspring[i])

            # Update best individual and best fitness
            if not best_fitness or offspring_fitness[i] < best_fitness:
                best_individual = offspring[i]
                best_fitness = offspring_fitness[i]

        # Combine population and offspring
        population = np.vstack((population, offspring))
        fitness = np.concatenate((fitness, offspring_fitness))

        # Select mu best individuals
        best_indices = np.argsort(fitness)[:mu]
        population = population[best_indices]
        fitness = fitness[best_indices]
    
    return best_individual, best_fitness


def ES(parameters):
    # Unpack parameters
    generations, dim, bounds, mu, lambda_, seed, obj_no = parameters

    # Set random seed
    random.seed(seed)

    # Initialize population and variance
    population = np.random.uniform(low=bounds[0], high=bounds[1], size=(mu, dim))
    """ Modify the strategy of create variance """
    sigma = [[4.0] * dim for _ in range(mu)]  
    
    # Generate T and t_prime
    τ = (np.sqrt(2 * np.sqrt(dim))) ** -1
    τ_prime = (np.sqrt(2 * dim)) ** -1
    
    # EVALUATION: Evaluate population fitness
    fitness = np.zeros(mu) # Initialize fitness
    for i in range(mu):
        fitness[i] = objective_fun1(population[i]) if obj_no == 1 else objective_fun2(population[i])

    best_individual = []
    best_fitness = []

    # Evolution loop
    for generation in range(generations):
        print(f"Generation {generation+1} of {generations}...", end="\r")
        # Create offspring
        offspring = np.zeros((lambda_, dim))
        offspring_variance = np.zeros((lambda_, dim))
        offspring_fitness = np.zeros(lambda_)

        for i in range(lambda_):
            # Select parents
            parents1, parent1_id = tournament_selection(population, fitness, 2)
            parents2, parent2_id = tournament_selection(population, fitness, 2)

            # Recombination: use intermediate recombination
            combine = (parents1 + parents2) / 2
            combine_variance = [(a + b) / 2 for a,b in zip(sigma[parent1_id], sigma[parent2_id])]

            # Mutation variance of offspring
            offspring_variance[i] = combine_variance * np.exp(τ_prime * np.random.normal(0, 1, dim) + τ * np.random.normal(0, 1))

            # calculate diagonal matrix
            diag_matrix_temp = np.diag(offspring_variance[i])
            diag_matrix = np.diag(diag_matrix_temp)

            # Mutation of offspring
            offspring[i] = combine + np.random.normal(0, diag_matrix, dim)

            # Evaluate offspring
            offspring_fitness[i] = objective_fun1(offspring[i]) if obj_no == 1 else objective_fun2(offspring[i])
            
            # Update best individual and best fitness
            if not best_fitness or offspring_fitness[i] < best_fitness:
                best_individual = offspring[i]
                best_fitness = offspring_fitness[i]

        # Combine population and offspring
        population = np.vstack((population, offspring))
        sigma = np.vstack((sigma, offspring_variance))
        fitness = np.concatenate((fitness, offspring_fitness))

        # Select mu best individuals
        best_indices = np.argsort(fitness)[:mu]
        population = population[best_indices]
        fitness = fitness[best_indices]
        sigma = [sigma[i] for i in best_indices]

    return best_individual, best_fitness

        


        
def main():
    mu = 10  # Number of parents
    lambda_ = 10  # Number of offspring
    dimention_li = [20, 50]  # Number of dimensions
    generations = 100  # Number of generations
    times = 30  # Number of runs
    bounds = [-30, 30]  # Search space

    # generate 30 random seeds with determine incremental value
    seeds = [i+2 for i in range(times)]

    # Iterate over objective functions, once for each objective function
    for i in range(2):
        print()

        # for EP optmization algorithm     
        for run in range(times):
            print()

            for dim in dimention_li:
                print()
        
        # for ES optmization algorithm 
        for run in range(times):
            print(f"Run {run+1}/{times}...", end="\r")

            for dim in dimention_li:
                ES_parameters = [generations, dim, bounds, mu, lambda_, seeds[run], i+1] # (i) objective function number, 1 = objective_fun1, 2 = objective_fun2
                best_individual, best_fitness = ES(ES_parameters)
                print(f"Objective function {i+1}, Dimension {dim}, Run {run+1}/{times}: Best fitness: {best_fitness}")
                



if __name__ == "__main__":
    main()