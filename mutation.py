from deap import base, creator, tools, algorithms

def my_mutGaussian(individual, mu, sigma, indpb):
    individual = tools.mutGaussian(individual, mu, sigma, indpb)[0]
    # ensure no gene is out of range
    for i in range(len(individual)):
        gene = individual[i]
        # genes are between 0 and 1 but could be outside bounds due to mutation, fix this.
        if (gene > 1):
            bounded_gene = 1
        elif (gene < 0):
            bounded_gene = 0
        else:
            bounded_gene = gene
        individual[i] = bounded_gene
    return individual,