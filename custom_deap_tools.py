from deap import base, creator, tools, algorithms

class my_HallOfFame(tools.HallOfFame):
    # inherit from the DEAP HallOfFame class BUT overwrite(and fix) their broken update function, doesn't work reliably
    def update(self, population):
        """Update the hall of fame with the *population* by replacing the
        worst individuals in it by the best individuals present in
        *population* (if they are better). The size of the hall of fame is
        kept constant.

        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        for ind in population:
            if len(self) == 0 and self.maxsize !=0:
                # Working on an empty hall of fame is problematic for the
                # "for else"
                self.insert(population[0])
                continue
            if ind.fitness > self[-1].fitness or len(self) < self.maxsize:
                # The individual is unique and strictly better than
                # the worst
                if len(self) >= self.maxsize:
                    self.remove(-1)
                self.insert(ind)
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