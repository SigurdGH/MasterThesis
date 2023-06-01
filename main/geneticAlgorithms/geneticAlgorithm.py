import os
import sys
PATH = os.path.abspath(__file__)
PATH = PATH[:PATH.find("main")-1]
sys.path.insert(0, PATH)

import random
import math


class GeneticAlgorithm():
    """
    ## Genetic algorithm

    ### Params:
        * populationLimit: int, how many 
        * maxGenerations: int, 
        * maxValue: int, 
        * minValue: int, 
        * numberOfVariables: int, 
        * sd: float, 
        * mutationChance: float, 
        * toKeep: int, 
    """
    def __init__(self,
                 populationLimit=1000,
                 maxGenerations=10000,
                 maxValues=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.463,-0.463,-0.553,-0.553,-1.539,-1.539,-1.539,-1.539,-1.684,-1.684,-0.611,-0.611],
                 minValues=[50.0,50.0,50.0,50.0,100.0,100.0,100.0,100.0,89.18,89.18,89.18,56.688,33.352,34.721,33.352,34.721,1.158,1.158,0.49,0.49,1.407,1.468,1.468,1.468,1.079,1.079,0.884,0.884],
                 numberOfVariables=28,
                 selectionRate=3,
                 sd=0.2,
                 mutationChance=0.4,
                 fitnessGoal = 10000,
                 toKeep=4):
        self.populationLimit = populationLimit
        self.maxGenerations = maxGenerations
        self.numberOfVariables = numberOfVariables
        self.minValues = minValues
        self.maxValues = maxValues
        self.selectionRate = selectionRate
        self.sd = sd
        self.mutationChance = mutationChance
        self.fitnessGoal = fitnessGoal
        self.toKeep = toKeep
        self.population = [tuple(random.uniform(min, max) for min, max in zip(self.minValues, self.maxValues)) for _ in range(self.populationLimit)]


    # def fitness(self, features):
    #     # f = math.log(((dto[-1]/ttc[-1])+speed[-1])/17)
    #     ttc = features[0:4]
    #     dto = features[4:8]
    #     speed = features[12:16]
    #     sum = 0
    #     for i, w in enumerate([0.1, 0.2, 0.3, 0.4]):
    #         sum += (ttc[i] * speed[i] - dto[i]/ttc[i]) * w
    #         # sum += math.log(((dto[i]/ttc[i])+speed[i])/17) * w
    #     f = sum/4
    #     return 99999 if f == 0 else 1/f

    # def fitness(self, features):
    #     # f = math.log(((dto[-1]/ttc[-1])+speed[-1])/17)
    #     ttc = features[0:4]
    #     dto = features[4:8]
    #     speed = features[12:16]
    #     sum = 0
    #     for i, w in enumerate([0.1, 0.2, 0.3, 0.4]):
    #         sum += math.log(((dto[i]/ttc[i])+speed[i])/17) * w
    #     f = sum/4
    #     return 999999 if f == 0 else 1/f

    def fitness(self, features):
        ttc = features[0:4]
        dto = features[4:8]
        speed = features[12:16]
        sum = 0
        for i, w in enumerate([0.1, 0.15, 0.25, 0.5]):
            sum += abs(ttc[i] * speed[i] - dto[i] + ttc[i]**2/dto[i] - speed[i]/2) * w
            # sum += abs(ttc[i] * speed[i] - dto[i] + speed[i]**2/(dto[i]+1)**2 - (ttc[i]+1)**2) * w
            # sum += abs((dto[i]/speed[i]) - ttc[i]) * w
            # sum += math.log((dto[i]/ttc[i])+speed[i] + 1)
        # f = sum/4
        return 999999 if sum == 0 else 1/sum

    # def fitness(self, features):
    #     ttc = features[0:4]
    #     dto = features[4:8]
    #     speed = features[12:16]
    #     if speed[-1] > 17 and dto[-1] < 5 and ttc[-1] < 2:
    #         return speed[-1] * dto[-1] * ttc[-1]
    #     return 0

    # def fitness(self, features):
    #     # Proposed fitness function
    #     # f = math.log(((dto[-1]/ttc[-1])+speed[-1])/17)
    #     # ttc = features[0:4]
    #     # dto = features[4:8]
    #     # speed = features[12:16]
    #     # return math.log(((dto[-1]/ttc[-1])+speed[-1])/17)
    #     return math.log(((features[3]/features[7])+features[15])/17)


    def run(self):
        """
        Runs the genetic algorithm.\\
        Returns if a certain fitness value is reached or if the maximum number of generations has been evolved.
        """
        newGen = None
        rankedPopulation = None
        for i in range(self.maxGenerations):
            rankedPopulation = []
            for individual in self.population:
                rankedPopulation.append((self.fitness(individual), individual)) # fitness value, parameters
            rankedPopulation.sort(reverse=True)

            if rankedPopulation[0][0] > self.fitnessGoal: # exit if the fitness is good enough
                print(f"\n=== Fit enough in gen {i} ===", end="\n\t")
                self._printIndividual(rankedPopulation[0])
                return rankedPopulation

            if i % 100 == 0:
                print(f"=== Gen {i} best individual ===", end="\n\t")
                self._printIndividual(rankedPopulation[0])

            # Selection
            newGen = [individual[1] for individual in rankedPopulation[:self.toKeep]] # Keep n best from previous generation
            for _ in range((self.populationLimit - self.toKeep)//2):
                # parent1, parent2 = random.sample(rankedPopulation[:100], 2)
                parent1, parent2 = self._selection(rankedPopulation)
                
                children = self._crossover(rankedPopulation[parent1][1], rankedPopulation[parent2][1])
                for child in children:
                    if random.random() < self.mutationChance:
                        newGen.append(self._mutation(child))
                    else:
                        newGen.append(child)
            self.population = newGen
        else:
            # Only prints this if the results are not fit enough after reaching maxGenerations
            print(f"\n=== Best individual after {self.maxGenerations} generations ===", end="\n\t")
            self._printIndividual(rankedPopulation[0])
            return rankedPopulation

    
    def _selection(self, rankedPopulation):
        def chooseParent():
            return min([random.randint(0, self.populationLimit-1) for _ in range(self.selectionRate)])
        def diversity(parent1, parent2):
            nZeros = 0
            for g1, g2 in zip(parent1, parent2):
                if abs(g1-g2) < 0.01:
                    nZeros += 1
            return nZeros
        p1rank = chooseParent()
        p2rank = chooseParent()
        # s = diversity(rankedPopulation[p1rank], rankedPopulation[p1rank])
        # print(rankedPopulation[p1rank][1])
        return (p1rank, p2rank) if diversity(rankedPopulation[p1rank][1], rankedPopulation[p2rank][1]) < 1 else self._selection(rankedPopulation)
        # return (p1rank, p2rank) if p1rank != p2rank else self._selection()


    def _crossover(self, parent1, parent2):
        """
        Mixes the genes from the two parents into two chidren. \\
        Here the children are opposites of each other. \\
        I.e. if child1 gets the first gene from parent2, child2 will get the first gene from parent1.
        """
        child1 = []
        child2 = []
        for g1, g2 in zip(parent1, parent2):
            if random.randint(0, 1):
                child1.append(g1)
                child2.append(g2)
            else:
                child1.append(g2)
                child2.append(g1)
        return tuple(child1), tuple(child2)


    def _mutation(self, child):
        """
        Mutates the gene of a child if random.randint returns 1. Does this for every gene in the chromosome.
        TODO
        More mutation if avg parent rank is low
        """
        # chromosome = list(child)
        mutated = []
        variation = 5
        for gene, minVal, maxVal in zip(child, self.minValues, self.maxValues):
            if random.randint(0, 100) < 25:
                maxG = gene+variation if gene+variation < maxVal else maxVal
                minG = gene-variation if gene-variation > minVal else minVal
                mutated.append(random.uniform(minG, maxG))
            else:
                mutated.append(gene)
        return tuple(mutated)
        # return tuple(gene * random.uniform(1-self.sd, 1+self.sd) if random.randint(0, 1) else gene for gene in child)
    
    
    def _printIndividual(self, individual):
        print(f"Fitness: {str(round(individual[0], 8)).ljust(15)} Values: {individual[1]}")

    
if __name__ == "__main__":
    # pass
    ga = GeneticAlgorithm(populationLimit=1000,
                          maxGenerations=1000,
                        #   maxValue=30,
                        #   minValue=0,
                        #   numberOfVariables=28,
                          sd=0.2,
                          mutationChance=0.4,
                          fitnessGoal=100,
                          toKeep=4)
    bestSolutions = ga.run()
    print(ga.population[:10])

    # print(bestSolution)

    numberOfEach = 4
    colsToUse = ["TTCs", "DTOs", "JERKs", "Speeds"] #, "asX", "asY", "asZ"]
    print("\n")
    for values in bestSolutions[:10]:
        print(f"Score: {round(values[0], 3)}".ljust(17), end=" ")
        for i in range(0, 16, 4):
            if (i) % 4 == 0:
                s = colsToUse[(i) // 4]
            vals = [round(v, 3) for v in values[1][i:i+numberOfEach]]
            print(f"{s}: {vals}".ljust(40), end=" ")
        # print(f"\n\t\t  Angular: {[round(v, 3) for v in values[1][16:]]}", end="")
        print()

