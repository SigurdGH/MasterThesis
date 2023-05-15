import os
import sys
PATH = os.path.abspath(__file__)
PATH = PATH[:PATH[:PATH[:PATH.rfind("\\")].rfind("\\")].rfind("\\")]
sys.path.insert(0, PATH)

import random
import math


class GeneticAlgorithm():
    def __init__(self,
                 startingPopulation=1000,
                 childredPopulation=1000,
                 maxGenerations=10000,
                 maxValue=20,
                 minValue=0,
                 numberOfVariables=8,
                 sd=0.1,
                 mutationChance=0.4,
                 toKeep=4):
        # self.startingPopulation = startingPopulation
        self.childredPopulation = childredPopulation-toKeep
        self.maxGenerations = maxGenerations
        self.numberOfVariables = numberOfVariables
        self.solutions = [
            tuple(random.uniform(minValue, maxValue) 
                  for _ in range(self.numberOfVariables)) 
                  for _ in range(startingPopulation)]
        self.sd = sd
        self.mutationChance = mutationChance
        self.toKeep = toKeep
    
    
    # def problem(self, features):
    #     """
    #     Sum is supposed to be as close to 0 as possible, but the real target is 1, as in a boolean true value.
    #     """
    #     promlemFeautures = [2.81, 0.14, 4.63, 7.34, 10.09, 12.9, 15.63, 17.58]
    #     sum = -1
    #     for f, pf in zip(features, promlemFeautures):
    #         sum += f * pf
    #     return sum


    # def fitness(self, features):
    #     # features = [DTO, JERK, s1, s2, s3, s4, s5, s6]
    #     # DTO = features[0]
    #     # speed6 = features[-1]
    #     if features[-1] > features[0]:
    #         f = math.log(features[-1]/15)+math.log(features[0])
    #     else:
    #         f = math.log(features[-1])+math.log(features[0])
    #     return 99999 if f == 0 else 1/f

    def fitness(self, features):
        # f = math.log(((DTO/TTC)+Speed)/17)
        ttc = features[0:4]
        dto = features[4:8]
        speed = features[12:16]
        sum = 0
        for i, w in enumerate([0.1, 0.2, 0.3, 0.4]):
            sum += math.log(((dto[i]/ttc[i])+speed[i])/17) * w
            # sum += math.log(((dto[i]/ttc[i])+speed[i])/17) * w
        f = sum/4
        return 99999 if f == 0 else f # 1/f


    def evaluate(self):
        newGen = None
        rankedSolutions = None
        for i in range(self.maxGenerations):
            rankedSolutions = []
            for s in self.solutions:
                rankedSolutions.append((self.fitness(s), s )) # fitness value, parameters
            rankedSolutions.sort(reverse=True)

            if i % 100 == 0:
                print(f"=== Gen {i} best solution ===", end="\n\t")
                print(f"Fitness: {str(round(rankedSolutions[0][0], 8)).ljust(15)} Values: {rankedSolutions[0][1]}")
                # print(f"=== Gen {i} best solution === Fitness: {str(round(rankedSolutions[0][0], 8)).ljust(15)} Values: {rankedSolutions[0][1]}", end="\r")

            if rankedSolutions[0][0] > 9999: # exit if the fitness is good enough
                print(f"\n=== Fit enough in gen {i} ===", end="\n\t")
                print(f"Fitness: {str(round(rankedSolutions[0][0], 8)).ljust(15)} Values: {rankedSolutions[0][1]}")
                f = [rankedSolutions[0][1][i] for i in range(self.numberOfVariables)]
                # res = self.problem(f) + 1
                # print(f"Test: {res}")
                return rankedSolutions

            newGen = [solution[1] for solution in rankedSolutions[:self.toKeep]] # Keep n best from previous generation

            for _ in range(self.childredPopulation//2): # The new generation consists of n + self.childredPopulation solutions
                parent1, parent2 = random.sample(rankedSolutions[:100], 2)
                children = self._crossover(parent1[1], parent2[1])
                for child in children:
                    if random.random() < self.mutationChance:
                        newGen.append(self._mutation(child))
                    else:
                        newGen.append(child)
            self.solutions = newGen
        else:
            # Only prints this if the results are not fit enough
            print(f"\n=== Best solution after {self.maxGenerations} generations ===", end="\n\t")
            print(rankedSolutions[0])
            f = [rankedSolutions[0][1][i] for i in range(self.numberOfVariables)]
            # res = self.problem(f) + 1
            # print(f"Test: {res}")
            return rankedSolutions


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
        # mutatedChild = list(child)
        # for gene in range(len(mutatedChild)):
        #     if random.randint(0, 1):
        #         mutatedChild[gene] *= random.uniform(1-self.sd, 1+self.sd)
        # return tuple(mutatedChild)
        return tuple(gene * random.uniform(1-self.sd, 1+self.sd) if random.randint(0, 1) else gene for gene in list(child))
    

    def run(self):
        return self.evaluate()
    

# if __name__ == "__main__":
    # ga = GeneticAlgorithm(startingPopulation=1000,
    #                       childredPopulation=1000,
    #                       maxGenerations=1000,
    #                       maxValue=28,
    #                       minValue=0,
    #                       numberOfVariables=28,
    #                       sd=0.1,
    #                       mutationChance=0.4,
    #                       toKeep=4)
    # bestSolutions = ga.run()
    # # bestSolution = ga.evaluate()
    # # print(bestSolution)

    # numberOfEach = 4
    # colsToUse = ["TTCs", "DTOs", "JERKs", "Speeds"] #, "asX", "asY", "asZ"]
    # print("\n")
    # for values in bestSolutions[:10]:
    #     print(f"Score: {round(values[0], 3)}".ljust(17), end=" ")
    #     for i in range(0, 16, 4):
    #         if (i) % 4 == 0:
    #             s = colsToUse[(i) // 4]
    #         vals = [round(v, 3) for v in values[1][i:i+numberOfEach]]
    #         print(f"{s}: {vals}".ljust(40), end=" ")
    #     print(f"\n\t\t  Angular: {[round(v, 3) for v in values[1][16:]]}", end="")
    #     print()

