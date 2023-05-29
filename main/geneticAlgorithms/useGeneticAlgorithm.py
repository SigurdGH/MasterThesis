import os
import sys
PATH = os.path.abspath(__file__)
PATH = PATH[:PATH.find("main")-1]
sys.path.insert(0, PATH)

from main.geneticAlgorithms.geneticAlgorithm import GeneticAlgorithm
from main.ML.Model import Predicter

def printSols(solutions):
    numberOfEach = 4
    colsToUse = ["TTCs", "DTOs", "JERKs", "Speeds"] #, "asX", "asY", "asZ"]
    print()
    for values in solutions:
        print(f"Score: {round(values[0], 3)}".ljust(17), end=" ")
        for i in range(0, 16, 4):
            if (i) % 4 == 0:
                s = colsToUse[(i) // 4]
            vals = [round(v, 3) for v in values[1][i:i+numberOfEach]]
            print(f"{s}: {vals}".ljust(40), end=" ")
        # print(f"\n\t\t  Angular: {[round(v, 3) for v in values[1][16:]]}", end="")
        print()


class GA(GeneticAlgorithm):
    def __init__(self, 
                 populationLimit=1000,
                 maxGenerations=10000,
                 maxValues=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.463,-0.463,-0.553,-0.553,-1.539,-1.539,-1.539,-1.539,-1.684,-1.684,-0.611,-0.611],
                 minValues=[50.0,50.0,50.0,50.0,100.0,100.0,100.0,100.0,89.18,89.18,89.18,56.688,33.352,34.721,33.352,34.721,1.158,1.158,0.49,0.49,1.407,1.468,1.468,1.468,1.079,1.079,0.884,0.884],
                 numberOfVariables=28,
                 sd=0.2,
                 mutationChance=0.4,
                 toKeep=4):
        super().__init__(populationLimit, maxGenerations, maxValues, minValues, numberOfVariables, sd, mutationChance, toKeep)

    def fitness(self, features):
        # return super().fitness(features)
        ttc = features[0:4]
        dto = features[4:8]
        speed = features[12:16]
        sum = 0
        for i, w in enumerate([0.1, 0.2, 0.3, 0.4]):
            sum += (ttc[i] * speed[i] - dto[i]/ttc[i]) * w
        f = sum/4
        return 99999 if f == 0 else 1/f
    

def approximateDataBefore():
    pass



if __name__ == "__main__":
    # p = Predicter()
    # p.loadModel("xgb_gen_80-15-37-65")
    # print(f"Number of features: {p.numberOfFeatures}")

    # test = [4, 4, 4, 50, 20,20,20,20, 0.576, 0.412, 24.86, 0.03919, 20, 20, 20, 40, 8.22, 0.42131214, 0.71372, 1.23866, 0.5605666, 0.06162596, 0.35024, 2.4556, 1.02989784, 2.425, 2.8417, 0.82873]
    # print(p.predict([test]))

    # ga = GA(populationLimit=1000,
    #                       maxGenerations=1000,
    #                       maxValue=[0,0,0,0,0,0,0,0],
    #                       minValue=[100,100,100,100,35,35,35,35], # DTO x 4, speed x 4
    #                       numberOfVariables=8,
    #                       sd=0.2,
    #                       mutationChance=0.4,
    #                       toKeep=4)
    
    ga = GeneticAlgorithm()

    solutions = ga.run()
    
    # printSols(solutions[:10])
    # printSols(solutions[-10:])

    # solDict = {1:0, 0:0}
    # so = []
    # for sols in solutions:
    #     preProcessed = p.preProcess(list(solutions[0][1]))[0]
    #     s = p.predict(preProcessed)
    #     solDict[int(s)] += 1
    #     so.append(s[0])
    
    # # print(so[:10])
    # # print(so[-10:])
    # print()
    # print(solDict)
    
    
    
    # p.loadModel("MLPClassifier_4843-1648-201-14")
    # ga = GeneticAlgorithm(startingPopulation=10000,
    #                       childredPopulation=1000,
    #                       maxGenerations=10000,
    #                       maxValue=20,
    #                       minValue=0,
    #                       numberOfVariables=8,
    #                       sd=0.1,
    #                       mutationChance=0.4)

    # solutions = ga.run()
    # # print(bestSolutions[0])
    
    # solDict = {1:0, 0:0}
    # ttc = 10
    # so = []
    # for sols in solutions:
    #     preProcessed = p.preProcess([ttc] + list(solutions[0][1]))[0] # Adding ttc because it needs one more parameter on this specific model
    #     s = p.predict(preProcessed)
    #     solDict[int(s)] += 1
    #     so.append(s[0])
    
    # # print(so[:10])
    # # print(so[-10:])
    # print(solDict)
    # for sols in solutions[:10]:
    #     print(sols)
    # for sols in solutions[-10:]:
    #     print(sols)

