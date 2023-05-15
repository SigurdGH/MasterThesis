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



if __name__ == "__main__":
    p = Predicter()
    p.loadModel("xgb_gen_80-15-37-65")

    ga = GeneticAlgorithm(startingPopulation=1000,
                          childredPopulation=1000,
                          maxGenerations=1000,
                          maxValue=30,
                          minValue=0,
                          numberOfVariables=28,
                          sd=0.2,
                          mutationChance=0.4)
    
    solutions = ga.run()
    # print(bestSolutions[0])
    
    solDict = {1:0, 0:0}
    so = []
    for sols in solutions:
        preProcessed = p.preProcess(list(solutions[0][1]))[0]
        s = p.predict(preProcessed)
        solDict[int(s)] += 1
        so.append(s[0])
    
    # print(so[:10])
    # print(so[-10:])
    print()
    print(solDict)
    
    printSols(solutions[:10])
    printSols(solutions[-10:])
    
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

