# Master's Thesis Repository
This repository contains the files needed to create and use the digital twins for our master's thesis.

## Instructions
To be able to run the different files, the requirements in "requirements.txt" needs to be satisfied.

If a file that is running the simualor is used, the instructions found [here](https://www.svlsimulator.com/docs/python-api/python-api/#requirements) must be followed.

The simulator itself can be downloaded from OSSDC [here](https://github.com/OSSDC/OSSDC-SIM/releases/tag/v1.2).

## Main files
* [digital-twin.generated.ipynb](https://github.com/SigurdGH/MasterThesis/blob/main/main/digital-twin.generated.ipynb) and [digital-twin.ipynb](https://github.com/SigurdGH/MasterThesis/blob/main/main/digital-twin.ipynb)
    * For creating the digital twin classifiers
* [UsingSim.py](https://github.com/SigurdGH/MasterThesis/blob/main/main/usingSim.py)
    * For running the simulator together with the digital twin
* [evaluatingGA.ipynb](https://github.com/SigurdGH/MasterThesis/blob/main/main/evaluatingGA.ipynb)
    * For evaluating the genetic algorithm
* [evaluatingGA.ipynb](https://github.com/SigurdGH/MasterThesis/blob/main/main/evaluatingGA.ipynb)
    * For evaluating the genetic algorithm
* [generatingDataWithSim.py](https://github.com/SigurdGH/MasterThesis/blob/main/main/generateTestingData/generatingDataWithSim.py)
    * For creating more data that the models can use for training
* [useGeneratedData.py](https://github.com/SigurdGH/MasterThesis/blob/main/main/generateTestingData/useGeneratedData.py)
    * For using the created data and creating models with self-generated data

## Results from different models:

- DeepScenario based models:
    - MLP: MLPClassifier_deep_577-16-29-188 =>
        - predicted collision just abit to late and it crashed.

    - Random Forest: RandomForestClassifier_deep_583-10-28-189 =>
        - This model never crashed but was overly cautious and braked down too early. (50 + meters before collision)
        - Stored image in raport folder.
    
    - SVM: SVC_deep_582-11-70-147 =>
        - This one was very very good, breaked down in the right moment and never crashed.
        - Stored image in raport folder.
    - XGBoost: XGBClassifier_deep_582-11-16-201 =>
        - This predicted the first collision to late and crashed.
        - Stored image in raport folder.

- Self Generated data based models:
    - MLP:
        - Model: MLPClassifier_gen_24-14-14-39 => 
                 - predicted first collision quite well, breaked down and started driving again
                 - crashed after that because of high speed, but was able to predict collision might happen before it happened.
                 - Crash Log:
                    - Time: 21.0 s          TTC: 1.24 s           DTO: 14.15 m          JERK: 0.59 m/s^3      Speed: 19.674 m/s
                    Time: 21.5 s          TTC: 0.69 s           DTO: 8.22 m           JERK: 0.55 m/s^3      Speed: 20.178 m/s
                    A COLLISION IS GOING TO HAPPEN!
                    Time: 22.0 s          TTC: 0.25 s           DTO: 2.71 m           JERK: 2.04 m/s^3      Speed: 18.097 m/s
                    47b529db-0593-4908-b3e7-4b24a32a0f70 collided with Sedan at Vector(-158.284042358398, 10.4187526702881, -127.416595458984), time: 22.29
                    Time: 22.5 s          TTC: 50 s             DTO: 1.29 m           JERK: 12.39 m/s^3     Speed: 0.466 m/s
    - Random Forest:
        - Model: RandomForestClassifier_gen_29-12-9-41 =>
                - This one was reaaallyyy good, never crashed and predicted braked down perfektly.
                - Stored image in raport folder.
    - SVM:
        - Model: SVC_gen_26-15-12-38 =>
                - This one was ok, it never crashed but i predictid a collision too early and the car braked down prematurely.
                - Stored image in raport folder.
    - XGBoost:
        - Model: XGBClassifier_gen_27-11-10-43 =>
                - This one was as good as the random forest one, never crashed and predicted braked down perfektly.
                - stored image in raport folder.
