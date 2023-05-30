# MasterThesis
Code for master thesis





Running with different models:

- DeepScenario based models:
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