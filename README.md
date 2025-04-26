# bad_comments
Проект по автоматическому определению токсичности в текстах на русском языке с использованием классических ML-алгоритмов
Сравнение моделей по F1-score (основная метрика)

                    f1      Улучшение vs базовая
LogReg_Optimized    0.88    +0.006
LogisticRegression  0.87    -
CatBoost            0.83    -
NaiveBayes          0.85    -
RF_Optimized        0.82    +0.007
RandomForest        0.81    -
XGBoost             0.80    -

- Оптимизировала LogisticRegression и RandomForest через GridSearchCV, улучшив F1-score на 0.6-0.7%  
- Выявила ключевые параметры:  
  - Для LogisticRegression: C=1, только униграммы (F1=0.877)  
  - Для RandomForest: 200 деревьев с биграммами (F1=0.822) 
