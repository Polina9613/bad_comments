# bad_comments
Проект по автоматическому определению токсичности в текстах на русском языке с использованием классических ML-алгоритмов
Сравнение моделей по F1-score (основная метрика)

Final Model Comparison:
                    precision    recall        f1
LogReg_Optimized     0.877272  0.876518  0.876849
LogisticRegression   0.871970  0.871315  0.871609
NaiveBayes           0.862351  0.855359  0.846829
CatBoost             0.836187  0.827610  0.830192
RF_Optimized         0.820868  0.822754  0.821592
RandomForest         0.814244  0.816857  0.815110
XGBoost              0.812864  0.789802  0.795027

- Оптимизировала LogisticRegression и RandomForest через GridSearchCV, улучшив F1-score на 0.6-0.7%  
- Выявила ключевые параметры:  
  - Для LogisticRegression: C=1, только униграммы (F1=0.877)  
  - Для RandomForest: 200 деревьев с биграммами (F1=0.822) 
