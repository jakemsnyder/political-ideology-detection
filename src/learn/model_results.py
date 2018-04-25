import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.DataFrame({'model': ['Baseline',
                               'TFIDF Ridge',
                               'TFIDF Lasso',
                               'LSTM all',
                               'LSTM modern',
                               'LSTM taxes'],
                     'MSE': [.15801,
                             .15135,
                             .15656,
                             .14867,
                             .18055,
                             .16660]})

fig, ax = plt.subplots(figsize=(15,9))
plt.rcParams.update({'font.size': 12})

scores = sns.barplot(y='model', x='MSE', data=data, color='#164F86')

plt.xlim([0, 0.225])
#plt.show()

figure = scores.get_figure()
figure.savefig("../../results/model_scores.png")