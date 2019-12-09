
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('scores.csv')
# print(df.shape)
scores = df['Scores']
print(scores)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()