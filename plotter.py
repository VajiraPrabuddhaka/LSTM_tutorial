from plotly.utils import pandas
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

dataset = pandas.read_csv('fff.csv', usecols=[3], engine='python', skipfooter=3)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

print (dataset)

plt.plot(dataset)
plt.show()