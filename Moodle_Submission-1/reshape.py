import numpy as np
import pandas as pd

data = np.arange(15).reshape((3, 5))

indexers = ['Temperature', 'Pressure', 'Humidity']

dframe1 = pd.DataFrame(
    data,
    index=indexers,
    columns=['CityA', 'CityB', 'CityC', 'CityD', 'CityE']
)

print(dframe1)
print("----------\n")

stacked = dframe1.stack()
print(stacked)
print("----------\n")

print(stacked.unstack())
print("----------\n")


series1 = pd.Series(
    [10, 20, 30, 40],
    index=['One', 'Two', 'Three', 'Four']
)

series2 = pd.Series(
    [50, 60, 70],
    index=['Five', 'Six', 'Seven']
)

frame2 = pd.concat([series1, series2], keys=['GroupA', 'GroupB'])

print(frame2)
print("----------\n")

print(frame2.unstack())
print("----------\n")