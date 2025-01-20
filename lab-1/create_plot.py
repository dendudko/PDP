import matplotlib.pyplot as plt
import pandas

plt.figure(figsize=(8,6))

data = pandas.read_csv("output.csv")
print(data)
length = data.shape[0]

t = data[0:0 + length]["Threads"]
duration = data[0:0 + length]["Average Time (s)"]
plt.plot(t, [float(i) for i in duration], label="Время выполнения", c="blue")

plt.xlabel('Количество потоков')
plt.ylabel('Время выполнения, с')
plt.xticks(t)
# plt.yticks([0.1 * i for i in range(0,21)])
plt.legend()
plt.savefig('output.png')
plt.show()