import matplotlib.pyplot as plt
import pandas

plt.figure(figsize=(16,6))

data = pandas.read_csv("output.csv")
print(data)
length = data.shape[0]

t = data[0:0 + length]["T"]
duration = data[0:0 + length]["Duration"]
plt.plot(t, [float(i) for i in duration], label="Время выполнения", c="blue")

plt.xlabel('Количество потоков')
plt.ylabel('Время выполнения, мс')
plt.xticks(t)
# plt.yticks([1000 * i for i in range(0,7)])
plt.legend()
plt.savefig('output.png')
plt.show()