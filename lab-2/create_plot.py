import matplotlib.pyplot as plt
import pandas

plt.figure(figsize=(8,6))

data = pandas.read_csv("output.csv")
print(data)
length = data.shape[0]

t = data[0:0 + length]["Experiment"]
scalar = data[0:0 + length]["Scalar"]
vector = data[0:0 + length]["Vector"]
plt.plot(t, [float(i) for i in scalar], label="Время скаляр", c="blue")
plt.plot(t, [float(i) for i in vector], label="Время вектор", c="red")

plt.xlabel('Номер эксперимента')
plt.ylabel('Время выполнения, мс')
plt.xticks(t)
# plt.yticks([500 * i for i in range(0,7)])
plt.legend()
plt.savefig('output.png')
plt.show()