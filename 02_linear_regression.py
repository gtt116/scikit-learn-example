import matplotlib.pyplot as plt
from sklearn import linear_model


X = [[6], [8], [10], [14],   [18]]
y = [[7], [9], [13], [17.5], [18]]


model = linear_model.LinearRegression()
model.fit(X, y)
y1 = model.predict([12])[0]
print 'A 12 pizza should cost: $%.2f' % y1

X2 = [[0], [10], [14], [25]]
y2 = model.predict(X2)

plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X, y, 'k.')
plt.plot(X2, y2, 'g-')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()
