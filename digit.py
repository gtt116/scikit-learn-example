import matplotlib.pyplot as plt
from sklearn import datasets


digits = datasets.load_digits()

fig = plt.figure(figsize=(6, 6))

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in xrange(64):

    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    ax.text(0, 7, str(digits.target[i]))
