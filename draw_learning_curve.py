from matplotlib import pyplot as plt

fname = '/home/seungkwan/Log_resnet34_cifar100_s1_t13.0_d0.04_seed2.txt'
f = open(fname, 'r')
lines = f.readlines()
X = [i for i in range(1, 301)]
train_acc = []
test_acc = []
for i in range(len(lines)):
    line = lines[i]
    if line[0] != '=':
        continue
    a = lines[i + 1].split()[-1]
    a = float(a[:-1])
    train_acc.append(a)
    a = lines[i + 2].split()[-1]
    a = float(a[:-1])
    test_acc.append(a)

print(train_acc)
print(test_acc)

plt.plot(X, train_acc)
plt.plot(X, test_acc)

plt.show()