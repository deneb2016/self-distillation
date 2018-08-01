from matplotlib import pyplot as plt

without = '/home/seungkwan/Log_resnet34_cifar100_s1_without_sd_seed5.txt'
sto = '/home/seungkwan/Log_resnet34_cifar100_s1_t1.0_d0.0_seed5.txt'
distill = '/home/seungkwan/Log_resnet34_cifar100_s1_t13.0_d0.04_seed4.txt'


def parse_data(fn):
    f = open(fn, 'r')
    lines = f.readlines()
    train_loss = []
    val_loss = []
    hard_loss = []
    soft_loss = []

    for i in range(len(lines)):
        line = lines[i]
        if line[0] != '=':
            continue
        train_line = lines[i + 1]
        val_line = lines[i + 2]
        soft_loss.append(float(train_line.split()[-1]))
        hard_loss.append(float(train_line.split()[-3]))
        train_loss.append(float(train_line.split()[-7]))
        val_loss.append(float(val_line.split()[-3]))

    return {'soft': soft_loss, 'hard': hard_loss, 'train': train_loss, 'val': val_loss}


X = [i for i in range(1, 301)]

without_data = parse_data(without)
sto_data = parse_data(sto)
distill_data = parse_data(distill)

plt.plot(X, without_data['train'], color='r', linestyle='dashed')
plt.plot(X, without_data['val'], color='r')

plt.plot(X, sto_data['train'], color='g', linestyle='dashed')
plt.plot(X, sto_data['val'], 'g')

plt.plot(X, distill_data['train'], color='b', linestyle='dashed')
plt.plot(X, distill_data['val'], 'b')
plt.plot(X, distill_data['soft'], color='pink')
plt.plot(X, distill_data['hard'], color='yellow')

plt.legend(['no_sd_train', 'no_sd_val', 'sd_train', 'sd_val', 'distill_train', 'distill_val', 'distill_soft', 'distill_hard'])

plt.show()