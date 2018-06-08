import numpy
import matplotlib.pyplot as plt

rd1 = [3, 1.5, 0]
rd2 = [4, 1.5, 0]
rd3 = [3.5, 0.5, 0]
rd4 = [5.5, 1, 0]

bd1 = [2, 1, 1]
bd2 = [3, 1, 1]
bd3 = [2, 0.5, 1]
bd4 = [1, 1, 1]

plt.subplot(221)

data = [rd1, rd2, rd3, rd4, bd1, bd2, bd3, bd4]
plt.title("data set")
plt.grid()
plt.axis([0, 6, 0, 2])
plt.xlabel("width")
plt.ylabel("height")
for i in range(len(data)):
    point = data[i]
    color = "r"
    if point[2] == 1:
        color = "b"
    plt.scatter(point[0], point[1], c=color)

# guess
unknown_data1 = [4.5, 1]


def NN(m1, m2, w1, w2, bias):
    z = m1 * w1 + m2 * w2 + bias
    return z


def sigmoid(n):
    return 1 / (1 + numpy.exp(-n))


def train(unknown):
    w1 = numpy.random.randn()
    w2 = numpy.random.randn()
    b = numpy.random.randn()

    learning_rate = 0.1

    print(str(unknown) + " unknown initial guess:")
    init_guess = sigmoid(NN(unknown[0], unknown[1], w1, w2, b))
    print("guess: " + str(init_guess))
    print("means - " + ["RED", "BLUE"][round(init_guess)])

    costs = []

    for i in range(500000):
        ri = numpy.random.randint(len(data))
        point = data[ri]
        # print(pick_flower)
        target = point[2]

        z = NN(point[0], point[1], w1, w2, b)
        prediction = sigmoid(z)

        cost = (prediction - target) ** 2
        dcost_dpred = 2 * (prediction - target)

        dpred_dz = sigmoid(z) * (1 - sigmoid(z))

        dz_dw1 = point[0]
        dz_dw2 = point[1]
        dz_db = 1

        dcost_dz = dcost_dpred * dpred_dz

        dcost_dw1 = dcost_dz * dz_dw1
        dcost_dw2 = dcost_dz * dz_dw2
        dcost_db = dcost_dz * dz_db

        w1 = w1 - learning_rate * dcost_dw1
        w2 = w2 - learning_rate * dcost_dw2
        b = b - learning_rate * dcost_db

        costs_sum = 0
        # cost saving

        for point in data:
            z = NN(point[0], point[1], w1, w2, b)
            prediction = sigmoid(z)
            target = point[2]

            cost = (prediction - target) ** 2
            costs_sum += cost
        costs.append(costs_sum)

    unknown_pred = NN(unknown[0], unknown[1], w1, w2, b)
    final_guess = sigmoid(unknown_pred)

    print(str(unknown) + " final guess:")
    print("guess: " + str(final_guess))
    print("means - " + ["RED", "BLUE"][round(final_guess)])
    plt.subplot(222)
    plt.plot(costs)
    plt.title("Learning curve")
    plt.ylabel("Average cost")
    plt.xlabel("Iteration")
    plt.show()
    return


def round(i):
    if i < 0.5:
        return 0
    return 1


def gen_result():
    train(unknown_data1)


print("call train([width, length])")
print("call gen_result() to see the unknown original [4.5, 1, ??]")

