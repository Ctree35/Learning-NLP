import numpy as np
from data import Universe


def inference(dataset, pca):
    correct_judge = 0
    for data_pt in dataset:
        prob, sol, stu, ocrr = data_pt
        pred = False
        if pca[sol][ocrr] >= 0.5:
            pred = True
        if pred == (stu == sol):
            correct_judge += 1

    print("Total Accuracy is: {}".format(correct_judge / len(dataset)))


def main():
    train_num = 100000
    test_num = 10000
    universe = Universe()
    dataset = universe.gen_dataset(train_num)
    pba = np.zeros((19, 21), dtype=float)  # pba[a][b] = Pr(B=b | A=a)
    pdigit = np.zeros((10, 10), dtype=float)  # pdigit[x][y] = Pr(ocrr=y | real=x)
    total_cnt = 0
    for data_pt in dataset:
        prob, sol, stu, ocrr = data_pt
        total_cnt += (sol == stu)
        pred = list(str(ocrr))
        stu = list(str(stu))
        for i in range(len(pred)):
            pdigit[int(stu[i])][int(pred[i])] += 1
    wrong_p = 1 - (total_cnt / len(dataset))
    pba[0][0] = 1 - wrong_p
    pba[0][1] = wrong_p / 2
    pba[0][2] = wrong_p / 2
    pba[1][1] = 1 - wrong_p
    pba[1][0] = wrong_p / 3
    pba[1][2] = wrong_p / 3
    pba[1][3] = wrong_p / 3
    for i in range(2, 19):
        for j in range(i-2, i+3):
            if i == j:
                pba[i][j] = 1 - wrong_p
            else:
                pba[i][j] = wrong_p / 4
    # print(pba)
    pdigit = pdigit / np.sum(pdigit, axis=1, keepdims=True)
    pcb = np.zeros((21, 100), dtype=float)
    for i in range(21):
        for j in range(100):
            if len(str(i)) != len(str(j)):
                continue
            pcb[i][j] = 1.
            b = list(str(i))
            c = list(str(j))
            for k in range(len(b)):
                pcb[i][j] *= pdigit[int(b[k])][int(c[k])]
    pca = np.zeros((19, 100), dtype=float)
    for i in range(19):
        for j in range(100):
            pr = pba[i][i] * pcb[i][j]
            total = 0
            for k in range(21):
                total += pba[i][k] * pcb[k][j]
            if total == 0:
                pca[i][j] = 0
            else:
                pca[i][j] = pr / total
    test_set = universe.gen_dataset(test_num)
    inference(test_set, pca)

if __name__ == '__main__':
    main()
