import numpy as np

def create_ocr_noise(correct_pr):
    assert (0 < correct_pr < 1), "WTF"
    wrong_pr = 1 - correct_pr
    return wrong_pr

class Universe:

    def __init__(self):
        self.student_correct_pr = np.random.uniform(0.8, 1.0)
        self.ocr_correct_pr = np.random.uniform(0.7, 1.0)
        self.ocr_noise = dict()
        for i in range(10):
            wrong_pr = np.array([np.random.random() for _ in range(10)])
            wrong_pr[0] = 0
            wrong_pr = wrong_pr / np.sum(wrong_pr)
            self.ocr_noise[i] = wrong_pr

    def gen_problem(self):
        return [np.random.randint(10) for _ in range(2)]

    def student(self, problem):
        if np.random.random() < self.student_correct_pr:
            return problem[0] + problem[1]
        else:
            std = problem[0] + problem[1] + (1 - np.random.randint(3))
            while std < 0:
                std = problem[0] + problem[1] + (1 - np.random.randint(3))
            return std

    def ocr(self, std_sol):
        std_sol = [int(x) for x in list(str(std_sol))]
        ret = []
        for x in std_sol:
            if np.random.random() < self.ocr_correct_pr:
                ret.append(x)
            else:
                x_wrong = np.random.choice([i for i in range(10)], p=self.ocr_noise[x])
                ret.append(x_wrong)
        return int("".join([str(x) for x in ret]))

    def gen_dataset(self, n = 1000):
        ret = []
        for _ in range(n):
            prob = self.gen_problem()
            sol = prob[0] + prob[1]
            std = self.student(prob)
            ocrr = self.ocr(std)
            ret.append((prob, sol, std, ocrr))
        return ret

    def score(self, data_pt, prediction):
        prob, sol, std, ocrr = data_pt
        true_judgement = sol == std
        return prediction == true_judgement

def simple_score(data_pt):
    prob, sol, _, ocrr = data_pt
    return sol == ocrr

if __name__ == '__main__':
    # print (gen_problem())
    universe = Universe()
    # print (universe.student_correct_pr)
    # print (universe.ocr_correct_pr)
    # print (universe.ocr_noise)
    # universe.student_correct_pr = 1.0
    # universe.ocr_correct_pr = 1.0

    prob = universe.gen_problem()
    sol = universe.student(prob)
    print (prob)
    print (sol)
    print (universe.ocr(sol))

    dataset = universe.gen_dataset(100)
    correct_judgement = 0
    for d in dataset:
        if universe.score(d, simple_score(d)):
            correct_judgement += 1
    print (correct_judgement)







