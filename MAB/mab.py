#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
class BernoulliMAB:
    def __init__(self,k):
        self.prob = np.random.uniform(size=k) # 随机生成的K个数字，每个人带有每个人的概率
        self.best_idx = np.argmax(self.prob) # 获奖拉杆的数值
        self.best_pro = self.prob[self.best_idx] # 最大的获奖概率为
        self.k =k
    def step(self,k):
        if np.random.rand() < self.prob[k]:
            return 1
        else :
            return 0

np.random.seed(0)
K = 50

mab_10 = BernoulliMAB(K)

class Agient:
    def __init__(self,bandit):
        # 记录一下 每个拉杆尝试的次数，当前懊悔值，状态，懊悔值积累
        self.bandit = bandit # 传入老虎机环境,实例化一个类
        self.counts = np.zeros(self.bandit.k)
        self.regret = 0.
        self.actions = []
        self.regrets = []

    def update_regret(self,k):
        # 采用蒙特卡洛增量式
        self.regret += self.bandit.best_pro - self.bandit.prob[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError # 只有被子类调用才可以失效，不然就error

    def run(self,num_steps):
        for _ in range (num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)



class e_greedy(Agient):
    def __init__(self, bandit,e = 0.1,init_prob = 1.0):
        super(e_greedy,self).__init__(bandit)
        self.e = e
        self.expection = np.array([init_prob] * self.bandit.k)
        print([init_prob])
        # 初始化所有拉杆的期望
        #print(self.expection)

    def run_one_step(self):
        if np.random.rand() < self.e:
            k = np.random.randint(0,self.bandit.k)
            #######################################################################################print('e')
        else:
            #print('~e')
            k = np.argmax(self.expection)
        r = self.bandit.step(k)
        self.expection[k] += 1. /(self.counts[k] + 1) *(r-self.expection[k])
        return k

def plot_results(Agient, Agient_name):
    for i,sol in enumerate(Agient):
        time_list = range(len(sol.regrets))
        plt.plot(time_list,sol.regrets,label =Agient_name[i])
    plt.xlabel('time steps')
    plt.ylabel('regrets')

    plt.legend()
    plt.show()


class time_e_greedy(Agient):
    def __init__(self, bandit,init_pro = 1.0):
        super(time_e_greedy,self).__init__(bandit)
        self.__expection = np.array([init_pro] * self.bandit.k)
        self.__num = 0
       # self.__expection

    def run_one_step(self):
        self.__num = self.__num + 1
        if np.random.rand() < 1 / self.__num:
            k = np.random.randint(0,self.bandit.k)
        else: 
            k = np.argmax(self.__expection)
        r = self.bandit.step(k)
        self.__expection[k] += 1./(self.counts[k]+1) * (r - self.__expection[k])
        print("k=",k,"Expection=", self.__expection)
        return k


# UCB算法：a_t = argmax(Q^(a) + cU^(a))
class UCB(Agient):
    def __init__(self,bandit,c,init_prob = 1.0):
        super(UCB,self).__init__(bandit)
        self.__c = c
        self.__expections = np.array([init_prob]* self.bandit.k)
        #self.__ucb = 0
        self.__total_count = 0

    def run_one_step(self):
        
        self.__total_count += 1         

        self.__ucb = self.__expections + self.__c * np.sqrt(   np.log(self.__total_count) / (2 * (self.counts+1)))
        k = np.argmax(self.__ucb)
        #self.__expections += 
        r = self.bandit.step(k)
        self.__expections[k] += 1./(self.counts[k]+1) * (r - self.__expections[k])
        print(" ucb_k=",k,"Expection=", self.__expections)
        return k

class ThompsomSampling(Agient):
    def __init__(self, bandit):
        super(ThompsomSampling,self).__init__(bandit)
        self.__a = np.ones(self.bandit.k)
        self.__b = np.ones(self.bandit.k)

    def run_one_step(self):
        samples = np.random.beta(self.__a,self.__b)
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self.__a[k] += r
        self.__b[k] += (1-r)
        
        return k

time_e_greedy_sol = time_e_greedy(mab_10,init_pro=1.0)

time_e_greedy_sol.run(5000)

e_greedy_sol = e_greedy(mab_10,init_prob=1.0)
e_greedy_sol.run(5000)
#e_greedy_sol_list = [time_e_greedy_sol,e_greedy_sol]
#e_greedy_name = ["time-e-greedy","e-greedy"]
ucb_agient = UCB(mab_10,c=0.5,init_prob=1.0)
ucb_agient.run(5000)

TS_agient = ThompsomSampling(mab_10)
TS_agient.run(5000)


e_greedy_sol_list = [time_e_greedy_sol,e_greedy_sol,ucb_agient,TS_agient]
e_greedy_name = ["time-e-greedy","e-greedy","ucb_agient","Thompsom"]

plot_results(e_greedy_sol_list,e_greedy_name)



#e_data = [1e-4,0.01,0.5,0.1]
#e_greedy_sol_list = [e_greedy(mab_10,e = pika) for pika in e_data]
#e_greedy_name = ["e={}".format(pika) for pika in e_data ]
#
#for sol in e_greedy_sol_list:
#    sol.run(5000)

#plot_results(e_greedy_sol_list,e_greedy_name)
