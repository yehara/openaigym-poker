import numpy as np
import gym
import pokerenv

ENV_NAME = 'PokerEnv-v1'
env = gym.make(ENV_NAME)

np.random.seed(123)
env.seed(123)

def compute(cards):
    change = np.zeros(5, dtype=bool)

    nums = np.array(cards, dtype=int) // 4
    suits = np.array(cards, dtype=int) % 4
    nums_count = np.zeros(14, dtype=int)
    suits_count = np.zeros(4, dtype=int)
    for i in range(5):
        nums_count[nums[i]] += 1
        suits_count[suits[i]] += 1
    nums_count[13] = nums_count[1]
    #フラッシュ
    for i in range(4):
        if suits_count[i] == 5:
            return change
        if suits_count[i] == 4:
            for j in range(5):
                if suits[j] != i:
                    change[j] = True
                    return change
    #ストレート
    for i in range(10):
        if all(nums_count[i:i+5] == np.array([1, 1, 1, 1, 1])):
            return change
    for i in range(11):
        if all(nums_count[i:i+4] >= np.array([1, 1, 1, 1])):
            keep = np.zeros(14, dtype=int)
            keep[i:i+4] = [1, 1, 1, 1]
            keep[0] += keep[13]
            remain = np.array((nums_count - keep)[0:13])
            num = remain.argmax()
            for j in range(5):
                if nums[j] == num:
                    change[j] = True
                    return change
    for i in range(10):
        for mask in [[1,0,1,1,1],[1,1,0,1,1], [1,1,1,0,1]]:
            if all(nums_count[i:i+5] >= np.array(mask)):
                keep = np.zeros(14, dtype=int)
                keep[i:i+5] = mask
                keep[0] += keep[13]
                remain = np.array((nums_count - keep)[0:13])
                num = remain.argmax()
                for j in range(5):
                    if nums[j] == num:
                        change[j] = True
                        return change

    for i in range(13):
        # フォーカード
        if nums_count[i] == 4:
            return change
        # フルハウス/スリーカード
        if nums_count[i] == 3:
            for j in range(13):
                if nums_count[j] == 2:
                    return change
            for k in range(5):
                if nums[k] != i:
                    change[k] = True
            return change
    for i in range(13):
        if nums_count[i] == 2:
            for j in range(i+1, 13):
                if nums_count[j] == 2:
                    for k in range(5):
                        if nums[k] != i and nums[k] != j:
                            change[k] = True
                            return change
            for k in range(5):
                if nums[k] != i:
                    change[k] = True
            return change

    change[1:5] = [True, True, True, True]
    return change

def serialize(a):
    res = 0
    for i in range(5):
        if a[i]:
            res += (1 << i)
    return res


sum = 0
count = 0

for i in range(10000):
    cards = env.reset().copy()
    change = compute(cards)
    env.step(serialize(change))
    sum += env.calc_score(env.observation_space)
    print("{} => {}    score:{}  ".format(env.cards_to_string(cards), env.cards_to_string(env.observation_space), env.calc_score(env.observation_space)))

print("avarage: {}".format(sum/count))




