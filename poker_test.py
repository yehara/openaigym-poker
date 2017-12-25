import poker

env = poker.PokerEnv()

print (env._reset())

print(env.calc_score([0,13,26,39,4]))