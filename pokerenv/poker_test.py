import pokerenv

env = pokerenv.PokerEnv()

#  0:♥A 1:♦A 2:♠A 3:♣A 4:♥2 5:♦2...
assert env.calc_score([0,13,26,39,4]) == 0
assert env.calc_score([0,1,4,8,12]) == 10
assert env.calc_score([0,1,4,5,12]) == 20
assert env.calc_score([0,1,2,8,12]) == 30
assert env.calc_score([0,4,8,12,17]) == 40
assert env.calc_score([0,36,40,44,49]) == 40
assert env.calc_score([0,4,8,12,20]) == 50
assert env.calc_score([0,1,2,4,5]) == 70
assert env.calc_score([0,4,8,12,16]) == 80
assert env.calc_score([0,1,2,3,4]) == 90
assert env.calc_score([36,40,44,48,0]) == 100

