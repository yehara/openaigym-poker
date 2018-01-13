import io
import gym
import numpy as np
import gym.spaces
from gym.utils import seeding
import sys

class PokerEnv(gym.Env):

    metadata = {'render.modes': ['human', 'ansi']}

    CARD_NUM = [ 'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    CARD_SUIT = [ '♥', '♦', '♠', '♣' ]
    SCORE = {
        'ROYAL_STRAIGHT_FLASH': 100,
        'FOUR_CARDS': 90,
        'STRAIGHT_FLASH': 80,
        'FULL_HOUSE': 70,
        'FLUSH': 50,
        'STRAIGHT': 40,
        'THREE_CARDS': 30,
        'TWO_PAIRS': 20,
        'ONE_PAIR': 10,
        'NONE': 0,
    }

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.MultiDiscrete([[0,1],[0,1],[0,1],[0,1],[0,1]])
        self.reward_range = [0, 100]
        self._seed()
        self._reset()

    def _reset(self):
        self.remain_cards = list(range(52))
        self.np_random.shuffle(self.remain_cards)
        self.observation_space = np.zeros(5, dtype=int)
        for i in range(5):
            self.observation_space[i] = self.get_next_card()
        self.initial_cards = self.cards_to_string(self.observation_space)
        return self.observation_space

    def _step(self, action_scalar):
        before = self.observation_space.copy()
        action = '{0:05b}'.format(action_scalar)
        for i in range(5):
            if (action_scalar >> i) % 2 == 1:
                self.observation_space[i] = self.get_next_card()
        after = self.observation_space.copy()
        # print('%s => %s => %s' % (before, action, after))
        return self.observation_space, self.calc_score(self.observation_space), True, {}

    def _render(self, mode='ansi', close=False):
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        last_cards = self.cards_to_string(self.observation_space)
        score = self.calc_score(self.observation_space)
        outfile.write("  {} => {}  score:{}  ".format(
            self.initial_cards,
            last_cards,
            score))
        return outfile

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _close(self):
        pass

    def get_next_card(self):
        return self.remain_cards.pop()

    def calc_score(self, cards):
        nums = np.array(cards, dtype=int) // 4
        suits = np.array(cards, dtype=int) % 4

        # 数字・スーツごとの枚数
        nums_count = np.zeros(14, dtype=int)
        suits_count = np.zeros(4, dtype=int)
        for i in range(5):
            nums_count[nums[i]] += 1
            suits_count[suits[i]] += 1
        nums_count[13] = nums_count[0]

        # 同じ・スーツが揃っている数ごとの件数。
        multi_cards = np.zeros(5, dtype=int)
        multi_suits = np.zeros(6, dtype=int)
        for i in range(13):
            multi_cards[int(nums_count[i])] += 1
        for i in range(4):
            multi_suits[int(suits_count[i])] += 1

        is_straight = False
        if multi_cards[1] == 5:
            for s in range(0, 10):
                if all(nums_count[s:s+5] == np.array([1, 1, 1, 1, 1])):
                    is_straight = True
        is_flash = (multi_suits[5] == 1)

        if multi_cards[4] == 1:
            return PokerEnv.SCORE['FOUR_CARDS']
        if is_straight and is_flash:
            if nums_count[0] == 1 and nums_count[9] == 1: # A と 10 を含む
                return PokerEnv.SCORE['ROYAL_STRAIGHT_FLASH']
            else:
                return PokerEnv.SCORE['STRAIGHT_FLASH']
        if multi_cards[3] == 1 and multi_cards[2] == 1:
            return PokerEnv.SCORE['FULL_HOUSE']
        if is_flash:
            return PokerEnv.SCORE['FLUSH']
        if is_straight:
            return PokerEnv.SCORE['STRAIGHT']
        if multi_cards[3] == 1:
            return PokerEnv.SCORE['THREE_CARDS']
        if multi_cards[2] == 2:
            return PokerEnv.SCORE['TWO_PAIRS']
        if multi_cards[2] == 1:
            return PokerEnv.SCORE['ONE_PAIR']
        return PokerEnv.SCORE['NONE']

    def cards_to_string(self, cards):
        c = ['', '', '', '', '']
        for i in range(5):
            c[i] = self.card_to_string(cards[i])
        return ' '.join(c)


    def card_to_string(self, card):
        (num, suit) = card // 4, card %4
        return PokerEnv.CARD_SUIT[suit] + PokerEnv.CARD_NUM[num]





