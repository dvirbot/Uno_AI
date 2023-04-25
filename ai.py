# Imports from uno(taken from github), copy, random, OpenAI gym, and numpy (all open source libraries)
from uno import UnoPlayer, UnoGame, ReversibleCycle, UnoCard
from copy import deepcopy
from random import shuffle, choice
import gym
from gym.spaces import Discrete
import numpy as np

# lists needed to classify different cards
COLORS = ['red', 'yellow', 'green', 'blue']
ALL_COLORS = COLORS + ['black']
NUMBERS = list(range(10)) + list(range(1, 10))
SPECIAL_CARD_TYPES = ['skip', 'reverse', '+2']
COLOR_CARD_TYPES = NUMBERS + SPECIAL_CARD_TYPES * 2
BLACK_CARD_TYPES = ['wildcard', '+4']
CARD_TYPES = NUMBERS + SPECIAL_CARD_TYPES + BLACK_CARD_TYPES


class UnoAI(UnoPlayer):
    """
    This is the player that the agent plays as. It holds all of the observation data that is used in training.

    functions:
    possible_actions(current_card: UnoCard)
    update_unknown_cards(card: UnoCard)
    drew_from_pile(card: UnoCard)
    opponent_put_down_card(card: UnoCard)
    opponent_picked_up(self, current_card: UnoCard)
    opponent_drew_from_pile(self, num_of_cards)
    reshuffle(self, deck)
    """
    def __init__(self, cards, first_card: UnoCard, player_id=None):
        """
        Creates and gives initial values to the opponent_deck, draw_pile, and hand_dict dictionaries;
        the draw_pile_size and opponent_deck_size attribute, and stores the initial_possible_actions.
        """
        super(UnoAI, self).__init__(cards, player_id)
        self.unknown_cards = {
            "black": {
                "wildcard": 4,
                "+4": 4,
            },
            "red": {
                0: 1,
                1: 2,
                2: 2,
                3: 2,
                4: 2,
                5: 2,
                6: 2,
                7: 2,
                8: 2,
                9: 2,
                "reverse": 2,
                "skip": 2,
                "+2": 2,
            },
            "green": {
                0: 1,
                1: 2,
                2: 2,
                3: 2,
                4: 2,
                5: 2,
                6: 2,
                7: 2,
                8: 2,
                9: 2,
                "reverse": 2,
                "skip": 2,
                "+2": 2,
            },
            "blue": {
                0: 1,
                1: 2,
                2: 2,
                3: 2,
                4: 2,
                5: 2,
                6: 2,
                7: 2,
                8: 2,
                9: 2,
                "reverse": 2,
                "skip": 2,
                "+2": 2,
            },
            "yellow": {
                0: 1,
                1: 2,
                2: 2,
                3: 2,
                4: 2,
                5: 2,
                6: 2,
                7: 2,
                8: 2,
                9: 2,
                "reverse": 2,
                "skip": 2,
                "+2": 2,
            }
        }
        self.unknown_cards_size = 108
        self.hand_dict = deepcopy(self.unknown_cards)
        for card in self.hand:
            self.unknown_cards[card.color][card.card_type] -= 1
            self.unknown_cards_size -= 1
        self.update_unknown_cards(first_card)
        self.opponent_deck_size = 7
        self.draw_pile_size = 93
        self.opponent_deck = deepcopy(self.unknown_cards)
        self.draw_pile = deepcopy(self.unknown_cards)
        self.initial_possible_actions = self.possible_actions(first_card)
        for color in self.unknown_cards:
            for card in self.unknown_cards[color]:
                self.opponent_deck[color][card] *= 7 / self.unknown_cards_size
                self.draw_pile[color][card] *= 93 / self.unknown_cards_size
                self.hand_dict[color][card] = 0
        for card in self.hand:
            self.hand_dict[card.color][card.card_type] += 1

    def possible_actions(self, current_card: UnoCard):
        """Returns a list of possible cards that can be played by the agent and the index in the
        agent's deck in which the cards are.
        Ex: current card is a yellow 7 and player cards are a green 3, a blue 7, a yellow 4, and a +4
        ([(7, 'blue'),(4, 'yellow'), ('+4', 'red'), ('+4', 'yellow'), ('+4', 'green'), ('+4', 'blue')],
        [1, 2, 3, 3, 3, 3])
        """
        if self.can_play(current_card):
            actions = []
            action_indexes = []
            for i, card in enumerate(self.hand):
                if card.color == 'black':
                    possible_new_colors = []
                    for other_card in self.hand:
                        if other_card.color not in possible_new_colors and other_card.color != "black":
                            possible_new_colors.append(other_card.color)
                    if len(possible_new_colors) == 0:
                        possible_new_colors = COLORS
                    for color in possible_new_colors:
                        actions.append((card.card_type, color))
                        action_indexes.append(i)
                else:
                    if current_card.playable(card):
                        actions.append((card.card_type, card.color))
                        action_indexes.append(i)
            return actions, action_indexes
        return [None], [-1]

    def update_unknown_cards(self, card: UnoCard):
        """Removes a card from the unknown_cards and decreases the number of unknown cards by 1. Returns the number
        of cards identical to the one now discovered are still unknown"""
        self.unknown_cards_size -= 1
        self.unknown_cards[card.color][card.card_type] -= 1
        return self.unknown_cards[card.color][card.card_type]

    def drew_from_pile(self, card: UnoCard):
        """Calculates what can be discovered from the card the agent drew from the deck and updates the necessary
        attributes."""
        unknown_cards_of_type = self.update_unknown_cards(card)
        shrink_coefficient = unknown_cards_of_type / (unknown_cards_of_type + 1)
        self.draw_pile_size -= 1
        self.hand_dict[card.color][card.card_type] += 1
        self.opponent_deck[card.color][card.card_type] *= shrink_coefficient
        self.draw_pile[card.color][card.card_type] *= shrink_coefficient
        for color in self.draw_pile:
            for card in self.draw_pile[color]:
                self.draw_pile[color][card] *= self.draw_pile_size / (self.draw_pile_size + 1)

    def opponent_put_down_card(self, card: UnoCard):
        """Calculates what can be discovered from about the game from the card the agent opponent put down nd updates
        the necessary attributes."""
        unknown_cards_of_type = self.update_unknown_cards(card)
        shrink_coefficient = unknown_cards_of_type / (unknown_cards_of_type + 1)
        self.opponent_deck_size -= 1
        self.opponent_deck[card.color][card.card_type] *= shrink_coefficient
        self.draw_pile[card.color][card.card_type] *= shrink_coefficient
        for color in self.opponent_deck:
            for card in self.opponent_deck[color]:
                self.opponent_deck[color][card] *= self.opponent_deck_size / (self.opponent_deck_size + 1)

    def opponent_picked_up(self, current_card: UnoCard):
        """Calculates what can be discovered about the cards in the draw pile and opponent hand from the fact that
        the opponent drew a card. For example, if the opponent drew when the card was a yellow 7, it means he has no
        7s, no yellows, and no black cards, which means that all these have to be in the draw pile."""
        amount_shifted = 0
        for color in self.unknown_cards:
            for card_type in self.unknown_cards[color]:
                if color == current_card.color or card_type == current_card.card_type or color == "black":
                    shift_amount = self.opponent_deck[color][card_type]
                    self.draw_pile[color][card_type] += shift_amount
                    amount_shifted += shift_amount
                    self.opponent_deck[color][card_type] = 0
        for color in self.unknown_cards:
            for card_type in self.unknown_cards[color]:
                if color != current_card.color and card_type != current_card.card_type:
                    self.draw_pile[color][card_type] *= (self.draw_pile_size - amount_shifted) / self.draw_pile_size
                    self.opponent_deck[color][card_type] *= self.opponent_deck_size / (
                            self.opponent_deck_size - amount_shifted)
        self.opponent_drew_from_pile(num_of_cards=1)

    def opponent_drew_from_pile(self, num_of_cards):
        """Moves the values from unknown cards from the draw_pile tot he opponent deck for each card the opponent drew"""
        for i in range(num_of_cards):
            self.opponent_deck_size += 1
            self.draw_pile_size -= 1
            for color in self.draw_pile:
                for card in self.draw_pile[color]:
                    self.opponent_deck[color][card] += self.draw_pile[color][card] / (self.draw_pile_size + 1)
                    self.draw_pile[color][card] *= self.draw_pile_size / (self.draw_pile_size + 1)

    def reshuffle(self, deck):
        """Updates the agent's observations of the deck given a reshuffle"""
        self.draw_pile_size = len(deck)
        self.unknown_cards_size += len(deck)
        for card in deck:
            self.unknown_cards[card.color][card.card_type] += 1
            self.draw_pile[card.color][card.card_type] += 1


class UnoGameWithAI(UnoGame):

    def __init__(self, players, random=True):
        if not isinstance(players, int):
            raise ValueError('Invalid game: players must be integer')
        if not 2 <= players <= 15:
            raise ValueError('Invalid game: must be between 2 and 15 players')
        self.deck = self._create_deck(random)
        self.discard_pile = [self.deck.pop(-1)]
        self.players = [
            UnoPlayer(self._deal_hand(), n) for n in range(players - 1)
        ]
        self.players.append(UnoAI(self._deal_hand(), self.current_card, "AI"))
        self._player_cycle = ReversibleCycle(self.players)
        self._current_player = next(self._player_cycle)
        self._winner = None

    @property
    def ai_player(self):
        """Property that tells the index of the AI player"""
        ai_player: UnoAI = self.players[-1]
        return ai_player

    def play(self, player, card=None, new_color=None):
        """Basically acts as the dealer like in monopoly, carrying out a valid action and following all the appropriate
        rules. If there is a winner, also updates the winner."""
        if player == "AI":
            player = len(self.players) - 1
        _player = self.players[player]
        if card is None:
            self._pick_up(_player, 1)
            next(self)
            return
        _card = _player.hand[card]

        played_card = _player.hand.pop(card)
        self.discard_pile.append(played_card)

        card_color = played_card.color
        card_type = played_card.card_type
        if card_color == 'black':
            self.current_card.temp_color = new_color
            if card_type == '+4':
                next(self)
                self._pick_up(self.current_player, 4)
        elif card_type == 'reverse':
            self._player_cycle.reverse()
            if len(self.players) == 2:
                next(self)
        elif card_type == 'skip':
            next(self)
        elif card_type == '+2':
            next(self)
            self._pick_up(self.current_player, 2)

        if self.is_active:
            next(self)
        else:
            self._winner = _player

    def _pick_up(self, player, n):
        """Gives players their cards from the deck when necessary. In the case the deck runs out of cards, it is
        reshuffled and the giving of the cards is resumed."""
        if n > len(self.deck):
            cards_left = n - len(self.deck)
            n = len(self.deck)
            penalty_cards = [self.deck.pop(0) for i in range(n)]
            player.hand.extend(penalty_cards)
            if str(player) == "AI":
                for card in penalty_cards:
                    self.ai_player.drew_from_pile(card)
            else:
                self.ai_player.opponent_drew_from_pile(n)
            self.deck = self.discard_pile[:-1]
            shuffle(self.deck)
            self.ai_player.reshuffle(self.deck)
            penalty_cards_2 = [self.deck.pop(0) for i in range(cards_left)]
            player.hand.extend(penalty_cards_2)
            if str(player) == "AI":
                for card in penalty_cards_2:
                    self.ai_player.drew_from_pile(card)
            elif (n + cards_left) == 1:
                self.ai_player.opponent_picked_up(self.current_card)
            else:
                self.ai_player.opponent_drew_from_pile(cards_left)
            # print(f"Player {player} drew {player.hand.extend(penalty_cards + penalty_cards_2)}")
            return
        penalty_cards = [self.deck.pop(0) for i in range(n)]
        player.hand.extend(penalty_cards)
        # print(f"Player {player} drew {penalty_cards}")
        if str(player) == "AI":
            for card in penalty_cards:
                self.ai_player.drew_from_pile(card)
        elif n == 1:
            self.ai_player.opponent_picked_up(self.current_card)
        else:
            self.ai_player.opponent_drew_from_pile(n)


def dict_to_list(dict_: dict):
    """Converts a dictionary with nested dictionaries to a list of the values of the nested dictionaries"""
    list_of_dicts = list(dict_.values())
    array = [list(inner_dict.values()) for inner_dict in list_of_dicts]
    flat_array = [item for sublist in array for item in sublist]
    return flat_array


class UnoAIEnvironment(gym.Env):
    """OpenAI gym custom environment for training the reinforcement learning AI. """
    def __init__(self):
        players = 2
        self.game = UnoGameWithAI(players)
        self.ai_player: UnoAI = self.game.ai_player
        self.first_move = True
        self.actions_map = {
            0: None,
            1: ("wildcard", "red"),
            2: ("wildcard", "green"),
            3: ("wildcard", "blue"),
            4: ("wildcard", "yellow"),
            5: ("+4", "red"),
            6: ("+4", "green"),
            7: ("+4", "blue"),
            8: ("+4", "yellow"),
            9: (0, "red"),
            10: (1, "red"),
            11: (2, "red"),
            12: (3, "red"),
            13: (4, "red"),
            14: (5, "red"),
            15: (6, "red"),
            16: (7, "red"),
            17: (8, "red"),
            18: (9, "red"),
            19: ("reverse", "red"),
            20: ("skip", "red"),
            21: ("+2", "red"),
            22: (0, "green"),
            23: (1, "green"),
            24: (2, "green"),
            25: (3, "green"),
            26: (4, "green"),
            27: (5, "green"),
            28: (6, "green"),
            29: (7, "green"),
            30: (8, "green"),
            31: (9, "green"),
            32: ("reverse", "green"),
            33: ("skip", "green"),
            34: ("+2", "green"),
            35: (0, "blue"),
            36: (1, "blue"),
            37: (2, "blue"),
            38: (3, "blue"),
            39: (4, "blue"),
            40: (5, "blue"),
            41: (6, "blue"),
            42: (7, "blue"),
            43: (8, "blue"),
            44: (9, "blue"),
            45: ("reverse", "blue"),
            46: ("skip", "blue"),
            47: ("+2", "blue"),
            48: (0, "yellow"),
            49: (1, "yellow"),
            50: (2, "yellow"),
            51: (3, "yellow"),
            52: (4, "yellow"),
            53: (5, "yellow"),
            54: (6, "yellow"),
            55: (7, "yellow"),
            56: (8, "yellow"),
            57: (9, "yellow"),
            58: ("reverse", "yellow"),
            59: ("skip", "yellow"),
            60: ("+2", "yellow"),
        }
        self.reverse_actions_map = {value: key for key, value in self.actions_map.items()}
        self.possible_actions = [self.reverse_actions_map[action] for action
                                 in self.ai_player.initial_possible_actions[0]]
        self.action_space = Discrete(61)
        observation_space_array_min = np.zeros(166)
        observation_space_array_max = np.array([4, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2,
                                                2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
                                                2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,

                                                4, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2,
                                                2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
                                                2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,

                                                4, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2,
                                                2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
                                                2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,

                                                108, 108, 108, 60], dtype=np.float32)
        self.observation_space = gym.spaces.Box(observation_space_array_min, observation_space_array_max)

    def step(self, action: int):
        """Plays a move in the game based on the action chosen by the AI and returns the associated observations
        and rewards for the AI to calculate the next move. Returns:
         np.array(observations): a numpy array of observations
         reward: a reward between -1 and 1, (-1 for loss, 1 for win, and 0 if the game is not over)
         done: whether the game is done
         info: an empty dictionary needed by OpenAI gym"""
        player_cards = len(self.ai_player.hand)
        opponent_cards = len(self.game.players[0].hand)
        if self.game.is_active:
            player = self.game.current_player
            player_id = player.player_id
            if player_id == "AI":
                action = self.actions_map[action]
                playable_indexes = self.ai_player.possible_actions(self.game.current_card)
                # action = playable_indexes[0][0]
                if action is None:
                    self.game.play(player="AI", card=None)
                elif action[0] == "wildcard" or action[0] == "+4":
                    # print("Player {} played {}".format(player, self.ai_player.hand[playable_indexes[1][playable_indexes[0].index(action)]]))
                    new_color = action[1]
                    self.game.play(player="AI", card=playable_indexes[1][playable_indexes[0].index(action)],
                                   new_color=new_color)

                else:
                    # print("Player {} played {}".format(player, self.ai_player.hand[playable_indexes[1][playable_indexes[0].index(action)]]))
                    self.game.play(player="AI", card=playable_indexes[1][playable_indexes[0].index(action)])
            if self.game.is_active and str(self.game.current_player) == "0":
                player = self.game.current_player
                player_id = player.player_id
                move = player.play(self.game.current_card)
                if move is not None:
                    # print("Player {} played {}".format(player, player.hand[move[0]]))
                    self.game.play(player=player_id, card=move[0], new_color=move[1])
                    self.ai_player.opponent_put_down_card(self.game.current_card)
                else:
                    self.game.play(player=player_id, card=None)
        reward = player_cards - len(self.ai_player.hand) + len(self.game.players[0].hand) - opponent_cards
        if not self.game.is_active:
            if str(self.game.winner) == "AI":
                reward += 50
                reward = 1
                done = True
            else:
                reward -= 50
                done = True
                reward = -1
        else:
            self.possible_actions = [self.reverse_actions_map[action] for action in
                                self.ai_player.possible_actions(self.game.current_card)[0]]
            done = False
            reward = 0

        info = {}
        observations = dict_to_list(self.ai_player.opponent_deck) + dict_to_list(
            self.ai_player.draw_pile) + dict_to_list(self.ai_player.hand_dict)
        observations.append(len(self.ai_player.hand))
        observations.append(self.ai_player.opponent_deck_size)
        observations.append(self.ai_player.draw_pile_size)
        if self.game.current_card._color == "black":
            observations.append(0)
        else:
            observations.append(
                self.reverse_actions_map[(self.game.current_card.card_type, self.game.current_card._color)])
        return np.array(observations), reward, done, info

    def reset(self):
        """Restarts the environment for another game to be played without creating another instance of the class"""
        players = 2
        self.game = UnoGameWithAI(players)
        self.ai_player: UnoAI = self.game.ai_player
        self.first_move = True
        self.possible_actions = [self.reverse_actions_map[action] for action in self.ai_player.initial_possible_actions[0]]
        observations = dict_to_list(self.ai_player.opponent_deck) + dict_to_list(
            self.ai_player.draw_pile) + dict_to_list(self.ai_player.hand_dict)
        observations.append(len(self.ai_player.hand))
        observations.append(self.ai_player.opponent_deck_size)
        observations.append(self.ai_player.draw_pile_size)
        if self.game.current_card._color == "black":
            observations.append(0)
        else:
            observations.append(
                self.reverse_actions_map[(self.game.current_card.card_type, self.game.current_card._color)])
        return np.array(observations)

    def action_mask(self):
        """Returns a list of boolean values for each action in the action space, with True if the move is legal and
        false otherwise. Used to make sure AI selects a legal move."""
        mask_list = []
        for i in range(61):
            if i in self.possible_actions:
                mask_list.append(True)
            else:
                mask_list.append(False)
        return mask_list
