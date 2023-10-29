# Imports from ai (from this project), random, itertools, threading, time, os, sb3_contrib, pgzrun, and pygame (all open source libraries)
from ai import *
from random import shuffle, choice, randint
from itertools import product, repeat, chain
from threading import Thread
from time import sleep
import os
from sb3_contrib import MaskablePPO
import pgzrun
import pygame

# Global constants taken from https://github.com/bennuttall/uno/blob/master/uno_pgz.py
COLORS = ['red', 'yellow', 'green', 'blue']
ALL_COLORS = COLORS + ['black']
NUMBERS = list(range(10)) + list(range(1, 10))
SPECIAL_CARD_TYPES = ['skip', 'reverse', '+2']
COLOR_CARD_TYPES = NUMBERS + SPECIAL_CARD_TYPES * 2
BLACK_CARD_TYPES = ['wildcard', '+4']
CARD_TYPES = NUMBERS + SPECIAL_CARD_TYPES + BLACK_CARD_TYPES


# Class taken from https://github.com/bennuttall/uno/blob/master/uno_pgz.py
class GameData:
    def __init__(self):
        self.selected_card = None
        self.selected_color = None
        self.color_selection_required = False
        self.log = ''

    @property
    def selected_card(self):
        selected_card = self._selected_card
        self.selected_card = None
        return selected_card

    @selected_card.setter
    def selected_card(self, value):
        self._selected_card = value

    @property
    def selected_color(self):
        selected_color = self._selected_color
        self.selected_color = None
        return selected_color

    @selected_color.setter
    def selected_color(self, value):
        self._selected_color = value


game_data = GameData()


# Class taken from https://github.com/bennuttall/uno/blob/master/uno_pgz.py, edited to mimic the UnoAIEnvironment class
# from ai.py.
class AIUnoGame:
    """A class that allows a player to play against an ai opponent. Takes as an argument the path of the model of the
    AI"""

    def __init__(self, model_path):
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
        self.player = self.game.players[0]
        self.player_index = self.game.players.index(self.player)
        self.model = MaskablePPO.load(model_path)

    def __next__(self):
        """Plays the next move in the Uno game. If it is the player's turn, it selects a card based on the game data.
        if it is the AI's turn, it plays based on the best action predicted by the model."""
        game = self.game
        player = game.current_player
        player_id = player.player_id
        current_card = game.current_card
        if player == self.player:
            game_data.log = "Your Turn"
            played = False
            while not played:
                card_index = None
                while card_index is None:
                    card_index = game_data.selected_card
                new_color = None
                if card_index is not False:
                    card = player.hand[card_index]
                    if not game.current_card.playable(card):
                        game_data.log = 'You cannot play that card'
                        continue
                    else:
                        game_data.log = 'You played card {:full}'.format(card)
                        if card.color == 'black' and len(player.hand) > 1:
                            game_data.color_selection_required = True
                            while new_color is None:
                                game_data.log = "please select a color"
                                new_color = game_data.selected_color
                            game_data.log = 'You selected {}'.format(new_color)
                else:
                    card_index = None
                    game_data.log = 'You picked up'
                game.play(player_id, card_index, new_color)
                played = True
        elif player.can_play(game.current_card):
            obs = self.get_observations()
            action_masks = self.get_action_mask()
            action = self.model.predict(obs, action_masks=action_masks)[0]
            action = self.actions_map[action]
            playable_indexes = self.ai_player.possible_actions(self.game.current_card)
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
        else:
            game_data.log = "Player {} picked up".format(player)
            game.play(player=player_id, card=None)

    def get_observations(self):
        """Creates a numpy array of observations for the AI to play by"""
        observations = dict_to_list(self.ai_player.opponent_deck) + dict_to_list(
            self.ai_player.draw_pile) + dict_to_list(self.ai_player.hand_dict)
        observations.append(len(self.ai_player.hand))
        observations.append(self.ai_player.opponent_deck_size)
        observations.append(self.ai_player.draw_pile_size)
        if self.game.current_card.color_in_practice == "black":
            observations.append(0)
        else:
            observations.append(
                self.reverse_actions_map[(self.game.current_card.card_type, self.game.current_card.color_in_practice)])
        return np.array(observations)

    def get_action_mask(self):
        """Creates the action mask list that informs the AI what moves are legal"""
        self.possible_actions = [self.reverse_actions_map[action] for action in
                                 self.ai_player.possible_actions(self.game.current_card)[0]]
        mask_list = []
        for i in range(61):
            if i in self.possible_actions:
                mask_list.append(True)
            else:
                mask_list.append(False)
        return mask_list

    def print_hand(self):
        """Prints the hand of the human player"""
        print('Your hand: {}'.format(
            ' '.join(str(card) for card in self.player.hand)
        ))


# ---------------------------------------------------------------------------#

# Sets up the game and the pygame zero screen
ai_model_path = os.path.join("Training", "Uno_Model_MaskablePPO_50M_3_Layers")
game = AIUnoGame(model_path=ai_model_path)

WIDTH = 1500
HEIGHT = 700

deck_img = Actor('back')  # Image taken from https://github.com/bennuttall/uno/tree/master/images
color_imgs = {color: Actor(color) for color in COLORS}
sprites = []


# Starts the game loop
def game_loop():
    while game.game.is_active:
        sleep(0.5)
        next(game)


game_loop_thread = Thread(target=game_loop)
game_loop_thread.start()


# Function taken from https://github.com/bennuttall/uno/blob/master/uno_pgz.py, edited to include the UI elements
# I added
def draw_deck():
    """Draws the UI of the game not including the player's hands"""
    deck_img.pos = (130, 110)
    deck_img.draw()
    deck_text = "Draw Pile"
    screen.draw.text(deck_text, (25, 20), fontsize=40, color="black")
    current_card = game.game.current_card
    sprite = Actor('{}_{}'.format(current_card.color, current_card.card_type))  # Image taken from
    # https://github.com/bennuttall/uno/tree/master/images
    sprite.pos = (210, 110)
    sprite.draw()
    current_card_text = "Current Card"
    screen.draw.text(current_card_text, (190, 20), fontsize=40, color="black")
    instruction_text = "Click a card to play it"
    screen.draw.text(instruction_text, (130, 240), fontsize=30, color="black")
    reset_sprite = Actor('reset_button')  # Image Creative Commons no rights reserved (CC0)
    reset_sprite._surf = pygame.transform.scale(reset_sprite._surf, (70, 70))
    reset_sprite.pos = (200, 750)
    reset_sprite.draw()
    if game_data.color_selection_required:
        for i, card in enumerate(color_imgs.values()):
            card.pos = (290 + i * 80, 110)
            card.draw()
    elif current_card.color == 'black' and current_card.temp_color is not None:
        color_img = color_imgs[current_card.temp_color]
        color_img.pos = (290, 110)
        color_img.draw()


# Function taken from https://github.com/bennuttall/uno/blob/master/uno_pgz.py
def draw_players_hands():
    """Draws the UI the player's hands"""
    global sprites
    sprites = []
    for p, player in enumerate(game.game.players):
        color = 'red' if player == game.game.current_player else 'black'
        text = '{} {}'.format("You" if p == 0 else "AI",
                              'win' if game.game.winner == player and p == 0 else 'wins' if game.game.winner == player else '')
        screen.draw.text(text, (0, 300 + p * 130), fontsize=80, color=color)
        for c, card in enumerate(player.hand):
            if player == game.player:
                sprite = Actor('{}_{}'.format(card.color, card.card_type))  # Image taken from
                # https://github.com/bennuttall/uno/tree/master/images
                sprites.append(sprite)
            else:
                sprite = Actor('back')  # Image taken from https://github.com/bennuttall/uno/tree/master/images
            sprite.pos = (150 + c * 80, 330 + p * 130)
            sprite.draw()


# Function taken from https://github.com/bennuttall/uno/blob/master/uno_pgz.py
def show_log():
    screen.draw.text(game_data.log, midbottom=(WIDTH / 2, HEIGHT - 50), color='black')


# Function taken from https://github.com/bennuttall/uno/blob/master/uno_pgz.py
def update():
    """Updates screen"""
    screen.clear()
    screen.fill((255, 255, 255))
    draw_deck()
    draw_players_hands()
    show_log()


# Function taken from https://github.com/bennuttall/uno/blob/master/uno_pgz.py, edited to include another button
def on_mouse_down(pos):
    """Function called when the mouse is pressed. It is used to detect clicking on cards, the deck, or the
    reset button"""
    global sprites, game
    if game.player == game.game.current_player:
        for i in range(len(game.player.hand)):
            if sprites[i].collidepoint(pos):
                game_data.selected_card = i
                print('Selected card {} index {}'.format(game.player.hand[i], i))
        if deck_img.collidepoint(pos):
            game_data.selected_card = False
            print('Selected pick up')
        for color, card in color_imgs.items():
            if card.collidepoint(pos):
                game_data.selected_color = color
                game_data.color_selection_required = False
    if 30 < pos[0] < 110 and 580 < pos[1] < 660:
        game = AIUnoGame(ai_model_path)
        game_loop_thread = Thread(target=game_loop)
        game_loop_thread.start()


# Starts the game
pgzrun.go()
