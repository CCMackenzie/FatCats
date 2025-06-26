import numpy as np
import gymnasium as gym

from .game_config import FatCatsConfig
from .mechanics import build_treat_deck, deal_trick_hands, resolve_bids



class FatCatsEnv(gym.Env):

    """Gym environment for the Fat Cats card game."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, game_config: FatCatsConfig, seed: int | None = None) -> None:
        super().__init__()
        self.game_config = game_config
        self.random_number_generator = np.random.default_rng(seed)
        self._initailise_action_and_observation_spaces()
        obs_length = (
            1                                           # treat index
            + self.game_config.trick_cards_per_player        # padded hand
            + 1                                         # my score
            + self.game_config.treat_deck_size               # discarded treats
        )
        self._obs_buffer = np.empty(obs_length, dtype=np.int16)

    # Gym API methods
     
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.random_number_generator = np.random.default_rng(seed)
        self._initialize_game_state()
        return self._create_observation(0), {}
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        player_index: int = self.turn
        bid_value: int = self._apply_action_and_get_bid(player_index, action)
        self.current_round_bids[player_index] = bid_value

        # rotate turn pointer
        self.turn = (self.turn + 1) % self.game_config.number_of_players
        episode_terminated = truncated = False
        reward = 0.0

        # if the last player has acted then resolve round
        if self.turn == 0:
            self._resolve_current_round()
            # Episode ends after the final Treat card is resolved.
            episode_terminated = (
                self.treat_idx >= self.game_config.treat_deck_size
            )
        else:
            episode_terminated = False  # game continues
        
        observation = self._create_observation(self.turn)
        return observation, reward, episode_terminated, truncated, {}

    # Structure helper methods

    def _initailise_action_and_observation_spaces(self) -> None:
        maximum_hand_size: int = self.game_config.trick_cards_per_player
        if self.game_config.allow_multi_bid:
            # each bit represents whether the respective trick card is bid
            self.action_space = gym.spaces.MultiBinary(maximum_hand_size)
        else:
            self.action_space = gym.spaces.Discrete(maximum_hand_size + 1)

        observation_length: int = (
            1 + # treat index
            maximum_hand_size + # player hand
            1 + # player score
            + self.game_config.treat_deck_size # discards
        )

        max_card_value = max(max(self.game_config.treat_card_values), max(self.game_config.trick_card_values))

        self.observation_space = gym.spaces.Box(
            low=0,
            high=max_card_value,
            shape=(observation_length,),
            dtype=np.int16
        )

        return None
    
    def _initialize_game_state(self) -> None:
        """ Allocate and reset all game state structures. """
        self.treat_deck = build_treat_deck(
            self.random_number_generator,
            self.game_config.treat_card_values,
            self.game_config.treat_deck_size
        )
        self.player_hands = deal_trick_hands(
            self.game_config.number_of_players,
            self.game_config.trick_cards_per_player,
            self.game_config.trick_card_values,
            self.random_number_generator
        )
        self.player_scores = np.zeros(self.game_config.number_of_players, dtype=np.int16)
        self.discarded_treats = np.zeros(self.game_config.treat_deck_size, dtype=np.int16)
        self.current_round_bids = np.zeros(self.game_config.number_of_players, dtype=np.int16)
        self.treat_idx = 0
        self.turn = 0 # current player index

    # Game logic methods

    def _apply_action_and_get_bid(self, player_index: int, action: np.ndarray) -> int:
        """ Mutate player hands according to the action and return the bid value. """
        player_hand = self.player_hands[player_index]
        if self.game_config.allow_multi_bid:
            bitmask = np.asarray(action, dtype=np.int16)
            if not bitmask.any():
                return 0
            chosen_cards = np.flatnonzero(bitmask & (np.arange(len(player_hand)) < len(player_hand)))
            bid_sum = sum(player_hand[card] for card in chosen_cards)
            for card in sorted(map(int, chosen_cards), reverse=True):
                player_hand.pop(card)
            return bid_sum
        else:
            index: int = int(action)
            if index >= len(player_hand):
                return 0
            return int(player_hand.pop(index))
        
    def _resolve_current_round(self) -> None:
        """ Resolve the current round of bids and update game state. """
        winner = resolve_bids(self.current_round_bids)
        treat_value = self.treat_deck[self.treat_idx]
        if winner is not None:
            self.player_scores[winner] += treat_value
        else:
            self.discarded_treats[self.treat_idx] = treat_value

        # advance to next round
        self.current_round_bids.fill(0)
        self.treat_idx += 1
        
        return None
    
    def _create_observation(self, player_index: int) -> np.ndarray:
        buf = self._obs_buffer          # alias – no new alloc
        max_hand = self.game_config.trick_cards_per_player
        
        buf[0] = self.treat_idx

        hand_len = len(self.player_hands[player_index])
        if hand_len:
            buf[1 : 1 + hand_len] = self.player_hands[player_index, :hand_len]
        if hand_len < max_hand:
            buf[1 + hand_len : 1 + max_hand] = 0

        buf[1 + max_hand] = self.player_scores[player_index]

        buf[-self.game_config.treat_deck_size :] = self.discarded_treats

        return buf.copy()
