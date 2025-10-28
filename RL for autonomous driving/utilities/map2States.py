import gymnasium as gym
from gymnasium import ObservationWrapper
import numpy as np
from .ttc import get_ttc_front_rear

class map2States(ObservationWrapper):
    def __init__(self, env ):
        super().__init__(env )
        self.lanes = 3
        self.y_ax = 2
        self.x_ax = 4
        self.ttc = 3
        self.num_bins_per_feature = [self.lanes, self.x_ax, self.x_ax,self.x_ax, self.x_ax, self.y_ax, self.y_ax,self.y_ax, self.y_ax, self.ttc, self.ttc]
        
    def map_to_bins(self, values, bins_edges):
        values = np.asarray(values)
        bin_indices = np.searchsorted(bins_edges, values, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, len(bins_edges) - 2)
        return bin_indices
    
    
    def combine_bins(self, bin_list):
        state = 0
        multiplier = 1
        for b, n in zip(reversed(bin_list), reversed(self.num_bins_per_feature)):
            state += b * multiplier
            multiplier *= n
        return state
    
    @staticmethod
    def features_extraction(obs_raw, env, dt):        
        # Filtriamo le macchine presenti e eliminiamo quelle che sono fuori dalla carreggiata
        condizione = obs_raw[:, 0] != 0
        obs = obs_raw[condizione]

        # Feature extraction
        ego_lane = obs[0][2]  # Lane index dell'ego vehicle
        
        # Inizializzazione distanze e coordinate
        min_front, min_back, min_left, min_right = 1e3, 1e3, 1e3, 1e3
        y_front, y_back, x_left, x_right = 0.0, 0.0, 0.0, 0.0
        
        # Coordinate relative rispetto all'ego (che è sempre obs[0])
        x = obs[:, 1]  # Distanza longitudinale (positiva = davanti)
        y = obs[:, 2]  # Distanza laterale (positiva = sinistra)
        
        # Trova veicolo più vicino davanti e dietro
        for t in range(len(x)):
            if x[t] > 0 and x[t] < min_front:  # Davanti
                min_front = x[t]
                y_front = y[t]
            elif x[t] < 0 and abs(x[t]) < min_back:  # Dietro
                min_back = abs(x[t])
                y_back = y[t]
        
        # Trova veicolo più vicino a sinistra e destra
        for t in range(len(y)):
            if y[t] > 0 and y[t] < min_left:  # Sinistra
                min_left = y[t]
                x_left = x[t]
            elif y[t] < 0 and abs(y[t]) < min_right:  # Destra
                min_right = abs(y[t])
                x_right = x[t]
        
        # Calcola TTC per veicoli davanti e dietro nella stessa corsia
        ttc_front, ttc_back = get_ttc_front_rear(env.unwrapped, dt, min_front, min_back)

        # Array finale con tutte le features
        obs = np.array([ego_lane, min_front, min_back, x_left, x_right, 
        min_left, min_right,y_front, y_back, ttc_front, ttc_back], dtype=np.float32) 
        
        return obs
    



    
    
    