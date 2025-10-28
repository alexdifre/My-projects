# Remember to code your baseline for this problem
import gymnasium as gym
import highway_env
import numpy as np
def configuration(none):

    config = {
    'action': {
        'type': 'DiscreteMetaAction'
    },
    'observation': {
        "type": "Kinematics",
        "lanes_count": 3,
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        
        "collision_reward": -4,  # The reward received when colliding with a vehicle.
        "reward_speed_range": [21, 26],
        "offroad_terminal": False,
        'right_lane': 0.3 ,
        # GENERALI
        'duration': 40,
        'lanes_count': 3,
        "vehicles_count": 5,
    }
}
    return config

class HeuristicPolicy:
    def __init__(self):
        # Parametri configurabili
        self.safety_time_gap = 2.0  # secondi
        self.min_front_distance = 0.15
        self.min_side_distance = 0.25
        self.comfortable_speed = 0.7  # velocità target relativa
        self.lane_change_benefit_threshold = 0.1
        
    def act(self, obs):
        """
        obs: array shape (5, 5)
        features: [presence, x_rel, y_rel, vx_rel, vy_rel]
        ego is always obs[0]
        3 lanes: left (y≈1), center (y≈0), right (y≈-1)
        """
        ego = obs[0]
        current_lane = self.get_current_lane(0)  # ego is at y=0 in relative coords
        
        # Analizza la situazione attorno all'ego
        front_vehicle = self.get_closest_vehicle_in_lane(obs, current_lane, ahead=True)
        situation = self.assess_situation(obs, front_vehicle)
        
        # Decisione strategica basata sulla situazione
        action = self.decide_action(obs, situation, current_lane, front_vehicle)
        
        return action
    
    def get_current_lane(self, y_pos):
        """Determina la corsia corrente basandosi sulla posizione y."""
        if y_pos > 0.5:
            return 1  # left lane
        elif y_pos < -0.5:
            return -1  # right lane
        else:
            return 0  # center lane
    
    def get_closest_vehicle_in_lane(self, obs, lane, ahead=True):
        """Trova il veicolo più vicino nella corsia specificata."""
        closest = None
        min_dist = float('inf')
        
        for i, veh in enumerate(obs[1:], start=1):
            presence, x_rel, y_rel, vx_rel, vy_rel = veh
            if presence < 0.5:
                continue
            
            veh_lane = self.get_current_lane(y_rel)
            if veh_lane != lane:
                continue
            
            # Filtra per direzione
            if ahead and x_rel <= 0:
                continue
            if not ahead and x_rel >= 0:
                continue
            
            dist = abs(x_rel)
            if dist < min_dist:
                min_dist = dist
                closest = {'index': i, 'data': veh, 'distance': dist}
        
        return closest
    
    def assess_situation(self, obs, front_vehicle):
        """Valuta la situazione corrente."""
        situation = {
            'safe': True,
            'congested': False,
            'emergency': False,
            'can_accelerate': True
        }
        
        if front_vehicle is None:
            situation['can_accelerate'] = True
            return situation
        
        _, x_rel, y_rel, vx_rel, vy_rel = front_vehicle['data']
        distance = front_vehicle['distance']
        
        # Calcola il time-to-collision se ci stiamo avvicinando
        relative_speed = -vx_rel  # velocità relativa (positiva = ci avviciniamo)
        
        if relative_speed > 0:
            ttc = distance / relative_speed if relative_speed > 0.01 else float('inf')
        else:
            ttc = float('inf')
        
        # Valutazione situazione
        if distance < self.min_front_distance or ttc < 1.5:
            situation['emergency'] = True
            situation['safe'] = False
        elif distance < 0.3 or ttc < self.safety_time_gap:
            situation['congested'] = True
            situation['safe'] = False
        
        if distance < 0.4 or relative_speed > 0.1:
            situation['can_accelerate'] = False
        
        return situation
    
    def decide_action(self, obs, situation, current_lane, front_vehicle):
        """Decide l'azione ottimale basandosi sulla situazione."""
        
        # Situazione di emergenza: frena
        if situation['emergency']:
            return 4  # Slower
        
        # Situazione congestionata: considera cambio corsia o rallenta
        if situation['congested']:
            best_lane = self.find_best_lane(obs, current_lane)
            
            if best_lane is not None and best_lane != current_lane:
                # Esegui cambio corsia se sicuro
                if best_lane == current_lane + 1:  # sinistra
                    if self.is_lane_change_safe(obs, 1):
                        return 0  # Lane left
                elif best_lane == current_lane - 1:  # destra
                    if self.is_lane_change_safe(obs, -1):
                        return 2  # Lane right
            
            # Se non possiamo cambiare corsia, mantieni distanza
            return 1  # Idle (mantieni velocità)
        
        # Nessun pericolo immediato: ottimizza velocità
        if situation['can_accelerate']:
            # Controlla se c'è spazio davanti per accelerare
            if front_vehicle is None or front_vehicle['distance'] > 0.5:
                return 3  # Faster
        
        # Default: mantieni velocità corrente
        return 1  # Idle
    
    def find_best_lane(self, obs, current_lane):
        """Trova la corsia con più spazio libero davanti."""
        lanes = [-1, 0, 1]  # right, center, left
        lane_scores = {}
        
        for lane in lanes:
            # Non considerare corsie inesistenti
            if lane < -1 or lane > 1:
                continue
            
            # Calcola score per questa corsia
            score = self.evaluate_lane(obs, lane, current_lane)
            lane_scores[lane] = score
        
        # Trova la corsia migliore
        best_lane = max(lane_scores, key=lane_scores.get)
        
        # Cambia solo se c'è un beneficio significativo
        if lane_scores[best_lane] > lane_scores[current_lane] + self.lane_change_benefit_threshold:
            return best_lane
        
        return None  # Resta nella corsia corrente
    
    def evaluate_lane(self, obs, lane, current_lane):
        """Valuta quanto è attraente una corsia (score più alto = migliore)."""
        score = 0.0
        
        # Trova veicoli davanti in questa corsia
        front_veh = self.get_closest_vehicle_in_lane(obs, lane, ahead=True)
        
        if front_veh is None:
            # Nessun veicolo davanti: ottimo!
            score += 1.0
        else:
            # Score basato sulla distanza
            distance = front_veh['distance']
            score += min(distance / 0.5, 1.0)  # normalizza a [0, 1]
            
            # Considera anche la velocità relativa
            _, _, _, vx_rel, _ = front_veh['data']
            if vx_rel < 0:  # veicolo davanti più lento
                score -= 0.2
            elif vx_rel > 0:  # veicolo davanti più veloce
                score += 0.1
        
        # Penalità per cambio corsia multiplo
        lane_change_distance = abs(lane - current_lane)
        if lane_change_distance > 1:
            score -= 0.5  # Non possiamo fare doppio cambio corsia
        
        return score
    
    def is_lane_change_safe(self, obs, direction):
        """
        Verifica se un cambio di corsia è sicuro.
        direction: 1 (sinistra) o -1 (destra)
        """
        target_lane_y = direction  # approssimazione della y target
        
        for veh in obs[1:]:
            presence, x_rel, y_rel, vx_rel, vy_rel = veh
            if presence < 0.5:
                continue
            
            # Verifica se il veicolo è nella corsia target
            veh_lane = self.get_current_lane(y_rel)
            target_lane = self.get_current_lane(0) + direction
            
            if veh_lane != target_lane:
                continue
            
            # Controlla distanza laterale e longitudinale
            if abs(x_rel) < self.min_side_distance:
                return False
            
            # Controlla veicoli che si avvicinano velocemente da dietro
            if x_rel < 0 and -vx_rel > 0.2:  # veicolo dietro più veloce
                if abs(x_rel) < 0.3:
                    return False
        
        return True

N_EPISODES_TEST = 50
episode_rewards = []
episode_lengths = []
successes = []
if __name__ == "__main__":
    # Senza rendering per test più veloci
    env = gym.make("highway-v0", render_mode= "human",config=configuration(None))
    policy = HeuristicPolicy()

    for ep in range(N_EPISODES_TEST):
        obs, info = env.reset()
        
        done, truncated = False, False
        total_reward, steps = 0, 0
        success = 1

        while not (done or truncated):
            action = policy.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1

            if info.get("crashed", False):
                success = 0

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        successes.append(success)

    env.close()


    # 🔹 Analisi statistica
    mean_episode = np.mean(episode_lengths)
    std_episode = np.std(episode_lengths)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    se = std_reward / np.sqrt(N_EPISODES_TEST)                # errore standard
    # intervallo di confidenza 95%

    import scipy.stats as stats
    ci = stats.norm.interval(0.95, loc=mean_reward, scale=se)

    success_rate = np.mean(successes) * 100

    print("\n📊 Risultati valutazione:")
    print(f"episode medio: {mean_episode:.2f} ± {std_episode:.2f}")
    print(f"Reward medio: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Reward medio: {mean_reward:.2f},  "
        f"(IC 95%: {ci[0]:.2f} – {ci[1]:.2f})")

    """Risultati valutazione:
    episode medio: 37.02 ± 8.63
    Reward medio: 26.07 ± 6.26
    Success rate: 86.0%
    Reward medio: 26.07,  (IC 95%: 24.34 – 27.81)"""