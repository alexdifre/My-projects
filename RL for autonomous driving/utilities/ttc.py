import numpy as np

def ttc_with_acceleration(d0, dv, da, eps=1e-3):
    """Calcola TTC con accelerazione costante."""
    if abs(da) < eps:  # caso lineare
        if dv > eps and d0 > 0:  # Si stanno avvicinando
            return d0 / dv
        else:
            return 100.0  
    
    disc = dv**2 - 2*da*d0
    if disc < 0:
        return 100.0  # Nessuna collisione prevista
    else:
        sqrt_disc = np.sqrt(disc)
        t1 = (-dv + sqrt_disc) / da
        t2 = (-dv - sqrt_disc) / da
        
        # Filtra solo tempi positivi
        valid_times = [t for t in (t1, t2) if t > 0]
        
        if len(valid_times) == 0:
            return 100.0  # Nessuna collisione futura
        else:
            return min(valid_times)


def get_ttc_front_rear(base_env, dt, min_front, min_back):
    """Trova i veicoli davanti e dietro nella stessa corsia e calcola TTC."""

    ego = base_env.vehicle
    ego_lane = ego.lane_index[2] if len(ego.lane_index) >= 3 else 0

    front_vehicle = None
    rear_vehicle = None
    min_front_dist = float('inf')
    min_rear_dist = float('inf')

    # Trova i veicoli più vicini davanti e dietro
    for v in base_env.road.vehicles:
        if v is ego:
            continue
        lane = v.lane_index[2] if len(v.lane_index) >= 3 else 0
        if lane != ego_lane:
            continue

        d0 = v.position[0] - ego.position[0]

        if d0 > 0 and d0 < min_front_dist:  # davanti
            min_front_dist = d0
            front_vehicle = v
        elif d0 < 0 and abs(d0) < min_rear_dist:  # dietro
            min_rear_dist = abs(d0)
            rear_vehicle = v
    
    # Inizializza prev_speed per l'ego vehicle se non esiste
    if not hasattr(ego, "prev_speed"):
        ego.prev_speed = ego.speed
    
    # Calcola accelerazione ego
    acceleration_ego = (ego.speed - ego.prev_speed) / dt
    ego_dict = {"x": ego.position[0], "v": ego.speed, "a": acceleration_ego}
    
    # Funzione helper per creare dizionario veicolo
    def vehicle_dict(v):
        if v is None:
            return None
        if not hasattr(v, "prev_speed"):
            v.prev_speed = v.speed
        acceleration_v = (v.speed - v.prev_speed) / dt
        v.prev_speed = v.speed
        return {"x": v.position[0], "v": v.speed, "a": acceleration_v}
    
    f_dict = vehicle_dict(front_vehicle)
    r_dict = vehicle_dict(rear_vehicle)
    
    # Aggiorna prev_speed per ego dopo aver processato tutto
    ego.prev_speed = ego.speed
    
    # Calcola TTC in base ai veicoli presenti
    if f_dict is not None and r_dict is None:
        d0 = f_dict["x"] - ego_dict["x"]
        dv = ego_dict["v"] - f_dict["v"]
        da = ego_dict["a"] - f_dict["a"]
        ttc_front = ttc_with_acceleration(d0, dv, da)
        # ⚠️ Gestisci None
        ttc_front = 100.0 if ttc_front is None else ttc_front
        return ttc_front, 100.0
        
    elif f_dict is None and r_dict is not None:
        d0 = ego_dict["x"] - r_dict["x"]
        dv = r_dict["v"] - ego_dict["v"]
        da = r_dict["a"] - ego_dict["a"]
        ttc_rear = ttc_with_acceleration(d0, dv, da)
        # ⚠️ Gestisci None
        ttc_rear = 100.0 if ttc_rear is None else ttc_rear
        return 100.0, ttc_rear
        
    elif f_dict is not None and r_dict is not None:
        # TTC frontale
        d0 = f_dict["x"] - ego_dict["x"]
        dv = ego_dict["v"] - f_dict["v"]
        da = ego_dict["a"] - f_dict["a"]
        ttc_front = ttc_with_acceleration(d0, dv, da)
        ttc_front = 100.0 if ttc_front is None else ttc_front
        
        # TTC posteriore
        d0 = ego_dict["x"] - r_dict["x"]
        dv = r_dict["v"] - ego_dict["v"]
        da = r_dict["a"] - ego_dict["a"]
        ttc_rear = ttc_with_acceleration(d0, dv, da)
        ttc_rear = 100.0 if ttc_rear is None else ttc_rear
        
        return ttc_front, ttc_rear
    else:
        # Nessun veicolo presente
        return 100.0, 100.0