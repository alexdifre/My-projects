import numpy as np

def ego_lane(lane_width=4.0):    
    lane_centers = np.array([lane_width, lane_width * 2, lane_width * 3])
    return lane_centers
   
def states_x_ax_pos(NUM_PUNTI=4):
    K_VALUE = 0.045
    Y_MAX = 100
    

    def expe(x, k):
        return np.exp(-k * x)

    # Genera i punti x
    x_punti = np.linspace(0, Y_MAX, NUM_PUNTI)

    # Calcola y
    y_punti = expe(x_punti, K_VALUE)

    # Inverti, moltiplica per 100 e formatta
    thr_states = [f'{y_val*100:.6f}' for y_val in y_punti[::-1]]

    print(thr_states)
    return thr_states


def states_x_ax_neg(NUM_PUNTI = 4):
    K_VALUE = 0.045
    Y_MAX = 100
    

    def expe(x, k):
        return np.exp(-k * x)

    # Genera i punti x
    x_punti = np.linspace(0, Y_MAX, NUM_PUNTI)

    # Calcola y
    y_punti = expe(x_punti, K_VALUE)

    # Inverti, moltiplica per 100 e formatta
    thr_states = [f'{y_val*100*-1:.6f}' for y_val in y_punti[::-1]]

    print(thr_states)
    return thr_states

def states_y_ax_pos(NUM_PUNTI=2):
    K_VALUE = 0.07
    Y_MAX = 50

    def expe(x, k):
        return np.exp(-k * x)

    # Genera i punti x
    x_punti = np.linspace(0, Y_MAX, NUM_PUNTI)

    # Calcola y
    y_punti = expe(x_punti, K_VALUE)

    # Inverti, moltiplica per 100 e formatta
    thr_states = [f'{y_val*50:.6f}' for y_val in y_punti[::-1]]

    print(thr_states)
    return thr_states


def states_y_ax_neg(NUM_PUNTI=2):
    K_VALUE = 0.07
    Y_MAX = 50

    def expe(x, k):
        return np.exp(-k * x)

    # Genera i punti x
    x_punti = np.linspace(0, Y_MAX, NUM_PUNTI)

    # Calcola y
    y_punti = expe(x_punti, K_VALUE)

    # Inverti, moltiplica per 100 e formatta
    thr_states = [f'{y_val*50*-1:.6f}' for y_val in y_punti[::-1]]

    print(thr_states)
    return thr_states

def time_collision(NUM_PUNTI=3):
    K_VALUE = 0.3
    Y_MAX = 10

    def expe(x, k):
        return np.exp(-k * x)

    # Genera i punti x
    x_punti = np.linspace(0, Y_MAX, NUM_PUNTI)

    # Calcola y
    y_punti = expe(x_punti, K_VALUE)

    # Inverti, moltiplica per 100 e formatta
    thr_states = [f'{y_val*10:.6f}' for y_val in y_punti[::-1]]

    print(thr_states)
    return thr_states




