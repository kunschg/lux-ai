import numpy as np

def place_best_ice(player, obs):
    
    opp_player = "player_1" if player == "player_0" else "player_0"
    
    if obs["teams"][player]["metal"] == 0:
        return dict()
    
    opp_factory_influence = np.ones((48, 48))
    if obs["factories"][opp_player] != {}:
        opp_fac = np.zeros((48, 48))
        for unit_id in obs["factories"][opp_player]:
            x, y = obs["factories"][opp_player][unit_id]["pos"]
            opp_fac[x-1:x+2, y-1:y+2] = 1
        opp_factory_influence = _clip(_expand_round(opp_fac, 3))
        opp_factory_influence = _normalise(_expand_round(opp_factory_influence, 7))
        opp_factory_influence = 1 - opp_factory_influence

    valid_mask = obs["board"]["valid_spawns_mask"]
    ice = obs["board"]["ice"]
    ice_mask = _clip(_expand_constrained(ice, dist=2))
    valid_near_ice_mask = ice_mask * valid_mask
    
    rubble_inv_norm = 1 - _normalise(obs["board"]["rubble"])
    rubble_mask = _normalise(_expand(rubble_inv_norm, 7))

    mask = rubble_mask * valid_near_ice_mask * valid_mask * opp_factory_influence 
    pos = np.unravel_index(np.argmax(mask), (48, 48))
    
    potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
    potential_spawns_set = set(potential_spawns)

    if not pos in potential_spawns_set:
        pos = potential_spawns[np.random.randint(0, len(potential_spawns))]
    
    assert pos in potential_spawns_set, (pos, rubble_mask.max(), valid_near_ice_mask.max())

    res = obs["teams"][player]["metal"]
    return dict(spawn=pos, metal=res, water=res)


def _expand_constrained(map, dist):
    mapping = np.zeros(np.array(map.shape) + 2 * (dist + 1))
    mapping[dist+1:-(dist+1), dist+1:-(dist+1)] = map
    for i in np.arange(-dist, dist+1):
        mapping[dist+1+i:-(dist+1)+i,dist+1:-(dist+1)] += map
    for j in np.arange(-dist, dist+1):
        mapping[dist+1:-(dist+1),dist+1+j:-(dist+1)+j] += map
    return mapping[dist+1:-(dist+1), dist+1:-(dist+1)]


def _expand_round(map, dist):
    mapping = np.zeros(np.array(map.shape) + 2 * (dist + 1))
    mapping[dist+1:-(dist+1), dist+1:-(dist+1)] = map
    for i in np.arange(-dist, dist+1):
        for j in np.arange(-dist, dist+1):
            if np.abs(i) + np.abs(j) > dist: continue
            mapping[dist+1+i:-(dist+1)+i,dist+1+j:-(dist+1)+j] += map
    return mapping[dist+1:-(dist+1), dist+1:-(dist+1)]


def _expand(map, dist):
    mapping = np.zeros(np.array(map.shape) + 2 * (dist + 1))
    mapping[dist+1:-(dist+1), dist+1:-(dist+1)] = map
    for i in np.arange(-dist, dist+1):
        for j in np.arange(-dist, dist+1):
            mapping[dist+1+i:-(dist+1)+i,dist+1+j:-(dist+1)+j] += map
    return mapping[dist+1:-(dist+1), dist+1:-(dist+1)]

def _clip(map, val_min=0, val_max=1):
    return np.clip(map, val_min, val_max)

def _normalise(map):
    return map / map.max()

def place_near_random_ice(player, obs):
    """
    This policy will place a single factory with all the starting resources
    near a random ice tile
    """
    if obs["teams"][player]["metal"] == 0:
        return dict()
    
    potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
    potential_spawns_set = set(potential_spawns)
    done_search = False

    rubble_inv_norm = 1 - _normalise(obs["board"]["rubble"])
    rubble_mask = _normalise(_expand(rubble_inv_norm, 7))

    # simple numpy trick to find locations adjacent to ice tiles.
    ice_diff = np.diff(obs["board"]["ice"])
    pot_ice_spots = np.argwhere(ice_diff == 1)
    if len(pot_ice_spots) == 0:
        pot_ice_spots = potential_spawns
    
    # pick a random ice spot and search around it for spawnable locations.
    trials = 5
    while trials > 0:
        
        pos_idx = np.random.randint(0, len(pot_ice_spots))
        
        ## MARC - start
        vals = rubble_mask[pot_ice_spots[pos_idx]]
        best_pos = pot_ice_spots[pos_idx]
        for xy in pot_ice_spots:
            test_val = rubble_mask[xy]
            if test_val > vals:
                vals = test_val
                best_pos = xy
        
        pos = best_pos
        ## MARC - end

        #pos = pot_ice_spots[pos_idx]
        area = 3
        for x in range(area):
            for y in range(area):
                check_pos = [pos[0] + x - area // 2, pos[1] + y - area // 2]
                if tuple(check_pos) in potential_spawns_set:
                    done_search = True
                    pos = check_pos
                    break
            if done_search:
                break
        if done_search:
            break
        trials -= 1
    
    if not done_search:
        spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
        pos = spawn_loc
    
    # this will spawn a factory at pos and with all the starting metal and water
    metal = obs["teams"][player]["metal"]
    return dict(spawn=pos, metal=metal, water=metal)
