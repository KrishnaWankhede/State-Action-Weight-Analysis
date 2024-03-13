import math
import numpy as np

# Parsing Functions

def parse_file(file_path):
    with open(file_path, 'r') as file:
        file_type = next(file).strip()  # Reading file type line (unused)
        header = next(file).strip().split()  # Reading header line
        data = [line.strip().split() for line in file]  # Reading actual data lines
    return header, data

def parse_state_weights(file_path):
    _, data = parse_file(file_path)  # Header is unused here
    return {row[0].strip('"'): float(row[1]) for row in data}

def parse_state_action_state_weights(file_path):
    header, data = parse_file(file_path)
    # print(header)
    default_weight = float(header[3]) / float(header[0]) # Normalizing default weight
    # print(default_weight)
    weights = {}
    actions = set()  # Create a set to store unique action names
    for row in data:
        state, action, next_state, weight = row
        weights.setdefault(state.strip('"'), {}).setdefault(action.strip('"'), {})[next_state.strip('"')] = float(weight)
        actions.add(action.strip('"'))  # Add the action name to the set
    actions = list(actions)  # Convert the set to a list of action names
    return weights, default_weight, actions

def parse_state_observation_weights(file_path):
    header, data = parse_file(file_path)
    default_weight = float(header[3]) / float(header[0])  # Normalizing default weight
    weights = {}
    for row in data:
        state, observation, weight = row
        weights.setdefault(state.strip('"'), {}).setdefault(observation.strip('"'), float(weight))
    return weights, default_weight

def parse_observation_action_sequence(file_path):
    _, data = parse_file(file_path)
    observation_action_sequence = []
    for row in data:
        if len(row) >= 2:
            # Both observation and action are present
            observation, action = row[0].strip('"'), row[1].strip('"')
        elif len(row) == 1:
            # Only observation is present, default action can be set to None or a default value
            observation = row[0].strip('"')
            action = None  # or a default value, e.g., "Unknown"
        else:
            # Skip empty lines
            continue
        observation_action_sequence.append((observation, action))
    return observation_action_sequence

# Normalization Functions

def normalize_weights(weights, total):
    return {key: value / total for key, value in weights.items()}

def normalize_state_weights(state_weights):
    total_weight = sum(state_weights.values())
    return normalize_weights(state_weights, total_weight) if total_weight else state_weights

def normalize_sas_weights(sas_weights, default_weight, states, actions):
    for state in states:
        for action in actions:
            sas_weights.setdefault(state, {}).setdefault(action, {})
            total_weight = sum(sas_weights[state][action].values()) + default_weight * (len(states) - len(sas_weights[state][action]))
            for next_state in states:
                sas_weights[state][action].setdefault(next_state, default_weight)
                sas_weights[state][action][next_state] /= total_weight
    return sas_weights

def normalize_so_weights(so_weights, default_weight, states, observations):
    for state in states:
        so_weights.setdefault(state, {})
        total_weight = sum(so_weights[state].values()) + default_weight * (len(observations) - len(so_weights[state]))
        for observation in observations:
            so_weights[state].setdefault(observation, default_weight)
            so_weights[state][observation] /= total_weight
    return so_weights

# Viterbi Algorithm

def viterbi(observation_actions, state_weights, state_action_state_weights, state_observation_weights, default_state_action_state_weight, default_state_observation_weight):
    states = list(state_weights.keys())
    # Initialize the trellis
    trellis = np.zeros((len(states), len(observation_actions)))
    backpointers = np.zeros((len(states), len(observation_actions)), dtype=int)

    # Initialize the first column of the trellis
    for state_index, state in enumerate(states):
        so_weight = state_observation_weights[state].get(observation_actions[0][0], default_state_observation_weight)
        trellis[state_index, 0] = state_weights[state] * so_weight

    # Fill in the rest of the trellis
    for observation_index, (observation, action) in enumerate(observation_actions[1:], start=1):
        for state_index, state in enumerate(states):
            max_weight = -math.inf
            max_weight_index = -1
            for prev_state_index, prev_state in enumerate(states):
                sas_weight = state_action_state_weights[prev_state].get(action, {}).get(state, default_state_action_state_weight)
                so_weight = state_observation_weights[state].get(observation, default_state_observation_weight)
                weight = trellis[prev_state_index, observation_index - 1] * sas_weight * so_weight
                if weight > max_weight:
                    max_weight = weight
                    max_weight_index = prev_state_index
            trellis[state_index, observation_index] = max_weight
            backpointers[state_index, observation_index] = max_weight_index
        
    # Find the best path
    best_path = []
    best_path_index = np.argmax(trellis[:, -1])
    best_path.append(states[best_path_index])
    for observation_index in range(len(observation_actions) - 1, 0, -1):
        best_path_index = backpointers[best_path_index, observation_index]
        best_path.append(states[best_path_index])
    best_path.reverse()
    return best_path


# Main Function

def main():
    # File paths
    state_weights_path = 'state_weights.txt'
    sas_weights_path = 'state_action_state_weights.txt'
    so_weights_path = 'state_observation_weights.txt'
    observation_actions_path = 'observation_actions.txt'

    # Parse files
    state_weights = parse_state_weights(state_weights_path)
    sas_weights, default_sas_weight, actions = parse_state_action_state_weights(sas_weights_path)
    so_weights, default_so_weight = parse_state_observation_weights(so_weights_path)
    observation_actions = parse_observation_action_sequence(observation_actions_path)

    # Prepare states and observations
    observations = [obs_action[0] for obs_action in observation_actions]  # Extract all observations

    # Normalize weights
    states = list(state_weights.keys())
    normalized_state_weights = normalize_state_weights(state_weights)
    normalized_sas_weights = normalize_sas_weights(sas_weights, default_sas_weight, states, actions)
    normalized_so_weights = normalize_so_weights(so_weights, default_so_weight, states, observations)


    # Viterbi algorithm
    most_likely_states = viterbi(observation_actions, normalized_state_weights, normalized_sas_weights, normalized_so_weights, default_sas_weight, default_so_weight)

    # Print the result
    print("Best State Sequence:", most_likely_states)
    write_output(most_likely_states)
# Output Function
def write_output(most_likely_states):
    with open('states.txt', 'w') as f:
        f.write("states\n")
        f.write(f"{len(most_likely_states)}\n")
        for state in most_likely_states:
            f.write(f'"{state}"\n')
if __name__ == "__main__":
    main()
