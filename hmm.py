import torch
class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = {state: i for i, state in enumerate(self.states)}
        self.emissions_dict = {emission: i for i, emission in enumerate(self.emissions)}

    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            Porbability of the hidden state at time t given an obeservation sequence 
        """
        T = len(seq)
        delta = torch.zeros((self.N, T))
        psi = torch.zeros((self.N, T), dtype=torch.long)
        delta[:, 0] = torch.log(torch.tensor(self.pi)) + torch.log(self.B[:, self.emissions_dict[seq[0]]])
        for time_index in range(1, T):
            for state_index in range(self.N):
                trans_prob = delta[:, time_index - 1] + torch.log(self.A[:, state_index])
                max_trans_prob, max_state = torch.max(trans_prob, dim=0)
                delta[state_index, time_index] = max_trans_prob + torch.log(self.B[state_index, self.emissions_dict[seq[time_index]]])
                psi[state_index, time_index] = max_state.item()

        result_states = [torch.argmax(delta[:, -1]).item()]
        for time_index in range(T-1, 0, -1):
            result_states.insert(0, psi[result_states[0], time_index].item())

        result_states = [self.states[i] for i in result_states]
        return result_states
    

        
