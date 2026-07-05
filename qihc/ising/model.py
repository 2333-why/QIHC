import numpy as np
import matplotlib.pyplot as plt

class IsingModel:

    def __init__(self, size, Weight = None, Field = None):
        """
        Initialize the Ising model
        :param size: size of the Ising model
        :param Weight: coupling matrix between P-bits
        :param Field: external magnetic field
        """
        self.size = size    
        
        if Weight is None:
            self.Weight = self.generate_random_Weight()
        else:
            if Weight.shape != (self.size, self.size):
                raise ValueError("Weight matrix has wrong size")
            self.Weight = Weight
        
        if Field is None:
            self.Field = np.zeros(self.size)
        else:
            if Field.shape == (self.size, ):
                pass
            elif Field.shape == (self.size, 1):
                Field = Field.flatten()
            else:
                raise ValueError("Field has wrong size")
            self.Field = Field
        
        # inital the p-bit states randomly (-1 or 1) (0/1 is also acceptable)
        self.State = np.random.choice([-1, 1], self.size)

    def generate_random_Weight(self):
        """
        Generate a random symmetric coupling matrix for the Ising model.
        The matrix will have values between -1 and 1 and no self-coupling (diagonal = 0).
        
        :return: Symmetric weight matrix with no self-coupling.
        """
        # Generate a random matrix with values between -1 and 1
        Weight_matrix = np.random.uniform(-1, 1, (self.size, self.size))
        
        # Ensure symmetry by averaging the matrix and its transpose (H[i][j] = H[j][i])
        Weight_matrix = np.tril(Weight_matrix, -1) + np.tril(Weight_matrix, -1).T
        
        # Set diagonal to 0 to avoid self-coupling
        np.fill_diagonal(Weight_matrix, 0)
        
        return Weight_matrix


    def calculate_Derivative(self, state=None):
        """
        Calculate the derivative of the energy with respect to the state.
        The derivative corresponds to the force acting on each P-bit
        due to its interactions with other P-bits and the external field.
        
        :param state: A specific state to calculate the derivative for. If None, uses the current state.
        :return: The derivative vector representing the force on each P-bit.
        """
        if state is None:
            state = self.State
        
        # Calculate the interaction contribution (weighted sum of neighboring states)
        interaction_derivative = -np.dot(self.Weight, state)
        
        # Add the contribution from the external magnetic field
        total_derivative = interaction_derivative - self.Field
        return total_derivative
    
    def calculate_energy(self):
        """
        Calculate the total energy of the Ising model.
        The energy is computed as the sum of the interaction energy 
        (based on the coupling matrix) and the external field energy.
        """
        interaction_energy = -0.5 * np.dot(self.State, np.dot(self.Weight, self.State))
        field_energy = -np.dot(self.Field, self.State)
        
        total_energy = interaction_energy + field_energy
        return total_energy
    
    def sigmoid(self, x, beta=1, mu=0): 
        """
        Sigmoid function to calculate the switching probability for P-bits.
        The function maps input 'x' (typically voltage) to a probability 
        between 0 and 1 using the sigmoid curve.

        :param x: Input value (voltage or related quantity) that influences the P-bit state.
        :param beta: Temperature-related parameter controlling the steepness of the curve.
                    A higher value of beta makes the curve steeper.
        :param mu: Offset (shift) that adjusts the center of the curve. Default is 0.
        :return: A probability value between 0 and 1 representing the P-bit switching probability.
        """
        z = np.clip(-beta * (x - mu), -500.0, 500.0)
        return 1 / (1 + np.exp(z))

    def inverse_sigmoid(self, y, beta, mu=0): 
        """
        Inverse sigmoid function with a translational shift.
        This function returns the input value (x) that corresponds to 
        the given probability (y) from the sigmoid curve.
        
        :param y: The output probability from the sigmoid function (between 0 and 1).
        :param beta: The steepness parameter of the sigmoid function (controls the slope).
        :param mu: The shift parameter that translates the sigmoid curve (default is 0).
        :return: The input value corresponding to the probability y.
        """
        if y <= 0 or y >= 1:
            raise ValueError("Input probability y must be between 0 and 1 (exclusive).")

        return -1 / beta * np.log(1 / y - 1) + mu

    
    def update_State(self, beta):
        """
        Update the state of the P-bits based on the calculated derivative and
        the sigmoid probability curve. Each P-bit is updated according to its 
        associated probability of switching, which is calculated using the 
        sigmoid function.

        :param beta: Temperature-related parameter that controls the steepness 
                    of the sigmoid curve (affects the switching probability).
        """
        # Calculate the energy derivative for the current state of all P-bits
        Derivative = self.calculate_Derivative(self.State)

        energy_diff = 2*Derivative*self.State
        
        # Calculate the switching probabilities using the sigmoid function
        Sigmoid = self.sigmoid(energy_diff, beta)
        
        # Update the state of each P-bit based on its influence to the total energy (randomly/exploring)
        random_numbers = np.random.rand(self.size)
        # if random number < Sigmoid, no flip itself, else flip
        for i in range(self.size):
            if random_numbers[i] < Sigmoid[i] or energy_diff[i] < 0:
                self.State[i] = -self.State[i]
        # Alternative implementation:

        # self.State = (random_numbers >= Sigmoid).astype(int)
        # self.State[self.State == 0] = -1  # Convert 0s to -1s

    
    def energy_landscape(self):
        """
        Calculate the energy landscape of the Ising model, which involves 
        computing the energy for every possible state of the P-bits.
        
        :return: state_list (all possible states), energy_list (energies of those states)
        """
        num_states = 2**self.size
        state_list = np.zeros((num_states, self.size), dtype=int)
        energy_list = np.zeros(num_states)

        # Precompute all possible states (binary representations)
        for i in range(num_states):
            state_list[i] = np.array([int(x) for x in np.binary_repr(i, width=self.size)])

        # Calculate energies for each state
        for i in range(num_states):
            state = state_list[i]
            energy_list[i] = -0.5 * np.dot(state, np.dot(self.Weight, state)) - np.dot(self.Field, state)

        return state_list, energy_list
    
    # plotting
    def plot_Weight(self, real_index=True):
        plt.figure()
        plt.imshow(self.Weight, cmap='viridis', interpolation='nearest')
        plt.gca().xaxis.tick_top()
        if real_index:
            plt.xticks(np.arange(self.size), [str(i) for i in range(self.size)])
            plt.yticks(np.arange(self.size), [str(i) for i in range(self.size)])
        else:
            plt.xticks(np.arange(self.size), [str(i) for i in range(1, self.size + 1)])
            plt.yticks(np.arange(self.size), [str(i) for i in range(1, self.size + 1)])
        plt.rcParams.update({'font.size': 14})
        plt.grid()
        plt.colorbar()
        plt.show()
    
    def plot_energy_landscape(self):
        _, energy_list = self.energy_landscape()
        plt.figure()
        plt.rcParams.update({'font.size': 14})
        plt.plot(energy_list, 'o-')
        plt.xlabel('State')
        plt.ylabel('Energy')
        plt.show()

    def ising_simulated_annealing_Maxcut_Asyn(self, J, steps=1000, T_start=10, T_end=0.1, k=1):
        """
        Performs simulated annealing to solve an Ising model given an interaction matrix J.
        The goal is to find the configuration of P-bits that minimizes the energy.
        Using the voltage to fulfill the simulated annealing algorithm.
        
        Parameters:
            J (dict): The interaction matrix where the keys are tuples of nodes (i, j) and the values are interaction terms.
            steps (int): Number of simulated annealing steps.
            T_start (float): Starting temperature.
            T_end (float): Ending temperature.
            k (float): scaling factor for the acceptance probability. 
        Returns:
            spins, current_energy_list, temperature_list
        """
        # Initialize nodes from the interaction matrix (edges of the graph)
        nodes = list(set([i for edge in J for i in edge]))
        
        # loading coupling matrix J into Weight matrix
        for edge, value in J.items():
            i, j = edge
            self.Weight[i][j] = value
            self.Weight[j][i] = value  # Ensure symmetry

        # Initial energy
        current_energy = self.calculate_energy()
        T = T_start  # Starting temperature

        # data collection for analysis (optional)
        current_energy_list, temperature_list = [current_energy], [T_start]
        
        # Simulated Annealing process
        for step in range(steps):
            # Update the temperature according to the annealing schedule
            T = T_start * (T_end / T_start) ** (step / steps)
            # another way to update temperature: T*= alpha  (alpha: damping factor)
            
            # Randomly select a node and flip its spin
            i = np.random.choice(nodes)
            # self.State[i] = 1-self.State[i]  # Flip the spin
            self.State[i] = -self.State[i]  # Flip the spin
            
            # Calculate new energy after flipping the spin
            new_energy = self.calculate_energy()
            pro = self.sigmoid((current_energy - new_energy), beta=k/T)
            # print(new_energy)
            
            # Metropolis acceptance criterion
            if new_energy < current_energy or np.random.rand() < pro:
                current_energy = new_energy  # Accept the new configuration
                # self.State[i] = 1-self.State[i]
            else:
                self.State[i] = -self.State[i]

            # data collection for analysis (optional)
            current_energy_list.append(current_energy)
            temperature_list.append(T)
            
            spins = {node: self.State[node] for node in nodes}
        
        return spins, current_energy_list, temperature_list  # Return spins and the negative of the energy
    
    def ising_simulated_annealing_Maxcut_Syn(self, J, steps=1000, T_start=10, T_end=0.1, k=1):
        """
        Performs simulated annealing to solve an Ising model given an interaction matrix J.
        The goal is to find the configuration of P-bits that minimizes the energy.
        Using the voltage to fulfill the simulated annealing algorithm.
        
        Parameters:
            J (dict): The interaction matrix where the keys are tuples of nodes (i, j) and the values are interaction terms.
            steps (int): Number of simulated annealing steps.
            T_start (float): Starting temperature.
            T_end (float): Ending temperature.
            k (float): scaling factor for the acceptance probability. 
        Returns:
            spins, current_energy_list, temperature_list
        """
        # Initialize nodes from the interaction matrix (edges of the graph)
        nodes = list(set([i for edge in J for i in edge]))
        
        # loading coupling matrix J into Weight matrix
        for edge, value in J.items():
            i, j = edge
            self.Weight[i][j] = value
            self.Weight[j][i] = value  # Ensure symmetry

        # Initial energy
        current_energy = self.calculate_energy()
        T = T_start  # Starting temperature

        # data collection for analysis (optional)
        current_energy_list, temperature_list = [current_energy], [T_start]
        
        # Simulated Annealing process
        for step in range(steps):
            # Update the temperature according to the annealing schedule
            T = T_start * (T_end / T_start) ** (step / steps)
            # another way to update temperature: T*= alpha  (alpha: damping factor)
            
            self.update_State(beta=k/T)  # Synchronous update of all spins

            # data collection for analysis (optional)
            current_energy_list.append(self.calculate_energy())
            temperature_list.append(T)
            
            spins = {node: self.State[node] for node in nodes}
        
        return spins, current_energy_list, temperature_list  # Return spins and the negative of the energy

    # ------------------------------------------------------------------
    # Shared helpers for Max-Cut / QUBO solvers
    # ------------------------------------------------------------------

    def _load_j_couplings(self, J):
        """Load interaction dict {(i,j): w} into Weight matrix; return sorted node list."""
        nodes = sorted(set(i for edge in J for i in edge))
        for edge, value in J.items():
            i, j = edge
            self.Weight[i][j] = value
            self.Weight[j][i] = value
        return nodes

    def _energy_of(self, state):
        """Hamiltonian energy for an arbitrary spin configuration (+/-1)."""
        return -0.5 * np.dot(state, np.dot(self.Weight, state)) - np.dot(self.Field, state)

    def _temperature_schedule(self, step, steps, T_start, T_end):
        if steps <= 1:
            return T_end
        return T_start * (T_end / T_start) ** (step / (steps - 1))

    def _gibbs_sequential(self, beta):
        """One full sweep of sequential Gibbs sampling (site updates in random order)."""
        for i in np.random.permutation(self.size):
            derivative = self.calculate_Derivative(self.State)
            energy_diff = 2 * derivative[i] * self.State[i]
            if energy_diff < 0 or np.random.rand() < self.sigmoid(energy_diff, beta):
                self.State[i] *= -1

    def _gibbs_parallel(self, beta):
        """One synchronous Gibbs sweep (same rule as update_State)."""
        self.update_State(beta)

    # ------------------------------------------------------------------
    # ④ Gibbs sampling
    # ------------------------------------------------------------------

    def gibbs_sampling_Maxcut(
        self,
        J,
        steps=1000,
        T_start=10.0,
        T_end=0.1,
        k=1.0,
        sequential=True,
    ):
        """
        Gibbs sampling with exponential cooling schedule for Max-Cut / Ising problems.

        Parameters
        ----------
        J : dict
            Interaction matrix {(i, j): weight}.
        steps : int
            Number of sampling sweeps.
        T_start, T_end : float
            Annealing temperature endpoints.
        k : float
            Boltzmann scaling factor (beta = k / T).
        sequential : bool
            True  -> sequential site updates (standard Gibbs).
            False -> synchronous / parallel Gibbs sweep.

        Returns
        -------
        spins : dict
            Final spin configuration.
        energy_trace : list
            Energy after each sweep.
        temperature_trace : list
            Temperature after each sweep.
        """
        nodes = self._load_j_couplings(J)
        energy_trace = [self._energy_of(self.State)]
        temperature_trace = [T_start]

        for step in range(steps):
            T = self._temperature_schedule(step, steps, T_start, T_end)
            beta = k / max(T, 1e-12)
            if sequential:
                self._gibbs_sequential(beta)
            else:
                self._gibbs_parallel(beta)
            energy_trace.append(self._energy_of(self.State))
            temperature_trace.append(T)

        spins = {node: int(self.State[node]) for node in nodes}
        return spins, energy_trace, temperature_trace

    # ------------------------------------------------------------------
    # ⑤ Parallel Tempering (Replica Exchange)
    # ------------------------------------------------------------------

    def parallel_tempering_Maxcut(
        self,
        J,
        steps=1000,
        temperatures=None,
        n_replicas=8,
        T_start=10.0,
        T_end=0.1,
        swap_interval=10,
        k=1.0,
        sequential=True,
    ):
        """
        Parallel tempering / replica-exchange MCMC for Ising energy minimization.

        Maintains replicas at different temperatures and periodically attempts
        adjacent swaps using the Metropolis criterion.

        Parameters
        ----------
        J : dict
            Interaction matrix.
        steps : int
            Total MCMC sweeps (per replica, per step).
        temperatures : array-like, optional
            Explicit temperature ladder (cold -> hot). If None, uses geometric
            spacing between T_end and T_start.
        n_replicas : int
            Number of temperature replicas when temperatures is None.
        T_start, T_end : float
            Hot / cold temperature endpoints for automatic ladder.
        swap_interval : int
            Attempt replica swaps every this many sweeps.
        k : float
            Boltzmann scaling factor.
        sequential : bool
            Gibbs update mode for each replica.

        Returns
        -------
        spins : dict
            Best (lowest-energy) configuration found on the coldest replica.
        energy_trace : list
            Best energy across replicas after each sweep.
        temperatures : ndarray
            Temperature ladder used.
        """
        nodes = self._load_j_couplings(J)

        if temperatures is None:
            temperatures = np.geomspace(max(T_end, 1e-6), T_start, n_replicas)
        else:
            temperatures = np.asarray(temperatures, dtype=float)

        n_rep = len(temperatures)
        replicas = np.random.choice([-1, 1], size=(n_rep, self.size))
        energies = np.array([self._energy_of(s) for s in replicas])

        best_idx = int(np.argmin(energies))
        best_state = replicas[best_idx].copy()
        best_energy = float(energies[best_idx])
        energy_trace = [best_energy]

        for step in range(steps):
            for r in range(n_rep):
                self.State = replicas[r].copy()
                beta = k / max(temperatures[r], 1e-12)
                if sequential:
                    self._gibbs_sequential(beta)
                else:
                    self._gibbs_parallel(beta)
                replicas[r] = self.State.copy()
                energies[r] = self._energy_of(self.State)

            if swap_interval > 0 and step > 0 and (step % swap_interval == 0):
                for r in range(n_rep - 1):
                    beta_i = k / max(temperatures[r], 1e-12)
                    beta_j = k / max(temperatures[r + 1], 1e-12)
                    delta = (beta_i - beta_j) * (energies[r] - energies[r + 1])
                    if delta >= 0 or np.random.rand() < np.exp(delta):
                        replicas[r], replicas[r + 1] = (
                            replicas[r + 1].copy(),
                            replicas[r].copy(),
                        )
                        energies[r], energies[r + 1] = energies[r + 1], energies[r]

            cold_idx = int(np.argmin(temperatures))
            if energies[cold_idx] < best_energy:
                best_energy = float(energies[cold_idx])
                best_state = replicas[cold_idx].copy()
            energy_trace.append(best_energy)

        self.State = best_state.copy()
        spins = {node: int(self.State[node]) for node in nodes}
        return spins, energy_trace, temperatures

    # ------------------------------------------------------------------
    # ⑥ Simulated Quantum Annealing (SQA, Suzuki–Trotter decomposition)
    # ------------------------------------------------------------------

    def _sqa_transverse_coupling(self, beta, gamma, m_slices):
        """Transverse coupling J_perp for Trotter decomposition."""
        x = beta * gamma / m_slices
        x = np.clip(x, 1e-8, 50.0)
        return -0.5 * np.log(np.tanh(x))

    def _sqa_slice_energy(self, slices, m_idx):
        """Ising energy of one Trotter slice."""
        s = slices[:, m_idx]
        return self._energy_of(s)

    def _sqa_total_energy(self, slices, J_perp):
        """Total SQA energy = sum of slice Ising energies + transverse couplings."""
        m_slices = slices.shape[1]
        energy = sum(self._sqa_slice_energy(slices, m) for m in range(m_slices))
        for m in range(m_slices):
            m_next = (m + 1) % m_slices
            energy -= J_perp * np.sum(slices[:, m] * slices[:, m_next])
        return energy

    def _sqa_flip_delta(self, slices, site, m_idx, J_perp):
        """Energy change when flipping spin (site, m_idx) in the Trotter lattice."""
        s = slices[:, m_idx]
        local_field = np.dot(self.Weight[site], s) + self.Field[site]
        delta_ising = 2 * s[site] * local_field

        m_prev = (m_idx - 1) % slices.shape[1]
        m_next = (m_idx + 1) % slices.shape[1]
        delta_transverse = 2 * slices[site, m_idx] * (
            slices[site, m_prev] + slices[site, m_next]
        ) * J_perp
        return delta_ising + delta_transverse

    def simulated_quantum_annealing_Maxcut(
        self,
        J,
        steps=1000,
        T_start=10.0,
        T_end=0.1,
        Gamma_start=5.0,
        Gamma_end=0.01,
        m_slices=8,
        k=1.0,
    ):
        """
        Simulated Quantum Annealing via Suzuki–Trotter decomposition.

        Maps the transverse-field Ising model to a classical (spatial x Trotter)
        Ising lattice and performs Metropolis updates while annealing temperature
        T and transverse field Gamma.

        Parameters
        ----------
        J : dict
            Interaction matrix.
        steps : int
            Annealing steps.
        T_start, T_end : float
            Thermal annealing schedule endpoints.
        Gamma_start, Gamma_end : float
            Transverse-field annealing endpoints (high -> low).
        m_slices : int
            Number of Trotter slices (path-integral discretization).
        k : float
            Boltzmann scaling factor.

        Returns
        -------
        spins : dict
            Spin configuration from the lowest-energy Trotter slice.
        energy_trace : list
            Best classical Ising energy across slices after each step.
        gamma_trace : list
            Transverse field Gamma after each step.
        """
        nodes = self._load_j_couplings(J)
        slices = np.random.choice([-1, 1], size=(self.size, m_slices))

        energy_trace = []
        gamma_trace = []

        best_energy = np.inf
        best_slice_idx = 0

        for step in range(steps):
            T = self._temperature_schedule(step, steps, T_start, T_end)
            Gamma = self._temperature_schedule(step, steps, Gamma_start, Gamma_end)
            beta = k / max(T, 1e-12)
            J_perp = self._sqa_transverse_coupling(beta, Gamma, m_slices)

            for m_idx in range(m_slices):
                for site in np.random.permutation(self.size):
                    delta = self._sqa_flip_delta(slices, site, m_idx, J_perp)
                    if delta < 0 or np.random.rand() < np.exp(-beta * delta):
                        slices[site, m_idx] *= -1

            slice_energies = [self._sqa_slice_energy(slices, m) for m in range(m_slices)]
            step_best = int(np.argmin(slice_energies))
            step_best_energy = slice_energies[step_best]
            if step_best_energy < best_energy:
                best_energy = step_best_energy
                best_slice_idx = step_best

            energy_trace.append(step_best_energy)
            gamma_trace.append(Gamma)

        self.State = slices[:, best_slice_idx].copy()
        spins = {node: int(self.State[node]) for node in nodes}
        return spins, energy_trace, gamma_trace