import numpy as np
from tqdm import tqdm


class KMC:
    """
    Class to do kinetic Monte Carlo calculations for H diffusion.
    """
    def __init__(
        self,
        binding_energy,
        pairs,
        diffusion_barriers,
        vectors,
        break_condition,
        attempt_frequency=1.0e13,
        number_of_steps=1000,
        heating_rate=None,
        **kwargs,
    ):
        """
        Args:
            binding_energy ((n,)-array): list of binding energies (for each site)
            pairs ((n,2)-array): list of pair indices
            diffusion_barriers ((n,2)-array): diffusion barriers (forward and backward)
            vectors ((n,3)-array): displacement vectors
            break condition (function): condition to return when to break KMC
            attempt_frequency (float): attempt frequency
            number_of_steps (int): number of KMC calculations to perform
            heating_rate (float): heating rate
        """
        self.attempt_frequency = attempt_frequency
        self.number_of_steps = number_of_steps
        self.acc_time_lst = []
        self.total_displacement_lst = []
        self.binding_energy = binding_energy-np.mean(binding_energy)
        self._set_environment(pairs, diffusion_barriers, vectors)
        self.heating_rate = heating_rate
        self.break_condition = break_condition

    def _set_environment(self, pairs, diffusion_barriers, vectors):
        self.indices_lst = []
        self.energies_lst = []
        self.vectors_lst = []
        for i in tqdm(np.unique(pairs)):
            self.indices_lst.append(pairs[pairs[:, 0] == i, 1])
            self.energies_lst.append(diffusion_barriers[pairs[:, 0] == i])
            self.vectors_lst.append(vectors[pairs[:, 0] == i])

    def run(self, temperature, charging_temperature=300):
        kB = 8.617e-5
        occ_enhancement = np.exp(-self.binding_energy/(kB*charging_temperature))
        occ_enhancement = np.cumsum(occ_enhancement)/occ_enhancement.sum()
        for i in tqdm(range(self.number_of_steps)):
            kBT = kB*temperature
            acc_time = 0
            displacement = np.zeros(3)
            index = np.sum(np.random.random() > occ_enhancement)
            while True:
                kappa = self.attempt_frequency*np.exp(-self.energies_lst[index]/kBT)
                index_jump = np.sum(np.random.random() > np.cumsum(kappa)/kappa.sum())
                dt = -np.log(np.random.random())/kappa.sum()
                acc_time += dt
                dr = self.vectors_lst[index][index_jump]
                displacement += dr
                if self.break_condition(displacement):
                    break
                index = self.indices_lst[index][index_jump]
                if self.heating_rate is not None:
                    kBT += kB*self.heating_rate*dt
            self.acc_time_lst.append(acc_time)
