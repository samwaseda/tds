import numpy as np

class KMC:
    def __init__(
        self,
        binding_energy,
        pairs,
        diffusion_barriers,
        vectors,
        attempt_frequency=1.0e13,
        z_lim=structure.cell[2,2]/2,
        number_of_steps=1_000,
        **kwargs,
    ):
        self.attempt_frequency = attempt_frequency
        self.z_lim = z_lim
        self.number_of_steps = number_of_steps
        self.acc_time_lst = []
        self.total_displacement_lst = []
        self.binding_energy = binding_energy-np.mean(binding_energy)
        self.set_environment(pairs, diffusion_barriers, vectors)

    def set_environment(self, pairs, diffusion_barriers, vectors):
        self.indices_lst = []
        self.energies_lst = []
        self.vectors_lst = []
        for i in tqdm(np.unique(pairs)):
            self.indices_lst.append(pairs[pairs[:,0]==i, 1])
            self.energies_lst.append(diffusion_barriers[pairs[:,0]==i])
            self.vectors_lst.append(vectors[pairs[:,0]==i])

    def run(self, temperature, charging_temperature=300):
        kBT = 8.617e-5*temperature
        occ_enhancement = np.ones_like(self.binding_energy)
        occ_enhancement = np.cumsum(occ_enhancement)/occ_enhancement.sum()
        for i in tqdm(range(self.number_of_steps)):
            acc_time = 0
            z_displacement = 0
            index = np.sum(np.random.random()>occ_enhancement)
            while True:
                kappa = self.attempt_frequency*np.exp(-self.energies_lst[index]/kBT)
                index_jump = np.sum(np.random.random()>np.cumsum(kappa)/kappa.sum())
                dt = -np.log(np.random.random())/kappa.sum()
                acc_time += dt
                dr = self.vectors_lst[index][index_jump]
                z_displacement += dr[-1]
                if np.absolute(z_displacement) > self.z_lim:
                    break
                index = self.indices_lst[index][index_jump]
            self.acc_time_lst.append(acc_time)
