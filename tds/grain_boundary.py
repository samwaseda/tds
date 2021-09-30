import numpy as np

def get_lattice_parameter(project):
    return get_bulk(project=project).get_structure().cell[0,0]

def get_potential():
    return '1995--Angelo-J-E--Ni-Al-H--LAMMPS--ipr1'

def get_bulk(project):
    lmp = project.create.job.Lammps('bulk')
    if lmp.status.initialized:
        lmp.structure = project.create.structure.bulk('Ni', cubic=True)
        lmp.potential = get_potential()
        lmp.calc_minimize(pressure=0)
        lmp.run()
    return lmp

def get_energy(element, project, repeat=4):
    lmp_Ni = get_bulk(project=project)
    if element == 'Ni':
        return lmp_Ni['output/generic/energy_pot'][-1]/len(lmp_Ni.structure)
    elif element == 'H':
        lmp = project.create.job.Lammps('bulk_H')
        if lmp.status.initialized:
            lmp.potential = get_potential()
            lmp.structure = lmp_Ni.get_structure().repeat(repeat)
            a_0 = lmp_Ni.get_structure().cell[0,0]
            lmp.structure += project.create.structure.atoms(positions=[[0, 0, 0.5*a_0]], elements=['H'])
            lmp.calc_minimize()
            lmp.run()
        return lmp.output.energy_pot[-1]-len(lmp.structure.select_index('Ni'))*get_energy('Ni', project=project)
    else:
        raise ValueError('element not recognized')

class GrainBoundary:
    def __init__(self, project, axis=[1, 0, 0], sigma=5, plane=[0, 1, 3], temperature=0, repeat=1):
        self._energy_per_atom_Ni = None
        self._energy_lst = []
        self._structure_lst = []
        self.project = project
        self.axis = axis
        self.sigma = sigma
        self.plane = plane
        self.temperature = temperature
        self.repeat = repeat

    @property
    def energy_per_atom_Ni(self):
        if self._energy_per_atom_Ni is None:
            self._energy_per_atom_Ni = get_energy('Ni', project=self.project)
        return self._energy_per_atom_Ni

    @property
    def gb_energy(self):
        if len(self._energy_lst)==0:
            self.run()
        return self._energy_lst.min()

    @property
    def grain_boundary(self):
        if len(self._structure_lst)==0:
            self.run()
        return self._structure_lst[self._energy_lst.argmin()]

    def run(self):
        lmp_bulk = get_bulk(project=self.project)
        for i in range(2):
            for j in range(2):
                gb = self.project.create.structure.aimsgb.build(
                    self.axis, self.sigma, self.plane, lmp_bulk.structure, delete_layer='{0}b{1}t{0}b{1}t'.format(i, j),
                    uc_a=self.repeat, uc_b=self.repeat
                )
                job_name = 'lmp_{}_{}_{}_{}_{}_{}_{}'.format(self.repeat, self.axis, self.sigma, self.plane, self.temperature, i, j)
                job_name = job_name.replace(',', 'c').replace('[', '').replace(']', '').replace(' ', '')
                lmp = self.project.create.job.Lammps(job_name)
                if lmp.status.initialized:
                    lmp.structure = gb
                    lmp.potential = get_potential()
                    if self.temperature > 0:
                        lmp.calc_md(temperature=self.temperature, pressure=0, n_ionic_steps=10000)
                    else:
                        lmp.calc_minimize(pressure=0)
                    lmp.run()
                E = lmp.output.energy_pot[-1]-self.energy_per_atom_Ni*len(gb)
                cell = lmp.output.cells[-1].diagonal()
                self._energy_lst.append(E/cell.prod()*np.max(cell)/2)
                self._structure_lst.append(lmp.get_structure())
        self._energy_lst = np.asarray(self._energy_lst)
