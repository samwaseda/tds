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

def get_gb(project, axis=[1, 0, 0], sigma=5, plane=[0, 1, 3], repeat=1):
    lmp_bulk = get_bulk(project=project)
    for i in range(2):
        for j in range(2):
            gb = project.create.structure.aimsgb.build(
                axis, sigma, plane, lmp_bulk.structure, delete_layer='{0}b{1}t{0}b{1}t'.format(i, j),
                uc_a=repeat, uc_b=repeat
            )
            job_name = 'lmp_{}_{}_{}_{}_{}_{}'.format(repeat, axis, sigma, plane, i, j)
            job_name = job_name.replace(',', 'c').replace('[', '').replace(']', '').replace(' ', '')
            lmp = project.create.job.Lammps(job_name)
            if lmp.status.initialized:
                lmp.structure = gb
                lmp.potential = get_potential()
                lmp.calc_minimize(pressure=0)
                lmp.run()
            E_current = lmp.output.energy_pot[-1]-lmp_bulk.output.energy_pot[-1]/4*len(gb)
            if i+j==0 or E_min > E_current:
                E_min = E_current
                structure = lmp.get_structure()
    return structure
