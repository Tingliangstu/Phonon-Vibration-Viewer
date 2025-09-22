#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualizing lattice vibration information from phonon dispersion for primitive atoms
Developed initially by Liang Ting in2021/12/18
Revised by Ying Penghua in 2022/2/50
"""

import numpy as np
import os
from ase.io.lammpsdata import read_lammps_data, write_lammps_data

def get_frequency_eigen_info(num_basis, eig_file='eigenvector.out', directory=None):
    if not directory:
        eig_path = os.path.join(os.getcwd(), eig_file)
    else:
        eig_path = os.path.join(directory, eig_file)

    eig_data_file = open(eig_path, 'r')
    data_lines = [line for line in eig_data_file.readlines() if line.strip()]
    eig_data_file.close()

    om2 = np.array([data_lines[0].split()[0:num_basis * 3]], dtype='float64')
    eigenvector = np.array([data_lines[1 + k].split()[0:num_basis * 3]
                                               for k in range(num_basis * 3)], dtype='float64')
    nu = np.sign(om2) * np.sqrt(abs(np.array(om2))) / (2 * np.pi)
    return nu, eigenvector

def position_plus_eigen(gamma_freq_points, nu, eigenvector, atom_num_in_box):
    if atom_num_in_box * 3 != np.size(eigenvector, 1):
        raise ValueError("The data dimension of the eigenvector is inconsistent with atomic number*3")
    print('************* Now the frequency is {0:10.6} THz, the visualization of the eigenvectors is at gamma point'
          '**************** '.format(nu[0][gamma_freq_points]))

    # reshape eigenvector
    eigenvector_x = eigenvector[gamma_freq_points][0:atom_num_in_box]
    eigenvector_y = eigenvector[gamma_freq_points][atom_num_in_box:atom_num_in_box*2]
    eigenvector_z = eigenvector[gamma_freq_points][atom_num_in_box*2:atom_num_in_box*3]
    eigenvector_xyz = np.c_[eigenvector_x, eigenvector_y, eigenvector_z]
    return eigenvector_xyz

if __name__ == "__main__":

    num_basis = int(input("Number of basis atoms: ")) #42
    gamma_freq_points = int(input("The nth frequency point at Gamma point: ")) #from 0 
    nu, eigenvector = get_frequency_eigen_info(num_basis)
    
    first_frame = read_lammps_data('lammps-data', style='atomic')
    positions_first_frame = first_frame.get_positions()
    atom_num_in_box = len(positions_first_frame)
    direct_cell = first_frame.cell[:]
    second_frame = read_lammps_data('lammps-data', style='atomic')
    positions_delta = position_plus_eigen(gamma_freq_points, nu, eigenvector, atom_num_in_box)
    positions_second_frame = positions_first_frame + positions_delta
    second_frame.set_positions(positions_second_frame)
    write_lammps_data("Phonon_{:0.6}THz_first_frame.data".format(nu[0][gamma_freq_points]), first_frame)
    write_lammps_data("Phonon_{:0.6}THz_second_frame.data".format(nu[0][gamma_freq_points]), second_frame)
    

    print('******************** All Done !!! *************************')