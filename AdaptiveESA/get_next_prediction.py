# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 12:13:08 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Automated Electrical Impedance Tomography for Graphene
Module: get_next_prediction.py
Dependancies: 
"""
import os
import numpy as np
import measurement_optimizer as measopt
import greit_rec_training_set as train
import h5py as h5
import cupy as cp
from pyeit.eit.fem_forallmeas import Forward
from pyeit.eit.fem_for_given_meas import Forward as Forward_given
from pyeit.eit.utils import eit_scan_lines

import matplotlib.pyplot as plt

from meshing import mesh

def save_small_Jacobian(save_filename, n_el=20, n_per_el=3):
    # number electrodes
    el_pos = np.arange(n_el * n_per_el)
    # create an object with the meshing characteristics to initialise a Forward object
    mesh_obj = mesh(n_el)
    ex_mat = train.orderedExMat(n_el=20)
    #ex_mat = train.generateExMat(ne=n_el)
    fwd = Forward_given(mesh_obj, el_pos, n_el)

    f, meas, new_ind = fwd.solve_eit(ex_mat=ex_mat, perm=fwd.tri_perm)
    
    #print(f)
    ind = np.arange(len(meas))
    np.random.shuffle(ind)
    pde_result = train.namedtuple("pde_result", ['jac', 'v', 'b_matrix'])

    f = pde_result(jac=f.jac[ind], v=f.v[ind], b_matrix=f.b_matrix[ind])
    meas = meas[ind]
    new_ind = new_ind[ind]
    h = h5.File(save_filename, 'w')

    try:
        h.create_dataset('jac', data=f.jac)
        h.create_dataset('v', data=f.v)
        h.create_dataset('b', data=f.b_matrix)
        h.create_dataset('meas', data=meas)
        h.create_dataset('new_ind', data=new_ind)
        h.create_dataset('p', data=mesh_obj['node'])
        h.create_dataset('t', data=mesh_obj['element'])
    except:
        TypeError('Error with saving files!')
    h.close()
    
def saveJacobian(save_filename, n_el=20, n_per_el=3):
    # number electrodes
    el_pos = np.arange(n_el * n_per_el)
    # create an object with the meshing characteristics to initialise a Forward object
    mesh_obj = mesh(n_el)
    fwd = Forward(mesh_obj, el_pos, n_el)
    ex_mat = train.generateExMat(ne=n_el)
    f, meas, new_ind = fwd.solve_eit(ex_mat=ex_mat, perm=fwd.tri_perm)
    
    #print(f)
    ind = np.arange(len(meas))
    np.random.shuffle(ind)
    pde_result = train.namedtuple("pde_result", ['jac', 'v', 'b_matrix'])

    f = pde_result(jac=f.jac[ind], v=f.v[ind], b_matrix=f.b_matrix[ind])
    meas = meas[ind]
    new_ind = new_ind[ind]
    h = h5.File(save_filename, 'w')

    try:
        h.create_dataset('jac', data=f.jac)
        h.create_dataset('v', data=f.v)
        h.create_dataset('b', data=f.b_matrix)
        h.create_dataset('meas', data=meas)
        h.create_dataset('new_ind', data=new_ind)
        h.create_dataset('p', data=mesh_obj['node'])
        h.create_dataset('t', data=mesh_obj['element'])
    except:
        TypeError('Error with saving files!')
    h.close()

def getNextPrediction(fileJac: str, measuring_electrodes: np.ndarray, voltages: np.ndarray, 
              num_returned: int=10, n_el: int=20, n_per_el: int=3, n_pix: int=64, pert: float=0.5, 
              p_influence: float=-10., p_rec: float=10.) -> np.ndarray:
    # extract const permittivity jacobian and voltage (& other)
    file = h5.File(fileJac, 'r')

    meas = file['meas'][()]
    new_ind = file['new_ind'][()]
    p = file['p'][()]
    t = file['t'][()]
    file.close()
    # initialise const permitivity and el_pos variables
    perm = np.ones(t.shape[0], dtype=np.float32)
    el_pos = np.arange(n_el * n_per_el).astype(np.int16)
    mesh_obj = {'element': t,
        'node':    p,
        'perm':    perm}
    # list all possible active/measuring electrode permutations of this measurement
    meas = cp.array(meas)
    # find their indices in the already calculated const. permitivity Jacobian (CPJ)
    measuring_electrodes = cp.array(measuring_electrodes)
    measurements_0 = cp.amin(measuring_electrodes[:, :2], axis=1)
    measurements_1 = cp.amax(measuring_electrodes[:, :2], axis=1)
    measurements_2 = cp.amin(measuring_electrodes[:, 2:], axis=1)
    measurements_3 = cp.amax(measuring_electrodes[:, 2:], axis=1)
    measuring_electrodes = cp.empty((len(measuring_electrodes), 4))
    measuring_electrodes[:, 0] = measurements_0
    measuring_electrodes[:, 1] = measurements_1
    measuring_electrodes[:, 2] = measurements_2
    measuring_electrodes[:, 3] = measurements_3
    index = (cp.sum(cp.equal(measuring_electrodes[:, None, :], meas[None, :, :]), axis=2) == 4)
    index = cp.where(index)
    #print(index)
    ind = cp.unique(index[1])
    #print(ind)
    i = cp.asnumpy(ind)
    j = index[0]
    mask = np.zeros(len(meas), dtype=int)
    mask[i] = 1
    mask = mask.astype(bool)
    # take a slice of Jacobian, voltage readings and B matrix (the one corresponding to the performed measurements)
    file = h5.File(fileJac, 'r')
    jac = file['jac'][mask, :][()]
    v = file['v'][mask][()]
    b = file['b'][mask, :][()]
    file.close()
    # put them in the form desired by the GREIT function
    pde_result = train.namedtuple("pde_result", ['jac', 'v', 'b_matrix'])
    f = pde_result(jac=jac,
           v=v,
           b_matrix=b)
    
    # now we can use the real voltage readings and the GREIT algorithm to reconstruct
    greit = train.greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=(meas[index[1], :2]), step=None)
    greit.setup(p=0.2, lamb=0.01, n=n_pix)
    h_mat = greit.H
    reconstruction = greit.solve(voltages, f.v).reshape(n_pix, n_pix)
    # fix_electrodes_multiple is in meshing.py
    _, el_coords = train.fix_electrodes_multiple(centre=None, edgeX=0.1, edgeY=0.1, a=2, b=2, ppl=n_el, el_width=0.2, num_per_el=3)
    # find the distances between each existing electrode pair and the pixels lying on the liine that connects them
    pixel_indices, voltage_all_possible = measopt.find_all_distances(reconstruction, h_mat, el_coords, n_el, cutoff=0.8)
    # call function get_total_map that generates the influence map, the gradient map and the log-reconstruction
    total_map, grad_mat, rec_log = np.abs(measopt.get_total_map(reconstruction, voltages, h_mat, pert=pert, p_influence=p_influence, p_rec=p_rec))
    # get the indices of the total map along the lines connecting each possible electrode pair
    total_maps_along_lines = total_map[None] * pixel_indices
    # find how close each connecting line passes to the boundary of an anomaly (where gradient supposed to be higher)
    proximity_to_boundary = np.sum(total_maps_along_lines, axis=(1, 2)) / np.sum(pixel_indices, axis=(1, 2))
    # rate the possible src-sink pairs by their proximity to existing anomalies
    proposed_ex_line = voltage_all_possible[np.argsort(proximity_to_boundary)[::-1]][:num_returned]

    number_of_voltages = 10
    # generate the voltage measuring electrodes for this current driver pair
    proposed_voltage_pairs = measopt.findNextVoltagePair(proposed_ex_line[0], fileJac, total_map, number_of_voltages, 0, npix=n_pix, cutoff=0.97)
    return proposed_ex_line, proposed_voltage_pairs, reconstruction, total_map

    # for new_ex_line in proposed_ex_line:
    #     index_source = new_ex_line[0] == meas[:, 0]
    #     index_sink = new_ex_line[1] == meas[:, 1]
    #     index_current = index_source * index_sink
    #     get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    #     print("Number of matching elements:", get_indexes(True,index_current))
    #     print("Check")
    
def volt_matrix_generator(ex_mat: np.ndarray, n_el: int, volt_pair_mode: str='adj' ):
    
    if (volt_pair_mode == 'adj'):
        print("\nAdjacent voltage pairs mode")
        volt_mat, new_ex_mat, new_ind = measopt.voltMeterwStep(n_el, cp.asarray(ex_mat), step_arr=1, parser=None)
        ind = cp.asnumpy(new_ind)
        volt_mat = cp.asnumpy(volt_mat)
        
    elif (volt_pair_mode == 'opp'):
        print("\nOpposite voltage pairs mode")
        volt_mat, new_ex_mat, new_ind = measopt.voltMeterwStep(n_el, cp.asarray(ex_mat), step_arr=n_el//2, parser=None)
        ind = cp.asnumpy(new_ind)
        volt_mat = cp.asnumpy(volt_mat)
        
    elif (volt_pair_mode == 'all'):
        print("\n All valid voltage pairs mode")
        all_pairs = cp.asnumpy(measopt.volt_mat_all(n_el))
        volt_mat = []
        for current_pair in ex_mat:
            index_source_1 = current_pair[0] == all_pairs[:, 0]
            index_source_2 = current_pair[0] == all_pairs[:, 1]
            index_sink_1 = current_pair[1] == all_pairs[:, 0]
            index_sink_2 = current_pair[1] == all_pairs[:, 1]
            invalid_pairs = index_source_1 + index_sink_1 + index_source_2 + index_sink_2
            valid_pairs = np.delete(all_pairs, invalid_pairs,axis=0)
            print("No. valid pairs: ",len(valid_pairs))
            if len(volt_mat)==0:
                volt_mat = valid_pairs
            else:
                volt_mat = np.append(volt_mat,valid_pairs,axis=0)
        ind = np.zeros(len(volt_mat))
        pairs_per_current = int(len(volt_mat)/len(ex_mat))
        print(pairs_per_current)
        for i in range(0,len(ex_mat)):
            ind[pairs_per_current*i:pairs_per_current*(i+1)] = i
    else:
        print("Incorrect voltage pair mode selected")
        

    return volt_mat, ind.astype(np.int32)
        

def getNextPrediction_partialforwardsolver(mesh_obj: None, volt_mat: None, ex_mat: np.ndarray, ind: None, voltages: np.ndarray, 
              num_returned: int=10, n_el: int=20, n_per_el: int=3, n_pix: int=64, pert: float=0.5, 
              p_influence: float=-10., p_rec: float=10.) -> np.ndarray:
    
    # create an object with the meshing characteristics to initialise a Forward object
    el_pos = np.arange(n_el * n_per_el).astype(np.int16)
    if mesh_obj == None:
        mesh_obj = mesh(n_el)
    
    fwd = Forward_given(mesh_obj, el_pos, n_el)
    if (volt_mat is None):
        f, meas, new_ind = fwd.solve_eit(ex_mat=ex_mat)
        #voltages = np.random.random(len(meas)) # Generate random voltage measurments for proof-of-concept
    #elif (volt_mat is not None):
    elif (volt_mat is not None) and (ind is not None):
        f, meas, new_ind = fwd.solve_eit(volt_mat=volt_mat, ex_mat=ex_mat, new_ind=ind)
    else:
        print("Eh-ohhhh!")
    
    # now we can use the real voltage readings and the GREIT algorithm to reconstruct
    greit = train.greit.GREIT(mesh_obj, el_pos, f=f, ex_mat=(meas[:, :2]), step=None)
    greit.setup(p=0.2, lamb=0.01, n=n_pix)
    h_mat = greit.H
    reconstruction = greit.solve(voltages, f.v).reshape(n_pix, n_pix)
    # fix_electrodes_multiple is in meshing.py
    _, el_coords = train.fix_electrodes_multiple(centre=None, edgeX=0.1, edgeY=0.1, a=2, b=2, ppl=n_el, el_width=0.2, num_per_el=3)
    # find the distances between each existing electrode pair and the pixels lying on the line that connects them
    pixel_indices, voltage_all_possible = measopt.find_all_distances(reconstruction, h_mat, el_coords, n_el, cutoff=0.8)
    # call function get_total_map that generates the influence map, the gradient map and the log-reconstruction
    total_map, grad_mat, rec_log = np.abs(measopt.get_total_map(reconstruction, voltages, h_mat, pert=pert, p_influence=p_influence, p_rec=p_rec))
    # get the indices of the total map along the lines connecting each possible electrode pair
    total_maps_along_lines = total_map[None]*pixel_indices
    # find how close each connecting line passes to the boundary of an anomaly (where gradient supposed to be higher)
    proximity_to_boundary = np.sum(total_maps_along_lines, axis=(1, 2)) / np.sum(pixel_indices, axis=(1, 2))
    # rate the possible src-sink pairs by their proximity to existing anomalies
    proposed_ex_line = voltage_all_possible[np.argsort(proximity_to_boundary)[::-1]][:num_returned]
    
    ex_line_loops = 1 # From 1 to len(proposed_ex_line)
    """
    Currently, we only pass new_ex_mat into the forward solver which regenerates the jacobian to include our new
    proposed excitations lines. This then uses the voltMeter function with step=1 to generate adjacent voltage pairs. 
    For now this suffices but for full user control of the new jacobian voltage pairs, will need to:
        1) Code methods to produce volt matrices for voltage pairs defined as per adj, opp, any/all, random, explicit (user defined)
        2) Create a temporary ind to match the number of elements added from the temp volt_mat 
        3) Pass the temp_volt_mat, temp_ind and new_ex_mat_lines into forward solver to solve for relevant jacobian parts
    
    Currently, this regenerates the jacobian each time but in the future we want to modify this to build and add onto a jacobian
    which has been saved to a file. We will have to be clever about how we instruct the code to search over the Jacobian of interest.
    Regenerating the jacobian each time, even for a only partial jacobian, is not ideal.
    
    """
    order = np.ones(ex_line_loops*num_returned)
    for i in range(0, ex_line_loops):
        order[num_returned*i:num_returned*(i+1)] = new_ind[-1]+1+i
    new_ind = np.append(new_ind, order).astype(np.int16)
    new_ex_mat = np.vstack((ex_mat, proposed_ex_line[0:ex_line_loops])) # Vertical stack new current pair onto excitation matrix
    #f, new_meas, new_ind_2 = fwd.solve_eit(ex_mat=new_ex_mat) # Regenerate Jacobian with new current pair
    f, partial_meas, new_ind_2 = fwd.solve_eit(ex_mat=proposed_ex_line[0:ex_line_loops]) # Regenerate Jacobian with new current pair
    
    save_filename = 'relevant_jacobian_slice.h5'
    h = h5.File(save_filename, 'w')

    try:
        h.create_dataset('jac', data=f.jac)
        h.create_dataset('meas', data=partial_meas)
        h.create_dataset('p', data=mesh_obj['node'])
        h.create_dataset('t', data=mesh_obj['element'])
    except:
        TypeError('Error with saving files!')
    h.close()
    
    number_of_voltages = num_returned
    proposed_ex_volt_mat = []
    for i in range(0, ex_line_loops): # loop over the proposed current pairs
        # Generate the voltage measuring electrodes for this current driver pair
        proposed_voltage_pairs = measopt.findNextVoltagePair(proposed_ex_line[i], save_filename, total_map, number_of_voltages, 0, npix=n_pix, cutoff=0.97)
        print("Proposed current pair:", proposed_ex_line[i])
        print("Proposed voltage pairs for given current pair:\n",proposed_voltage_pairs)
        for volt_pair in proposed_voltage_pairs:
            line = np.hstack((proposed_ex_line[i], volt_pair))
            if len(proposed_ex_volt_mat)==0:
                proposed_ex_volt_mat = line
            else:
                proposed_ex_volt_mat = np.vstack((proposed_ex_volt_mat, line))

    meas = np.vstack((meas, proposed_ex_volt_mat)) # meas == ex_volt_meas
    return proposed_ex_volt_mat, meas, new_ex_mat, new_ind, reconstruction, total_map

def initialise_ex_volt_mat(current_mode:str='adj', volt_mode:str='adj',n_el:int=32, ex_mat_length:int=10):
    """
    This function is used to initialise a current and voltage excitation matrix. Currently hardcoded for to opposite currents.
    returns
    ex_volt_mat - The combined current pair, volt pair matrix (N,4)
    ex_mat
    volt_mat
    ind - a list of integers which can be used like ex_mat = ex_mat[ind] to duplicate the current pair array entires to match
          the format of volt_mat for concatenation.
    
    """
    ex_mat = train.orderedExMat(n_el=n_el, length = ex_mat_length, el_dist=n_el//2) # currently set to opposite current pair mode
    ex_mat = ex_mat[0:10] # take the first 0 to N current pairs for initial ex_mat
    volt_mat, ind = volt_matrix_generator(ex_mat=ex_mat, n_el=n_el, volt_pair_mode=volt_mode)
    ex_volt_mat = np.concatenate((ex_mat[ind], volt_mat), axis=1) 
    return ex_volt_mat, ex_mat, volt_mat, ind

def adaptive_ESA_single_interation(mesh_obj,volt_mat, ex_mat, ind, voltages, 
                                   num_returned, n_el=32, do_plot=True):
    """
    

    Parameters
    ----------
    mesh_obj : TYPE
        Blank mesh.
    volt_mat : TYPE
        DESCRIPTION.
    ex_mat : TYPE
        DESCRIPTION.
    ind : TYPE
        DESCRIPTION.
    voltages : TYPE
        Measurement voltage values in the order which matches the combined ex_volt_mat.
    num_returned : TYPE
        Number of voltage pairs to return per proposed current line
    n_el : TYPE, optional
        DESCRIPTION. The default is 32.
    do_plot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    proposed_ex_volt_lines, ex_volt_meas, ex_mat, ind, reconstruction, total_map = getNextPrediction_partialforwardsolver(mesh_obj, 
                                        volt_mat=volt_mat, ex_mat=ex_mat, ind=ind, voltages=voltages, num_returned=num_returned, n_el=n_el) 
    volt_mat = ex_volt_meas[:, 2:]
    print("\nSize of excitation matrix.")
    print("No. of current pairs: ", len(ex_mat))
    print("No. of voltage pairs: ", len(volt_mat))
    if (do_plot == True):
        plt.figure()
        im = plt.imshow(reconstruction, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
        plt.title("GREIT Reconstruction\n(Current, Voltage) pairs: ("+str(len(ex_mat))+", "+str(len(volt_mat))+")")
        plt.colorbar(im)
        #plt.savefig(filepath+"ESA reconstruction "+str(i))
        plt.show()
    return proposed_ex_volt_lines, ex_volt_meas, ex_mat, ind, reconstruction, total_map


    
    

if __name__ == "__main__":
    
    # you need to input several things: conducted measurements in shape (num_meas, 4), 
    # where first column is source, second sink and other two are voltage measuring electrodes
    # voltage readings in shape (num_meas) in the same order as conducted measurements
    n_el = 32
    n_per_el = 3
    n_pix = 64
    simulate_anomalies = True

    # Define initial current and voltage pair matrix 
    ex_volt_mat, ex_mat, volt_mat, ind = initialise_ex_volt_mat(current_mode='poo', volt_mode='adj',n_el=32, ex_mat_length=10)
    mesh_obj = mesh(n_el)
    
    """
    Initialisation of ex_volt_mat
    ex_volt_mat format
    current pair 1 volt pair 1 
    current pair 1 volt pair 2
    current pair 1 volt pair 3
    current pair 1 volt pair 4
    current pair 1 volt pair 5
    current pair 2 volt pair 6
    current pair 2 volt pair 7
    current pair 2 volt pair 8
    current pair 2 volt pair 9
    ...
    ...
    current pair N volt pair M
    
    
    """
    
    # Either simulate data or read in real data
    if simulate_anomalies is True:
        print("Simulating anomalies and voltage data")
        a = 2.0
        anomaly = train.generate_anoms(a, a)
        true = train.generate_examplary_output(a, int(n_pix), anomaly)
        el_pos = np.arange(n_el * n_per_el).astype(np.int16)
        fwd = Forward_given(mesh_obj, el_pos, n_el)
        mesh_new = train.set_perm(mesh_obj, anomaly=anomaly, background=1)
        f_sim, dummy_meas, dummy_ind = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat,
                                                                      perm=mesh_new['perm'].astype('f8'))
        plt.figure()
        im = plt.imshow(true, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
        plt.colorbar(im)
        plt.title("True Image")
        #plt.savefig(filepath+"True image")
        plt.show()
        
    
    voltages = f_sim.v #  Assigning the simulated voltages in place of real voltages
    proposed_ex_volt_lines, ex_volt_meas, ex_mat, ind, reconstruction, total_map = adaptive_ESA_single_interation(mesh_obj,
                                                                                    volt_mat, ex_mat, ind, voltages, num_returned=10, n_el=32)
    volt_mat = ex_volt_meas[:, 2:]
    """
    proposed_ex_volt_lines has one current pair denoted A, and 10 corresponding voltages
    ex_volt_meas = old_ex_volt_mat + Proposed_ex_volt_lines

    current pair 1 volt pair 1      Ex Mat: Current pair 1
    current pair 1 volt pair 2              Current pair 2
    current pair 1 volt pair 3              ....
    current pair 1 volt pair 4              Current pair N
    current pair 1 volt pair 5              Current pair A
    current pair 2 volt pair 6
    current pair 2 volt pair 7
    current pair 2 volt pair 8
    current pair 2 volt pair 9
    ...
    ...
    current pair N volt pair 
    Current pair A volt pair A1
    ...
    current pair A voltpair A10
    
    Now we want to use proposed_ex_volt_lines to get the new measurements. Pass proposed_ex_volt_lines (say size 10 for sake of it) into the
    Hardware measurement functions. This will gives us a new set of voltages (New_voltages -> 10 new values)
    Append these voltages onto the old list of voltages: voltages = np.append(voltages, New_voltages)
    
    To repeat we need volt_mat, ex_mat, ind and voltages
    """ 
    
    
    # Ex_mat is the array of all of the current pairs we have measured over
    # volt_mat is the array of all of the voltage pairs we have measured over number_volt_pairs*number_current_pairs
    # For every current pair (ex_line) we have say 5 voltage pairs

    # Proposed_ex_volt_lines = an array of the new proposed measuremetns in Ex Mat + volt_mat format
    
    # Ex_volt_meas = old
    #expanded_ex_mat = ex_mat[ind]
    # Pass proposed_ex_volt_lines into equipment and get corresponding voltages, in matchnig order
    # Append new voltage values onto old voltages voltages = np.append(voltages, New_meas_voltages)
    volt_mat = ex_volt_meas[:, 2:]
"""
    
    el_pos = np.arange(n_el * n_per_el).astype(np.int16)
    ex_mat_full_adj_adj = train.orderedExMat(n_el=n_el, length = 32, el_dist=1) # Adj-adj ex_mat
    ex_mat_initial = ex_mat_full_adj_adj[0:5]# Only take the first 5 current pairs for an initial seeding
    volt_mat, ind = volt_matrix_generator(ex_mat=ex_mat_initial, n_el=n_el, volt_pair_mode='adj')
    #ex_mat_initial = ex_mat_initial[ind]
    #ex_volt_mat = np.concatenate((ex_mat, volt_mat), axis=1)
    
    plot_set = 5
    filepath = os.getcwd() + "\\comparison plots\\set_"+str(plot_set)+"\\"
    try:
        os.mkdir(filepath)
    except:
        choice = input("set_"+str(plot_set)+" already exists, do you want to replace these plots? (y/n):")
        if (choice=="n"):
            exit(0)
        print("Overriding plots in folder set_"+str(plot_set))
    
    mesh_obj = mesh(n_el)
    fwd = Forward_given(mesh_obj, el_pos, n_el)
    empty_mesh_f, meas, ind = fwd.solve_eit(volt_mat=volt_mat, ex_mat=ex_mat_initial, new_ind=ind)
    ex_mat = ex_mat_initial
    if simulate_anomalies is True:
        print("Simulating anomalies and voltage data")
        a = 2.0
        anomaly = train.generate_anoms(a, a)
        true = train.generate_examplary_output(a, int(n_pix), anomaly)
        mesh_new = train.set_perm(mesh_obj, anomaly=anomaly, background=1)
        f_sim, dummy_meas, dummy_ind = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat,
                                                                      perm=mesh_new['perm'].astype('f8'))
        greit = train.greit.GREIT(mesh_obj, el_pos, f=f_sim, ex_mat=ex_mat, step=1)
        greit.setup(p=0.2, lamb=0.01, n=n_pix)
        h_mat = greit.H
        reconstruction_initial = greit.solve(f_sim.v, empty_mesh_f.v).reshape(n_pix, n_pix)
        plt.figure()
        im1 = plt.imshow(true, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
        plt.colorbar(im1)
        plt.title("True Image")
        plt.savefig(filepath+"True image")
        plt.show()
        plt.figure()
        im2 = plt.imshow(reconstruction_initial, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
        plt.title("Initial Reconstruction\n(Current, Voltage) pairs: ("+str(len(ex_mat))+", "+str(len(volt_mat))+")")
        plt.colorbar(im2)
        #save_path = os.path.join(out_dir, ')
        plt.savefig(filepath+"Initial Reconstruction")
        plt.show()

    
    print("\nInitial size of excitation matrix.")
    print("No. of current pairs: ", len(ex_mat))
    print("No. of voltage pairs: ", len(volt_mat))
    prediction_loops = 25
    for i in range(1,prediction_loops+1):
        print("\nLoop:",i," ------------------------")
        if simulate_anomalies is True:
            voltages = f_sim.v
            
        proposed_ex_volt_lines, ex_volt_meas, ex_mat, ind, reconstruction, total_map = getNextPrediction_partialforwardsolver(mesh_obj, 
                                        volt_mat=volt_mat, ex_mat=ex_mat, ind=ind, voltages=voltages, num_returned=10, n_el=n_el) 
        volt_mat = ex_volt_meas[:, 2:]
        print("\nSize of excitation matrix.")
        print("No. of current pairs: ", len(ex_mat))
        print("No. of voltage pairs: ", len(volt_mat))
        if simulate_anomalies is True:
            f_sim, dummy_meas, dummy_ind = fwd.solve_eit(volt_mat=volt_mat, new_ind=ind, ex_mat=ex_mat,
                                                          perm=mesh_new['perm'].astype('f8'))
        if i in (1, prediction_loops//4,prediction_loops//2, 3*prediction_loops//4, prediction_loops):
            plt.figure()
            im = plt.imshow(reconstruction, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
            plt.title("Reconstruction: "+str(i)+"\n(Current, Voltage) pairs: ("+str(len(ex_mat))+", "+str(len(volt_mat))+")")
            plt.colorbar(im)
            plt.savefig(filepath+"ESA reconstruction "+str(i))
            plt.show()
            if i in (1, prediction_loops//4,3*prediction_loops//4):
                plt.close()
            
    print("\nCreating Adj-Adj Comparison plots")            
    #comp_ex_mat = train.orderedExMat(n_el=n_el, length = 32, el_dist=1) # Adj-adj ex_mat
    for i in (10,15,20,25,32):
        comp_ex_mat = ex_mat_full_adj_adj[0:i]
        empty_mesh_f, empty_meas, empty_ind = fwd.solve_eit(ex_mat=comp_ex_mat)
        f_comp, comp_meas, comp_ind = fwd.solve_eit(volt_mat=empty_meas[:, 2:], new_ind=empty_ind, ex_mat=comp_ex_mat,perm=mesh_new['perm'].astype('f8'))
        greit = train.greit.GREIT(mesh_obj, el_pos, f=f_comp, ex_mat=comp_ex_mat, step=1)
        greit.setup(p=0.2, lamb=0.01, n=n_pix)
        h_mat = greit.H
        comp_reconstruction = greit.solve(f_comp.v, empty_mesh_f.v).reshape(n_pix, n_pix)
        plt.figure()
        im_comp = plt.imshow(comp_reconstruction, cmap=plt.cm.viridis, origin='lower', extent=[-1, 1, -1, 1])
        plt.title("Adj-Adj Reconstruction\n(Current, Voltage) pairs: ("+str(len(comp_ex_mat))+", "+str(len(empty_meas[:, 2:]))+")")
        plt.colorbar(im_comp)
        plt.savefig(filepath+"adj-adj reconstruction "+str(i))
        plt.show()
        if i in (10,20,25):
            plt.close()
    print("\nEnd of Script ------------------------")



        # number of electrodes
    # num_meas = 20
    # n_el = 20
    # #save_file = fileJac = "given_meas.h5"
    # condition = True

    # #ex_mat_initial = train.generateExMat(ne=n_el)
    # #saveJacobian(save_filename = save_file,n_el=20, n_per_el=3)
    # #save_small_Jacobian(save_filename = save_file,n_el=20, n_per_el=3)
    # # this loop is to simulate an array of measurement electrodes without repeated indices on the same line; thus the check
    # # if you want to simulate lots of measurements, it will loop forever (there are always repeated ones)
    # while condition:
    #     measuring_electrodes = np.random.randint(0, 20, (num_meas, 4))
    #     col1 = measuring_electrodes[:, 0]
    #     col2 = measuring_electrodes[:, 1]
    #     col3 = measuring_electrodes[:, 2]
    #     col4 = measuring_electrodes[:, 3]
    #     # checks whether each column is different from the others (on the same line)
    #     eq1 = np.amax(np.diag(col1[None] == col2[:, None]))
    #     eq2 = np.amax(np.diag(col1[None] == col3[:, None]))
    #     eq3 = np.amax(np.diag(col1[None] == col4[:, None]))
    #     eq4 = np.amax(np.diag(col2[None] == col3[:, None]))
    #     eq5 = np.amax(np.diag(col2[None] == col4[:, None]))
    #     eq6 = np.amax(np.diag(col3[None] == col4[:, None]))
    #     # if any is true, keep looping
    #     condition = (eq1 + eq2 + eq3 + eq4 + eq5 + eq6)
    
    # simulate random voltage array
    #voltage_readings = np.random.random(num_meas)
"""