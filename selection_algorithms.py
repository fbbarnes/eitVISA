
import numpy as np
import numpy.random as random


def eit_scan_lines(ne=16, dist=1):
    """
    generate scan matrix
    Parameters
    ----------
    ne: int
        number of electrodes
    dist: int
        distance between A and B (default=1)
    Returns
    -------
    ex_mat: NDArray
        stimulation matrix
    Notes
    -----
    in the scan of EIT (or stimulation matrix), we use 4-electrodes
    mode, where A, B are used as positive and negative stimulation
    electrodes and M, N are used as voltage measurements
    1 (A) for positive current injection,
    -1 (B) for negative current sink
    dist is the distance (number of electrodes) of A to B
    in 'adjacent' mode, dist=1, in 'apposition' mode, dist=ne/2
    Examples
    --------
    # let the number of electrodes, ne=16
    if mode=='neighbore':
        ex_mat = eit_scan_lines()
    elif mode=='apposition':
        ex_mat = eit_scan_lines(dist=8)
    WARNING
    -------
    ex_mat is a local index, where it is ranged from 0...15, within the range
    of the number of electrodes. In FEM applications, you should convert ex_mat
    to global index using the (global) el_pos parameters.
    """
    ex = np.array([[i, np.mod(i + dist, ne)] for i in range(ne)])

    return ex

def voltage_meter(ex_line, n_el=16, step=1, parser=None):
    """
    extract subtract_row-voltage measurements on boundary electrodes.
    we direct operate on measurements or Jacobian on electrodes,
    so, we can use LOCAL index in this module, do not require el_pos.
    Notes
    -----
    ABMN Model.
    A: current driving electrode,
    B: current sink,
    M, N: boundary electrodes, where v_diff = v_n - v_m.
    'no_meas_current': (EIDORS3D)
    mesurements on current carrying electrodes are discarded.
    Parameters
    ----------
    ex_line: NDArray
        2x1 array, [positive electrode, negative electrode].
    n_el: int
        number of total electrodes.
    step: int
        measurement method (two adjacent electrodes are used for measuring).
    parser: str
        if parser is 'fmmu', or 'rotate_meas' then data are trimmed,
        boundary voltage measurements are re-indexed and rotated,
        start from the positive stimulus electrodestart index 'A'.
        if parser is 'std', or 'no_rotate_meas' then data are trimmed,
        the start index (i) of boundary voltage measurements is always 0.
    Returns
    -------
    v: NDArray
        (N-1)*2 arrays of subtract_row pairs
    """
    # local node
    drv_a = ex_line[0]
    drv_b = ex_line[1]
    i0 = drv_a if parser in ("fmmu", "rotate_meas") else 0

    # build differential pairs
    v = []
    for a in range(i0, i0 + n_el):
        m = a % n_el
        n = (m + step) % n_el
        # if any of the electrodes is the stimulation electrodes
        if not (m == drv_a or m == drv_b or n == drv_a or n == drv_b):
            # the order of m, n matters
            v.append([n, m])

    diff_pairs = np.array(v)
    return diff_pairs

def Standard(no_electrodes, step=1, parser=None):
    '''
    Inputs
    ------
    no_electrodes: int
        specifies number of electrodes
    step: int
        specifies the pattern to take measurements at (see voltage_meter), 
        step=1 for adj-adj, two adjacent electrodes are used for measuring
    parser: str
        parser: str
        if parser is 'fmmu', or 'rotate_meas' then data are trimmed,
        boundary voltage measurements are re-indexed and rotated,
        start from the positive stimulus electrodestart index 'A'.
        if parser is 'std', or 'no_rotate_meas' then data are trimmed,
        the start index (i) of boundary voltage measurements is always 0.
    
    Output
    ------
    electrodes: NDarray
        no_electrodes array*4 array of electrode numbers in the order ABMN aka 
        sin+, sin-, v+, v-
    '''
    #print(parser)
    scan_lines = eit_scan_lines(no_electrodes)


    electrodes = []
    for i in range(0, len(scan_lines)):
        measurements = voltage_meter(scan_lines[i], n_el=no_electrodes, step=step, parser=parser)

        for j in range(0, len(measurements)):
            electrodes.append(np.concatenate((scan_lines[i], measurements[j])))

    electrodes = np.array(electrodes)
    return electrodes


def GetNextElectrodes(algorithm='Standard', no_electrodes=32, all_measurement_electrodes=None, measurement=0, **algorithm_parameters):

    '''
    Returns electrode connections (eg sin+:2, sin-:1, v+: 18, v-:17 given algorithm used 
    and required information eg measurement no. or previous measurement. In order of sin+, sin-, v+, v-.
    If a list of electrodes are already given, it simply returns the nth element in that array. 
    '''
    not_last_measurement = True
    next_electrodes = np.zeros(4)

    #print(algorithm_parameters)
    if algorithm == 'Standard':

        #if all_measurement_electrodes == None:
            #all_measurement_electrodes = Standard(no_electrodes, **algorithm_parameters)

        if measurement >= len(all_measurement_electrodes):
            not_last_measurement = False
            
        if not_last_measurement == True:
            next_electrodes = all_measurement_electrodes[measurement]


    if algorithm == 'Random':
        rng = random.default_rng()
        next_electrodes = rng.choice(no_electrodes-1, size=4, replace=False)

    return next_electrodes, not_last_measurement

'''
def RunEIT(algorithm='Standard', no_electrodes=32, max_measurements=None, measurement_electrodes = None, **algorithm_parameters):

    ClearSwitches()


    #standard_measurement_electrodes = Standard(no_electrodes=6, step=1,parser='fmmu')

    #print(standard_measurement_electrodes)


    keep_measuring = True

    if max_measurements == None:
        max_measurements = 10000

    v_difference = []

    while keep_measuring == True:
        for i in range(0,max_measurements):

            next_electrodes, keep_measuring = GetNextElectrodes(algorithm=algorithm, no_electrodes=no_electrodes, measurement=i)
            print("measurement "+str(i)+", next electrode "+str(next_electrodes)+"keep measuring:"+str(keep_measuring))
            if keep_measuring == False:
                break
            print(next_electrodes)
            ClearSwitches()
            for i in next_electrodes:
                FlickSwitch(on, MapSwitches(electrode=next_electrodes[i], lockin_connection=i))
            r, theta, samp, fint = GetMeasurement(param_set=False)
            v_difference.append(r)
        v_difference = np.array(v_diff)

    return return v_difference
'''

#RunEIT(no_electrodes=6, max_measurements=1000)