# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:28:08 2020
Authors: Adam Coxson and Frederik Brooke Barnes, MPhys Undergrads, The University of Manchester
Project: Automated Electrical Impedance tomography of Graphene
Module:  component_control
Dependancies: pyVisa package

This script acts as a control interface between a remote system and the
lock-in amplifier and Cytec switchbox used to measure voltage across samples.
This uses GPIB commands.

So far it is just a skeleton
Setup:
    - Connect to device and do initial compatibility checks.
    - Allocate electrode position identifiers to the corresponding switch contacts.

Electrode switch algorithm:
    Main outer loop over the 32 electrodes for the driving current pairs:
    	- Switch to new contact pair
    	- Check and verify current contacts are open and active.
    	- Activate contacts to drive current between contacts.
    	- Apply delay required for lock-in amplifier to cycle up.
    	- Perform inner voltage loop as defined below.
    	- Terminate driving current and close contacts, write outputs to file.

	Inner loop over the remaining 30 electrodes for voltage pair measurements:
		- Switch to a new voltage pair.
		- Check status and open the contacts.
		- Query contacts for a voltage measurement.
		- Apply delay required for lock-in amplifier to cycle up.
		- Record average voltage measurement.
		- Close and re-check contacts .
		- Output voltage measurement and write to file.
	
Termination:
    After the current and voltage electrode loops, one needs to manually ensure the system is shut down.
    - Check status of each contact, ensure all are closed.
    - Write all final data out to file and save.
    - Terminate connection.

"""
#IMPORT DEPENDENCIES
import pyvisa
import numpy as np
from numpy import random
import time
from selection_algorithms import *

#SET DEVICE IP AND PORT
#lock-in amplifier
lockin_ip = '169.254.147.1'
lockin_port = '1865'  #By default, the port is 1865 for the SR860.
lockin_lan_devicename = 'inst0' #By default, this is inst0. Check NI MAX

#switchboard
''' SWITCH USED IS ACCESSED VIA GPIB NOT LAN
switch_ip = '10.0.0.2' #default for CYTECH IF-6 module as used in VX256 is 10.0.0.2
switch_port = '23456' #default for CYTECH IF-6 module as used in VX256 is 23
'''

#SET DEVICE GPIB ADDRESS
#switchboard
switch_primary_address = '7'


#create devices (resources) address strings
switch_address = 'GPIB0::'+switch_primary_address+'::INSTR'
lockin_address = 'TCPIP::'+lockin_ip+'::'+lockin_lan_devicename+'::'+'INSTR'


#create resource manager using py-visa backend ('@py') leave empty for NI VIS
rm = pyvisa.ResourceManager()
#print available devices (resources)
print(rm.list_resources())

#connect to devices
switch = rm.open_resource(switch_address)
lockin = rm.open_resource(lockin_address)

#set termination characters
#switch
switch.read_termination = '\n' #cytech manual says 'enter' so try \n, \r or combination of both
switch.write_termination = '\n'
#lockin
#lockin.read_termination = '\f' #SR860 manual says \lf so \f seems to be equivalent in python)
lockin.write_termination = '\f'

def SetMeasurementParameters(parameters):

	'''
	Input: list of strings
	Ouput: None
	Assigns a parameter to data channel for each parameter given in array of strings. 
	If fewer than 4 parameters are given, the remaining channels will not be changed from previous state.
	The parameter list is
	i enumeration
	0 X
	1 Y
	2 R
	3 THeta
	4 IN1
	5 IN2
	6 IN3
	7 IN4
	8 XNOise
	9 YNOise
	10 OUT1
	11 OUT2
	12 PHAse 
	13 SAMp 
	14 LEV el 
	15 FInt 
	16 FExt

	'''
	if parameters == None:
		parameters = ["R","THeta","SAMp","FInt"]
		
	channel = 1
	for i in range(0, min(4, len(parameters))):
		lockin.write('CDSP DAT'+str(channel)+", "+str(parameters[i])) #The CDSP j, param command assigns a parameter to data channel j. This is the same parameter assignment as pressing the [Config] key.
		channel += 1
	return

def GetMeasurement(parameters=None, param_set=True):

	'''
	Input 
	parameters: List of strings(optional) corresponding to parameters desired to be measured by lock-in SR860. If none, defaults to R, THeta, SAMp, FInt
	param_set: Boolean. If true, set the parameters to be measured. If false, take measurement using previously set parameters (Speeds up measurement by ~0.03s)
	Output 
	measurement_array: Numpy array of floats corresponding to mesaurment values in Volts, Hz or Degrees.
	'''
	if param_set == True:
		SetMeasurementParameters(parameters)
	measurement = lockin.query('SNAPD?')
	measurment_array = np.fromstring(measurement, sep=',')

	return measurment_array

def FlickSwitch(state, module, relay):

	'''
	input: state(string or int), module(int), relay(int)
	output: sends message to switchbox to change state of switch according to 
	state given by string ('on' or 'off') or int (0 or 1). Switch corresponds to
	relay within module.
	'''

	if state == 1 or state=='on':
		state_str = "L"
	elif state == 0 or state=='off':
		state_str = "U"
	else:
		print("Must include switch state. 0(open) 1(closed)")
		return
	
	switch.write(state_str+str(module)+" "+str(relay))
	
	return 

def MapSwitches(electrode, lockin_connection):
	'''
	given a lock-in connection ("sin+" or 0,"sin-" or 1,"v+" or 2,"v-" or 3) and electrode number (int)
	returns module and relay numbers
	'''
	if lockin_connection is str:
		lockin_connection = {'sin+':0, 'sin-':1, 'v+':2, 'v-':3}

	relay = electrode % 16
	module = ( electrode // 16 ) + lockin_connection


	return module, relay

def ClearSwitches():

	switch.write('C')

	return

def eit_scan_lines(ne=16, dist=1):
    """

	TAKEN FROM pyeit.eit.utils.py

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


def GetNextElectrodes(*algorithm_parameters, algorithm='Standard', no_electrodes=32, measurement):

	'''
	Returns electrode connections (eg sin+:2, sin-:1, v+: 18, v-:17 given algorithm used 
	and required information eg measurement no. or previous measurement. In order of sin+, sin-, v+, v-.
	If a list of electrodes are already given, it simply returns the nth element in that array. 
	'''

	if algorithm == 'Standard':
		all_measurement_electrodes = algorithm_parameters[0]

		if all_measurement_electrodes == None:
				all_measurement_electrodes = Standard(no_electrodes, step=1, parser=None)

		next_electrodes = all_measurement_electrodes[measurement]


	if algorithm == 'random':
		rng = random.default_rng()
		next_electrodes = rng.choice(15, size=4, replace=False)

	return next_electrodes

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
                FlickSwitch('on', MapSwitches(electrode=next_electrodes[i], lockin_connection=i))
            r, theta, samp, fint = GetMeasurement(param_set=False)
            v_difference.append(r)
        v_difference = np.array(v_diff)

    return v_difference

print(RunEIT(no_electrodes=6, max_measurements=100))



'''
#initialise devices
#open all switches
switch.write('C') #C = Clear = open all relays or turn OFF all relays
switch_status = switch.query('S') #S = Status which can be used on individual switch points, modules, or the entire system. Replies with string eg 0000000 showing 8 closed switches
print(switch_status)
switch.write('L3 5')
print(switch.query('S'))

#reset lock-in
print("resetting lock-in")
#check timebase is internal
tb_status = lockin.query('TBSTAT?') #Query the current 10 MHz timebase ext (0) or int (1)
print("Timebase status:", tb_status)

#set reference phase
PHASE_INIT = 0
lockin.write('PHAS '+str(PHASE_INIT)) #Set the reference phase to PHASE_INIT
phase = lockin.query('PHAS?') #Returns the reference phase in degrees
print("Reference phase:", phase)

#set frequency
FREQ_INIT = 1e4
lockin.write('FREQINT '+ str(FREQ_INIT)) #Set the internal frequency to FREQ_INIT
freq = lockin.query('FREQINT?') #Returns the internal frequency in Hz
print("Frequency: ", freq)

#set sine out voltage
VOUT_INIT = 0.5
lockin.write('SLVL '+str(VOUT_INIT)) #Set the sine out amplitude to VOUT_INT in Volts The amplitude may be programmed from 1 nV to 2.0V
vout = lockin.query('SLVL?') #Returns the sine out amplitude in Volts
print("Sine out amplitude: ", vout)


#Assign parameters to data channels. Lock-in is capable or reading 4 data points simultaneously. 
lockin.write('CDSP DAT1 R') 		#set channel 1 to R
lockin.write('CDSP DAT2 THetha')	#set channel 2 to theta
lockin.write('CDSP DAT3 SAMp')		#set channel 3 to sine out amplitude
lockin.write('CDSP DAT4 FInt')		#set channel 4 to internal reference frequency

#Auto adjust range and scaling of measurements
lockin.write("ARNG") #auto range 
lockin.write("ASCL") #auto scale


	

params = ["X", "Y", "OUT1", "OUT2"]
SetMeasurementParameters(params)
start_false = time.time()
data = GetMeasurement(param_set=False)
end_false=time.time()
time_false = -(start_false-end_false)
print("Time to get measurement without setting params:", time_false)

print(data)
start_true = time.time()
data = GetMeasurement()
end_true=time.time()
time_true = end_true - start_true
print("Time to get measurement with setting params:", time_true)
print(data)

#INSERT CODE HERE

'''