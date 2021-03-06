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
from datetime import datetime
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
rm = pyvisa.ResourceManager() #create resource manager using py-visa backend ('@py') leave empty for NI VIS
#print available devices (resources)
print(rm.list_resources())
switch = rm.open_resource(switch_address) #connect to devices
lockin = rm.open_resource(lockin_address)
#set termination characters
switch.read_termination = '\n' #cytech manual says 'enter' so try \n, \r or combination of both
switch.write_termination = '\n'
#lockin.read_termination = '\f' #SR860 manual says \lf so \f seems to be equivalent in python)
lockin.write_termination = '\f'

print("switch", switch.session)
print("lockin", lockin.session)
lockin.flush
def SetMeasurementParameters(parameters):

	'''
	Inputs
	------
	parameters: list of str or int
		parameters desired to be measured

	Outputs
	------
	None

	Notes
	------
	Assigns a parameter to data channel of SR860 lock-in for each parameter given in array of strings. 
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
		parameters = ["X","THeta","XNoise","FInt"]
		
	channel = 1
	for i in range(0, min(4, len(parameters))):
		lockin.write('CDSP DAT'+str(channel)+", "+str(parameters[i])) #The CDSP j, param command assigns a parameter to data channel j. This is the same parameter assignment as pressing the [Config] key.
		channel += 1
	return

def GetMeasurement(parameters=None, param_set=True):

	'''
	Inputs
	------ 
	parameters: list of str 
		corresponding to parameters desired to be measured by lock-in SR860. If none, defaults to R, THeta, SAMp, FInt
	param_set: bool. 
		If true, set the parameters to be measured. If false, take measurement using previously set parameters (Speeds up measurement by ~0.03s)
	Outputs
	------  
	measurement_array: NDarray
		Array of floats corresponding to mesaurment values in Volts, Hz or Degrees. Ordered in same order as specified in parameters.
	Notes
	------
	Uses SNAPD? lockin-command
	'''
	if param_set == True:
		SetMeasurementParameters(parameters)
	measurement = lockin.query('SNAPD?')
	measurment_array = np.fromstring(measurement, sep=',')

	return measurment_array

def FlickSwitch(state, module, relay):

	'''
	Inputs
	------  
	state: str or int
		State to change switch to. 'on' (0) or 'off' (1).
	module: int
		Module number desired switch is in. 
	relay: int
		Relay(aka switch) number of desired switch within module.

	Outputs
	------  
	None

	Notes
	------  
	Sends message to switchbox to change state of switch according to 
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
	switch.write(state_str+str(relay)+" "+str(module))
	return 0

def MapSwitches(electrode, lockin_connection):
	'''
	Inputs
	------ 
	electrode: int
		Electrode number corresponding to numbering on output of switchbox.
	lockin_connection: str
		Relevant lock-in connnection ("sin+" is 0,"sin-" is 1,"v+" is 2,"v-" is 3)

	Outputs
	------ 
	module: int
		Module number corresponding to relay needed to connect electrode to lockin_connection
	relay: int
		Relay number within module needed to connect electrode to lockin_connection
	'''
	relay = electrode % 16
	module = ((electrode // 16) * 8)+ lockin_connection
	return module, relay



def ClearSwitches():
	'''
	Opens all switches in switchbox
	'''
	switch.write('C')
	return

def eit_scan_lines(ne=16, dist=1):
	"""
	TAKEN FROM pyeit.eit.utils.py
	Generates an excitation scan matrix of current and voltage electrode pairs.
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




def RunEIT(algorithm='Standard', no_electrodes=32, max_measurements=10000, measurement_electrodes = None, 
			print_status=True, voltage=2, freq=30, wait=60, tc=12, **algorithm_parameters):
	'''
	Inputs
	------ 
	algorithm: str
		Specifies electrode selection agolrithm. eg 'Standard' for adj-adj or 'Random' for radnom electrode placements. 
	no_electrodes: int
		Number of electrodes attached to sample
	max_measurements: int
		Maximum voltage measurements to be taken
	measurement_electrodes: NDarray
		A 4*N array of electrode positions for all measurements. Allows user to pre-generate desired electrode positions instead of using algorithm.
		Helps to speed up when using Standard algorithm.
	voltage: float
		Voltage of lock-in driving signal in Volts rms. Default 2V.
	freq: int
		Frequency of lock-in driving signal in Hz. Default 30Hz.
	tc: int
		Time constant used by lock-in amplifer. Corresponds to OFLT command fpr SR865 lock-in.
		0->1us, 1->3us, 2->10us, 3->30us, 4->100us, 5->300us,... 20->10ks, 21->30ks.
		Default 12->1s.
	wait: int	
		Time to wait between measurements divided by (1/f) of driving frequency ie no. of periods.
		Default 60, ie 2s for 30Hz.  
	print_status: bool
		Sets whether to print status messages
	algorithm_parameters: **kwargs
		Allows user to pass relevant parameters of desired algorithm
	
	Outputs
	------ 
	v_difference: NDarray
		1*N float array of all voltage measurements taken
	flick_times_np: NDarray 
		Float array of all time durations during which a switch command was executed
	get_times_np: NDarray
		Float array of all time durations during which a lock-in command was executed
	'''
	if print_status:
		print("starting EIT...")
		start = time.time()

	ClearSwitches()
	SetMeasurementParameters(["X","THeta","XNoise","FInt"])
	lockin.write("SLVL " + str(voltage))
	lockin.write("FREQ " + str(freq)) 	# frequency
	lockin.write("OFLT " + str(tc)) # time constant 11 = 300ms, 12 = 1s
	lockin.write("PHAS 0") # set phase offset to 0
	lockin.write("ASCL") # autoscale
	#standard_measurement_electrodes = Standard(no_electrodes=no_electrodes, step=1,parser='fmmu')
	#print(standard_measurement_electrodes)	
	v_diff = []
	electrode_posns =[]
	flick_times = []
	get_times = []
	keep_measuring = True

	while keep_measuring == True:
		for i in range(0,max_measurements):
			print("Measurement", i)
			next_electrodes, keep_measuring = GetNextElectrodes(algorithm=algorithm, no_electrodes=no_electrodes, measurement=i, all_measurement_electrodes = None, **algorithm_parameters)
			#print("measurement "+str(i)+", next electrode "+str(next_electrodes)+"keep measuring:"+str(keep_measuring))
			if keep_measuring == False:
				break
			#print(next_electrodes)
			start_clear = time.time()
			ClearSwitches()
			end_clear = time.time()
			clear_time = end_clear - start_clear
			flick_times.append(clear_time)
			#next_electrodes = np.random.randint(no_electrodes, size=(2,4))
			try:
				next_electrodes_shape = (next_electrodes.shape[0],  next_electrodes.shape[1])
			except IndexError:
				next_electrodes_shape = (1, next_electrodes.shape[0])
			'''
			try: 
				print("next .shpae", next_electrodes.shape[1])
			except IndexError:
				print("index error")
			print("next electrode shape", next_electrodes_shape)
			'''
			for i in range(0, next_electrodes_shape[0]):
				for j in range(0, next_electrodes_shape[1]):
					try:
						module, relay = MapSwitches(electrode=next_electrodes[i][j], lockin_connection=i)
					except IndexError:
						module, relay = MapSwitches(electrode=next_electrodes[j], lockin_connection=j)
						#print("next electrodes, j ", next_electrodes[j])

					start_flick = time.time()
					#print("module", module)
					#print("relay", relay)
					FlickSwitch('on', module, relay)
					end_flick = time.time()
					flick_times.append(end_flick - start_flick)
				start_get =time.time()
				#switch_status = switch.query('S')
				#print(switch_status)
				time.sleep(wait * (1/freq)) # Wait to let lockin settle down - may not be nesceaary
				x, theta, xnoise, fint = GetMeasurement(param_set=False)
				end_get = time.time()
				get_time = end_get - start_get
				get_times.append(get_time)
				v_diff.append(x)
				#print("i", i)
				#print('next electrodse[j]', next_electrodes[i])
				
				try:
					electrode_posns.append(next_electrodes[i,:])
				except IndexError:
					electrode_posns.append(next_electrodes)
		
		
		v_difference = np.array(v_diff)
		electrode_positions = np.array(electrode_posns)
		flick_times_np = np.array(flick_times)
		get_times_np = np.array(get_times)
		break

	if print_status:
		end = time.time()
		duration = end - start
		no_voltages = len(v_difference)
		average_time = duration / no_voltages
		print("EIT finished")
		print("Voltages: ", v_difference)
		print("Positions:", electrode_positions)
		print(str(no_voltages)+" measurements taken in "+str(duration)+" seconds.")
		print("Average time for measurement: ", average_time)
		total_switch_time = np.sum(flick_times_np)
		average_switch_time = np.mean(flick_times_np)

		print("Switch commands: ", len(flick_times_np))
		print("Total switch time", total_switch_time)
		print("Average switch time", average_switch_time)

		total_lockin_time = np.sum(get_times_np)
		average_lockin_time = np.mean(get_times_np)

		print("Lock-in commands: ", len(get_times_np))
		print("Total lock-in time", total_lockin_time)
		print("Average lock-in time", average_lockin_time)

	return v_difference, electrode_positions, flick_times_np, get_times_np

def TimeStamp(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.now().strftime(fmt).format(fname=fname)

def SaveEIT(positions, voltages, filename):

	print("Saving...")
	filename = TimeStamp(filename+".csv")
	data = [positions[:,0], positions[:,1], positions[:,2] ,positions[:,3], voltages]
	data = np.asarray(data).T
	np.savetxt(filename, data, fmt=['%i', '%i', '%i', '%i', '%e'], delimiter=",", header="sin+,sin-,v_high,v_low,voltage", comments="")
	print("File saved as", str(filename))

	return filename

if  __name__ == "__main__":
	voltages, positions, switch_times, lockin_times = RunEIT(no_electrodes=32, algorithm='Standard', freq=30,tc=13,wait=300,step=16, dist=16)

	SaveEIT(positions, voltages, "resitor_grid_30Hz-tc13-wait300-step16-dist16")
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