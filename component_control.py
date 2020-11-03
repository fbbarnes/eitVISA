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
import time

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
lockin.write('SLVL '+str(VOUT_INIT)) #Set the sine out amplitude to FREQ_INIT in Volts The amplitude may be programmed from 1 nV to 2.0V
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

