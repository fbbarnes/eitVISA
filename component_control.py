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
lockin.read_termination = '\n'
lockin.read_termination = '\n'

#initialise devices
#open all switches
switch.write('C') #C = Clear = open all relays or turn OFF all relays
switch_status = switch.query('S') #S = Status which can be used on individual switch points, modules, or the entire system. Replies with string eg 0000000 showing 8 closed switches
print(switch_status)
switch.write('L3 5')
print(switch.query('S'))

#reset lock-in
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

'''
#set channels
lockin.write("COUT OCH1, XY") #Set CH1 to X
lockin.write("COUT OCH2, XY") #Set CH2 to Y

x = lockin.query("COUT? OCH1") #Returns the CH1 output X
y = lockin.query("COUT? OCH2") #Returns the CH2 output Y

print('x:', x)
print('y:', y)



lockin.write("COUT OCH1, RTheta") #Set CH1 to X
lockin.write("COUT OCH2, RTheta") #Set CH2 to Y

r = lockin.query("COUT? OCH1") #Returns the CH1 output X
theta = lockin.query("COUT? OCH2") #Returns the CH2 output Y

print('r:', x)
print('theta:', y)
'''

lockin.write("ARNG") #auto range 
lockin.write("ASCL") #auto scale

#CDSP

#get X
x = lockin.query('OUTP? X')
print("X:",x)

#get Y
y = lockin.query('OUTP? Y')
print("Y:",y)

#get R
r = lockin.query('OUTP? R')
print("R:",r)

#get THETA
theta = lockin.query('OUTP? theta')
print("THETA:",theta)



#INSERT CODE HERE

