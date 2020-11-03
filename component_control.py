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


#create resource manager using py-visa backend ('@py') leave empty for NI VISA
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
#reset lock-in
TBSTAT = lockin.query('TBSTAT?') #Query the current 10 MHz timebase ext (0) or int (1)
print("TBSTAT:", TBSTAT)
#INSERT CODE HERE

