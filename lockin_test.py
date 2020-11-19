import pyvisa
import numpy as np
from numpy import random
import time
from datetime import datetime
from selection_algorithms import *
from component_control import SetMeasurementParameters, MapSwitches, FlickSwitch, ClearSwitches, GetMeasurement, TimeStamp

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

SetMeasurementParameters("X, THeta, XNoise, FInt")

def Loop(freq, tc):
    lockin.write("FREQ" + str(freq))
    lockin.write("OFLT" + str(tc))
    lockin.write("ASCL")
    lockin.write("APHS")

    x, theta, xnoise, fint = GetMeasurement(param_set=False)

    return x, theta, xnoise, fint

def Test(freqs, tc):

    x = []
    theta = []
    xnoise = []
    fint = []
    for i in range(0, len(freqs)):

        data = Loop(freqs[i], tc)
        x.append(data[0])
        theta.append(data[1])
        xnoise.append(data[2])
        fint.append(data[3])
    return x, theta, xnoise, fint

def SideInidices(no_electrodes):

    posns = np.zeros((no_electrodes,4))

    for j in range(0, no_electrodes):
        posns[j,0] = j
        posns[j,1] = (j+3) % 32
        posns[j,2] = (j+1) % 32 
        posns[j,3] = (j+2) % 32

    return posns


print(SideInidices(32))

def Measure(posns, voltage=2, freq=30, tc=11, wait=60):

    SetMeasurementParameters(["X","THeta","XNOise","FInt"])
    lockin.write("SLVL " + str(voltage))
    lockin.write("FREQ " + str(freq)) #frequency
    lockin.write("OFLT " + str(tc)) #time constant 11 = 300ms
    lockin.write("PHAS 0") #autophase
    lockin.write("ASCL") #autoscale
    

    x=[]
    theta = []
    xnoise = []
    fint = []

    for i in range(0, len(posns)):

        ClearSwitches()
        #print("switches cleared")

        for j in range(0, 4):
            electrode = int(posns[i][j])
            lockin_connection= j
            #print("electrode", electrode)
            #print("lockin connect", lockin_connection)
            module, relay = MapSwitches(electrode = electrode, lockin_connection=lockin_connection)
            #print("module", module)
            #print("relay", relay)
            FlickSwitch(state=1, module=module, relay=relay)
            #print("switch status", switch.query('S'))
        
        print("Waiting...")
        time.sleep(wait * (1/freq))
        print("Taking measurement", i)
        data = GetMeasurement(param_set=False)
        print("Measurement "+str(i)+" done.")
        #print("data", data)
        x.append(data[0])
        theta.append(data[1])
        xnoise.append(data[2])
        fint.append(data[3])
        

    return x, theta, xnoise, fint

def FreqSweep(posns, freqs, tcs, voltage, wait=60):

    x = np.zeros((len(posns), len(freqs)))
    theta = np.zeros((len(posns), len(freqs)))
    xnoise = np.zeros((len(posns), len(freqs)))
    fint = np.zeros((len(posns), len(freqs)))

    for i in range(0, len(freqs)):

        x[:,i], theta[:,i], xnoise[:,i], fint[:,i] = Measure(posns = posns, voltage=voltage, freq=freqs[i], tc=tcs[i], wait=wait)

    return x, theta, xnoise, fint




if __name__ == "__main__":

    frequencies = np.array( [5, 10,20,30,40,50,70,1e2,1.5e2,2e2,3e2,6e2,1e3,2e3,3e3,1e4,3e4,1e5,3e5])
    tcs = np.array(         [13,13,12,12,11,11,11,11, 10,   10, 10, 9,  9,  8,  8,  7,  6,  5,  4])

    #frequencies = np.array([10,30,1e2,3e2,1e3,3e3,1e4,3e4,1e5,3e5])
    #tcs =         np.array([13,12,11, 10, 9,  8,  7,  6,  5,  4])

    #frequencies = np.array([1e5,3e5])
    #tcs = np.array([5,4])
    
    wait = 60 #time to wait between measurements divided by (1/f) of driving frequency ie no. of periods

    voltage = 2
    positions = SideInidices(32)

    x, theta, xnoise, fint = FreqSweep(posns=positions, freqs=frequencies, tcs=tcs, voltage=voltage, wait=wait)
    
    filename = "freq_sweep-"
    for i in range(0, len(frequencies)):
        filename_csv="freq_sweep-"+str(frequencies[i])+"Hz"
        filename_csv = TimeStamp(filename+".csv")
        data = [x[:,i], theta[:,i],  xnoise[:,i], fint[:,i], positions[:,0], positions[:,1], positions[:,2], positions[:,3]]
        data = np.asarray(data).T
        np.savetxt(filename_csv, data, fmt=['%e', '%e', '%e', '%e', '%i', '%i', '%i', '%i'], delimiter=",", header="[x,theta, xnoise,fint,sin+,sin-,v+,v-]", comments=(str(frequencies[i])+"Hz wait="+str(wait)+"periods"+" tc="+str(tcs[i])))
    
    filename_npz = TimeStamp(filename+".npz")
    np.savez(filename_npz, x=x, theta=theta, xnoise=xnoise, fint=fint, posns=positions, tcs=tcs, freqs=frequencies)
    freq_sweep_data = np.load(filename_npz)
    print(filename_npz)
    print(freq_sweep_data['x'][:,0])

    
'''
    


v_out = 2 #2V input voltage
shunt_resistor = 100e3
frequency = 100 # 30Hz frequency
tc = 11 #time constant 13->3s, 12->1s, 11->300ms, 10->100ms
positions = SideInidices(32)
results = Measure(positions, voltage=v_out, freq=frequency, tc=tc)
print(results)
voltages = np.asarray(results[0])
print("voltages")
print(voltages)
noise = np.asarray(results[2])
voltages_percantege_err = (noise/voltages) *100

current = v_out / shunt_resistor
resistances = voltages / current
print("Resistances")
for i in range(0, len(resistances)):
    print(str(resistances[i])+"+/-"+str(voltages_percantege_err[i])+"%")

print(switch.query('S'))
ClearSwitches()
print(switch.query('S'))
posns = np.array([0,3,1,2])
for j in range(0, 4):

    module, relay = MapSwitches(electrode = posns[j], lockin_connection=j)
    FlickSwitch("on", module, relay)
print(switch.query('S'))
'''