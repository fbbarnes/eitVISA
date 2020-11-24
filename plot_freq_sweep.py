import numpy as np
np.random.seed(19680801)
import matplotlib.pyplot as plt

def PlotSweep(filename):
    
    sweep_data = np.load(filename)
    freqs = sweep_data['freqs']



    fig, ax = plt.subplots()
    for i in range(0, len(freqs)):

        
        voltage = sweep_data['x'][:,i]
        position = np.arange(0, len(voltage))

        ax.plot(position, voltage, label="%.1e" % (freqs[i]) )

    ax.legend()
    ax.grid(True)

    plt.show()

def PlotSweepSingle(filename, point):
    
    sweep_data = np.load(filename)
    freqs = sweep_data['freqs']
    voltage = sweep_data['x'][point,:]
    theta = sweep_data['theta'][point,:]

    theta_abs = np.abs(theta)

    fig, ax = plt.subplots()
    
    ax.scatter(freqs, voltage, color='blue', label='Voltage')
    ax.set_xlim(0, 400)
    ax.set_ylim(170, 180)
    ax.set_xlabel("Frequency (Hz)")
    ax2 = ax.twinx()
    #ax2.set_ylim()
    ax2.set_ylabel("Voltage")
    ax2.scatter(freqs, theta_abs, color='orange', label='|Phase|')
    ax2.set_ylabel("Mean Phase (°)")

    #ax.legend()
    ax.grid(True)

    plt.show()

def PlotSweepSingleAverage(filename):
    
    sweep_data = np.load(filename)
    freqs = sweep_data['freqs']
    voltage = sweep_data['x']
    theta = sweep_data['theta']
    theta_abs_mean = np.mean(np.abs(theta), axis=0)
    voltage_mean = np.mean(voltage, axis=0)


    fig, ax = plt.subplots()
    
    ax.scatter(freqs, voltage_mean,  color='blue', label='Mean Voltage')
    ax.set_xlim(0, 400)
    ax.set_ylim(-0.00103, -0.00087)
    ax.set_xlabel("Frequency (Hz)", )
    ax.set_ylabel("Mean Voltage (V)", color='blue')
    ax2 = ax.twinx()
    ax2.set_ylim(174, 180)
    ax2.set_ylabel("Mean Phase (°)", color='orange')    
    ax2.scatter(freqs, theta_abs_mean, color='orange', label='Mean |Phase|')    

    #ax.legend()
    ax.grid(True)

    plt.show()


filename = "2020-11-19-17-57-23_freq_sweep-.npz"
PlotSweep(filename)
PlotSweepSingle(filename, 0)
PlotSweepSingleAverage(filename)