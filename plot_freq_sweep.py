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
    
    ax.scatter(freqs, voltage, label='Voltage')
    ax.scatter(freqs, theta_abs, label='|Phase|')

    ax.legend()
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
    
    ax.scatter(freqs, voltage_mean, label='Mean Voltage')
    ax.scatter(freqs, theta_abs_mean, label='Mean |Phase|')

    ax.legend()
    ax.grid(True)

    plt.show()


filename = "2020-11-19-17-57-23_freq_sweep-.npz"
PlotSweep(filename)
PlotSweepSingle(filename, 0)
PlotSweepSingleAverage(filename)