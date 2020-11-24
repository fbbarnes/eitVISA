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

def PlotSweepSingleXY(filename, point):
    
    sweep_data = np.load(filename)
    freqs = sweep_data['freqs']
    x = np.abs(sweep_data['x'][point,:])
    y = np.abs(sweep_data['y'][point,:])
    cos = (x)/np.sqrt(x**2+y**2)
    ratio1 = (x-y)/(x+y)

    print("x", x)
    print("y", y)
    print("cos", cos)


    fig, ax = plt.subplots()
    
    ax.scatter(freqs, x, marker='x', color='blue', label='x')
    #ax.set_xlim(0, 3e5)
    #ax.set_ylim(170, 180)
    ax.set_xlabel("Frequency (Hz)")
    ax.scatter(freqs, y, marker='x', color='orange', label='y')
    ax.tick_params(which='both', width=2)
    ax.legend()
    ax.grid(True)

    fig2, ax2 = plt.subplots()
    #ax2.set_ylim()
    
    ax2.set_ylabel("")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.scatter(freqs, cos, marker='x', color='red', label='cosθ = x/sqrt(x^2 + y^2)')
    ax2.scatter(freqs, ratio1, marker='x', color='green', label='(x-y)/(x+y)')
    ax2.tick_params(which='both', width=2)

    ax2.legend()
    ax2.grid(True)

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

    ax.legend()
    ax.grid(True)

    plt.show()






filename = "2020-11-24-18-17-38_freq_sweep-.npz"
PlotSweep(filename)
#PlotSweepSingle(filename, 0)
#PlotSweepSingleAverage(filename)
PlotSweepSingleXY(filename, 0)