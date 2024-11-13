import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def generate_synthetic_data():
    np.random.seed(0)
    time_steps = 1000
    time = np.arange(time_steps)
    np.arange(0,time_steps)
    #generate temp-time series data
    temperature = 50+ np.sin(0.1*time) + np.random.normal(0,0.5,time_steps) 
    # added the noise to the temperature data
    vibration = 0.02 * np.sin(0.2* time) + np.random.normal(0,0.005,time_steps)
    # added the noise to the vibration data
    #introduce a fault in the system
    fault_start = 700
    fault_label = np.zeros(time_steps)
    fault_label[fault_start:] = 1
    temperature[fault_start:] += 10 + 0.1 * (time[fault_start:] - fault_start)
    vibration[fault_start:] += 0.05 * (time[fault_start:] - fault_start)
    #create a pandas dataframe to store the data
    data = pd.DataFrame({ "time": time,
                          "temperature": temperature,
                          "vibration": vibration,
                          "fault_label": fault_label})
    #storing the data
    data.to_csv("synthetic_data.csv",index=False)
    #plot the data for temperature
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(data["time"],data["temperature"],color="blue")
    plt.axvline(x=fault_start, color="red", linestyle="--",label="Fault Start")
    plt.xlabel("time")
    plt.ylabel("temperature")
    plt.legend()
    #plot the data for vibration
    plt.subplot(2,1,2)
    plt.plot(data["time"],data["vibration"],color="yellow")
    plt.axvline(x=fault_start, color="red", linestyle="--",label="Fault Start")
    plt.xlabel("Time")
    plt.ylabel("vibration")
    plt.legend()

    plt.tight_layout()
    plt.savefig("synthetic_data.png")
    plt.show()
if __name__ == "__main__":
    generate_synthetic_data()



