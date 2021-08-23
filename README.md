# WEenergies-results
This repository contains all the code corresponding to the Faulty Meters Study.

The code runs on python 3. The necessary packages are: numpy, torch, sparse, dash, matplotlib, pandas, scipy,
plotly, sklearn.

```data``` contains the clean kwh files and compressed events files.
```results``` is the folder where the arrays generated from ```run_anom_detection.py``` will be saved.


## Documentation

To access the documentation go to ```docs/_build/html/```, then open ```index.html```, and that displays the documentation. 
Once the file is open is going to look like this:

![docs](https://user-images.githubusercontent.com/35930061/130507108-46297191-2afb-406a-85ee-9aac77d49c45.png)

That contains all functions and classes with what they do. There is no documentation for the ```visualization``` files and 
the ```run``` files because they do not contain function themselves, but instead are calling all the other functions. 


## Anomaly detection

To run the KWH anomaly detection algorithms run the following command from the terminal

```PowerShell
python run_anom_detection.py <filename>
```
```<filename>``` is located in data. With the current data it can be ```bus_B_reads.npy``` or ```bus_N_reads.npy```. 

By default it uses the three detection algorithms: general anomaly detection, slowing down meters, and frequency analysis.
The output, which is the anomaly scores, will be saved to results as well as the data. These can be visualized with 
```visualization_nns_anom_detection.py``` for general anomalies, and ```visuzaliation_SD_freq_meters.py``` for slowing down meters and frequency analysis.

To run the visualizations run the following commands from the terminal

```PowerShell
python visualization_nns_anom_detection.py
```

and the slowing down meters and frequency analysis visualization

```PowerShell
python visualization_SD_freq_meters.py <filename>
```

```<filename>``` is ```SDs.npy``` for slowing down meters, and ```freqs.npy``` for frequency analysis.

The visualizations look like this respectively: 

![visual_nn](https://user-images.githubusercontent.com/35930061/130512364-87229092-e5d3-47f3-94e8-bd395141da10.png)

and 


