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

To execute the KWH anomaly detection algorithms run the following command from the terminal

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

![SD_ss](https://user-images.githubusercontent.com/35930061/130512458-dba51e30-c2fe-41a6-9276-03aee9593253.png)

Each point on the top plot represents a meter. The coordinates are the anomalies scores computed by the approach with two
different hyper-parameters. For example, ```er1``` in the neural network visualization was computed with a 3-layer neural 
network; ```er2``` with a 5-layer neural network. When hovering over each meter, the plots below will display the KWH curve
for that meter as well as the error curve in the case of neural network. Additionally, the scores can also be sorted by 
ranking by clicking on the top bar and selecting ```Rankings``` instead of ```Errors```.

> These programs compute the anomaly scores and display them interactively, but they do not do the thresholding for selecting
> what meters are anomalies and what meters are not.

## Failure prediction

To run the failure prediction algorithms run the following command from the terminal

```PowerShell
python run_fail_pred.py <name>
```

```<name>``` can be ```bus_N``` or ```bus_B```.

By default both failure prediction algorithms will be executed, that is: event counts and autoencoder. The outputs are the roc
curves images that get saved in the ```figures``` folder. For other results the original code can be modified. The autoencoder
uses KWH as opposed to events, which it what was presented in the report.

*when choosing the period over which to count events in the event counts algorithm make sure there are enough failing meters
that failed during that interval. Otherwise that class could be empty, in which the algorithm might throw an error, or give
errroneous results*
