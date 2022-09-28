# IoT-Device-Fingerprinting
## <img src="https://latex.codecogs.com/gif.latex?\mathcal{M}_\text{seq}" /> Classifier
### Data required
Network traces are required to train the iPet generators.

`pcap` packet capture files must be used to generate corresponding `csv` files that contain the specific fields (stated below). They must be placed in the `data/original-trace` directory. Note that the rows in the `csv` file correspond to packet metadata for a particular device communicating with one or more servers on the internet. We decompose the `pcap` file contains traffic from multiple devices for each day into separate `csv` files. The file may follow a naming convention `day[x].csv`.

The reqiured fields are:
 - `frame.number`	
 - `frame.time_relative`	
 - `ip.src`	
 - `ip.dst`	
 - `ip.proto`	
 - `tcp.len`	
 - `tcp.stream`	
 - `udp.length`	
 - `eth.src`	
 - `eth.dst`	
 - `transport.len`

A sample of the network traces in the expected format has been shared [here](https://drive.google.com/drive/folders/1gRkcrPupkYTWvYlgkkDKDmsP2FsJzG-g?usp=sharing).

### Running
#### Training Generators

##### Configuration
To allow a user to customise their iPet instance, we expect them to specify the following variables in `constants.py`:
- `total_time`: The total observation time for the time series, in seconds.
- `omega` : The duration of a discrete time-slot in the time series, in seconds.
- `device_name` : Name list of the devices in the network. For e.g. `['device_A','device_B,'device_C']`

##### Generating Fetaure Vectors
The raw data is converted to numpy feature vectors for the model to training on by running the script:
```sh
$ python Feature_Generation_Mseq.py 
```

The feature vectors generated are saved in the ```data/Traces_Mseq``` directory

##### Training  <img src="https://latex.codecogs.com/gif.latex?\mathcal{M}_\text{seq}" />  Classifier
We train the sequential fingerprinting model using the following script: 
```sh
$ python Train_Mseq.py 
```
The trained model is saved in the ```Models/M_seq_fingeprinting``` directory

## <img src="https://latex.codecogs.com/gif.latex?\mathcal{M}_\text{agg}" /> Classifier

To be added