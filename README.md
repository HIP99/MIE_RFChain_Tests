# Module for RF analysis on RF Chain

- Assumes all data is within data
- Scope data is in CSV format with corresponding .Wfm.csv file
- VNA data is a s2p file
- SURF data is within file of it's name, there are 1000 files and they are pickle files

### Data Extractors

All data is either VNA_Data, Scope_Data or SURF_Data

### RF measurements

There are RF_<> classes which automatically get the RF data given a 3-digit channel.

For the SURF data the SURF channel is in SURF_Average