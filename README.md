# Module for RF analysis on RF Chain

- Assumes all data is within data
- Scope data is in CSV format with corresponding .Wfm.csv file
- VNA data is a s2p file
- SURF data is within file of it's name, there are 1000 files and they are pickle files

-data
--Scope_Data
---FullChain_{3-digit Channel}.csv
---FullChain_{3-digit Channel}.Wfm.csv
--SURF_Data
---SURF{Surf name}
--VNA_Data
---fullchain_{3-digit Channel}.s2p


### Data Extractors

All data is either VNA_Data, Scope_Data or SURF_Data

### RF measurements

There are RF_<> classes which automatically get the RF data given a 3-digit channel.

For the SURF data the SURF channel is in SURF_Average


### Prerequisites

skrf, pypdf

pip install scikit-rf