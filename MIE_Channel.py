import numpy as np
from pathlib import Path
import pandas as pd

class MIE_Channel():
    """
    Has some basic framework for information for any measurement taken through an MIE channel
    """
    def __init__(self, channel:str = None, surf:str = None, ampa:str = None, *args, **kwargs):
        self.info = {"Channel" : channel, "AMPA" : ampa, "SURF Channel" : surf}
        self.get_info(channel, surf, ampa)

    def __str__(self):
        return (
            f"Channel: {self.info['Channel']}\n"
            f"AMPA: {self.info['AMPA']}\n"
            f"Antenna: {self.info['Antenna']}\n"
            f"Phi Sector: {self.info['Phi Sector']}\n"
            f"SURF Channel: {self.info['SURF Channel']}"
        )
    
    @property
    def Channel(self):
        return self.info["Channel"]
    
    @property
    def AMPA(self):
        return self.info["AMPA"]
    
    @property
    def Antenna(self):
        return str(self.info["Antenna"])[0]
    
    @property
    def Phi(self):
        return self.info["Phi Sector"]
    
    @property
    def SURF(self):
        return self.info["SURF Channel"]
    
    @property
    def Polarisation(self):
        return self.info["SURF Channel"][1]
    
    @property
    def SURF_Unit(self):
        return self.info["SURF Channel"][0]

    def get_info(self, channel:str = None, surf:str = None, ampa:str = None):
        """
        Gets all channel infomation based on Channel, AMPA or SURF info
        """
        current_dir = Path(__file__).parent
        filepath = current_dir / 'Channel_Assignment.csv'

        df = pd.read_csv(filepath)

        if channel:
            row = df[df['Channel'] == int(channel)]
        elif surf:
            row = df[df['SURF Channel'] == surf]
        elif ampa:
            row = df[df['Channel'] == ampa]

        if not row.empty:
            self.info = row.iloc[0].to_dict()
        else:
            print(f"Value {channel} not found in the 'SURF Channel' column.")


if __name__ == '__main__':
    info = {'channel':None, 'surf':'LV6', 'AMPA':'A186'}
    channel = MIE_Channel(**info)

    print(channel.Polarisation)