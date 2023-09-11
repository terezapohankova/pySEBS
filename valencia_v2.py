import metpy
from metpy.units import units

from metpy.calc import specific_humidity_from_mixing_ratio, saturation_mixing_ratio
import numpy as np
#import sebs_function_bank as sfb
import openpyxl as xl
import rasterio as rio
from rasterio.plot import show

import pandas as pd
import scipy
import os, sys

from valencia_fcs import *


#######################################################################################################
#######################################################################################################
#######################################################################################################

## define inputs
INPUT_FOLDER = r'/home/tereza/ownCloud/PhD/staze/valencia2023/work/SEBS/20180718_1155_LST_EMISIVIDAD_ALBEDO_NDVI/'
OUTPUT_FOLDER = r'/home/tereza/ownCloud/PhD/staze/valencia2023/work/SEBS/20180718_1155_LST_EMISIVIDAD_ALBEDO_NDVI/TEST_OUTPUT/'
NDVI_name = '1155NDVI.gcu'
LST_name = '1155LST.gcu'
ALBEDO_name = '1155albedo.gcu'
LSE_name = '1155LSE.gcu'
Rnet = '1155_Rnet.gcu'
Fvc = 'fvc.gcu'
FOLDER_METEO = r'/home/tereza/ownCloud/PhD/staze/valencia2023/work/SEBS/fluxes_Braccagni'
METEO = 'meteo_cornJUN_JUL1.xlsx'

## CONSTANTS
epsg = 'EPSG:32632'
B_constant = units.Quantity(scipy.constants.sigma, "W/m**2 ")


##open LST raster
with rio.open(os.path.join(INPUT_FOLDER, LST_name)) as lst_raster:
    lst_b1 = lst_raster.read(1, masked = False)
    profile = lst_raster.profile

## open NDVI
with rio.open(os.path.join(INPUT_FOLDER, NDVI_name)) as ndvi_raster:
    ndvi_b1 = ndvi_raster.read(1, masked = False)
    profile = ndvi_raster.profile

## open emissivity
with rio.open(os.path.join(INPUT_FOLDER, LSE_name)) as emissivity_raster:
    lse_b1 = emissivity_raster.read(1, masked = False)
    profile = emissivity_raster.profile

## open albedo
with rio.open(os.path.join(INPUT_FOLDER, ALBEDO_name)) as albedo_raster:
    albedo_b1 = albedo_raster.read(1, masked = False)
    profile = albedo_raster.profile

# open meteorology
wb = pd.read_excel(os.path.join(FOLDER_METEO, 'meteo_cornJUN_JUL1.xlsx'), sheet_name='meteo_cornJUN_JUL')
#print(wb.head())

# get T_AIR_WET
my_t_ref = pd.to_numeric(wb.iloc[13680]['Tair_wet4'], errors='coerce', downcast='float') #Tair_wet4, 18/7/20218 12:00
my_t_ref_unit = units.Quantity(my_t_ref, "°C") # specify units

# get T_AIR_DRY
tair_dry4 = pd.to_numeric(wb.iloc[13680] ['Tair_dry4'], errors='coerce', downcast='float')
tair_dry4 = units.Quantity(tair_dry4, "°C") # specify units

# get W
my_w_ref = pd.to_numeric(wb.iloc[13680]['wind4'], errors='coerce', downcast='float') #questo e il valore che mi serve, wind4 alle 12 del 18.07.2018 (m/s)
my_w_ref = units.Quantity(my_w_ref, "m/s")

# Pressure
my_press_ref = 100858.1117 #Pa, valure ottenuto da ERA5
my_press_ref = units.Quantity(my_press_ref, "Pa")
my_press_surf = pd.to_numeric(wb.iloc[13680]['pressure']) 
my_press_surf = units.Quantity(my_press_surf, "hPa")


# create array in shape of LST with values of Tair_wet4
t_ref_array = np.full_like(lst_b1, my_t_ref)
w_ref_array = np.full_like(lst_b1, my_w_ref)
press_ref_array = units.Quantity(np.full_like(lst_b1, my_press_ref), 'Pa')

# pressure ### WHICH UNITS???
sat_pressure_0c = units.Quantity(611, 'Pa')
sat_vap_press = units.Quantity(saturation_vapor_pressure(sat_pressure_0c, my_t_ref_unit), 'Pa')
my_ea_ref = units.Quantity(psychrometric_vapor_pressure_wet(sat_vap_press, my_press_ref, tair_dry4, my_t_ref_unit), 'Pa')


## CALCULATIONS
# FVC
my_fvc = fvc(ndvi_b1)
saveimg(os.path.join(INPUT_FOLDER, LST_name), my_fvc, os.path.join(OUTPUT_FOLDER, 'fvc_1155.GCU'), 'EPSG:32632')

# LAI
ndvi_b1[ndvi_b1 <= 0] = 'nan'
my_lai = (ndvi_b1 * (1.0 + ndvi_b1) / (1.0 - ndvi_b1 + 1.0E-6))
saveimg(os.path.join(INPUT_FOLDER, LST_name), my_lai, os.path.join(OUTPUT_FOLDER, 'lai_1155.GCU'), 'EPSG:32632')

# Z0M
my_z0m = np.exp(-5.5 + 5.8 * ndvi_b1) #Tomelloso method, Bolle and Streckbach 1993
saveimg(os.path.join(INPUT_FOLDER, LST_name), my_z0m, os.path.join(OUTPUT_FOLDER, 'z0m_1155.GCU'), 'EPSG:32632')

# Crop Height
my_hc = my_z0m / 0.136 #Brutsaert, 1982
saveimg(os.path.join(INPUT_FOLDER, LST_name), my_hc, os.path.join(OUTPUT_FOLDER, 'hc_1155.GCU'), 'EPSG:32632')

# dew point
dew = dewpoint(my_ea_ref, sat_pressure_0c)
#print(dew)

psychro = psychrometric_vapor_pressure_wet(sat_vap_press, press_ref_array, tair_dry4, my_t_ref_unit) #kolik cca by to mělo vyjít?
saveimg(os.path.join(INPUT_FOLDER, LST_name), psychro, os.path.join(OUTPUT_FOLDER, 'psychro_1155.GCU'), 'EPSG:32632')
#print(psychro)

# specific humidity
my_q_ref = specific_humidity_from_dewpoint(my_press_ref, dew)
my_q_ref = units.Quantity(my_q_ref.magnitude, "g/kg")
q_ref_array = np.full_like(lst_b1, my_q_ref)

## TEZKO RICT CO TO JE
my_SWin = pd.to_numeric(wb.iloc[13680]['SWIN'], errors='coerce', downcast='float') 
my_SWin = units.Quantity(my_SWin, "m/s")
my_LWin = pd.to_numeric(wb.iloc[13680]['LWIN'], errors='coerce', downcast='float') 
my_LWin = units.Quantity(my_LWin, "m/s")

my_SWnet = (1 - albedo_b1) * my_SWin
saveimg(os.path.join(INPUT_FOLDER, LST_name), my_SWnet, os.path.join(OUTPUT_FOLDER, 'my_SWnet_1155.GCU'), 'EPSG:32632')

my_LWnet = (lse_b1 * my_LWin) - (lse_b1 * B_constant * (lst_b1 ** 4))
saveimg(os.path.join(INPUT_FOLDER, LST_name), my_LWnet, os.path.join(OUTPUT_FOLDER, 'my_LWnet_1155.GCU'), 'EPSG:32632')
sys.exit()



