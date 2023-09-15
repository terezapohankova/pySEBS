import os
import sys
from pprint import pprint

import numpy as np
import pandas as pd
import rasterio as rio
import scipy
import scipy.constants
from osgeo import gdal
from rasterio.plot import show

from valencia_fcs import *

#######################################################################################################
#######################################################################################################
#######################################################################################################
# SEBS
## define inputs
#FOLDERS
INPUT_FOLDER = r'/home/tereza/ownCloud/PhD/staze/valencia2023/work/SEBS/20180718_1155_LST_EMISIVIDAD_ALBEDO_NDVI/'
OUTPUT_FOLDER = r'/home/tereza/ownCloud/PhD/staze/valencia2023/work/SEBS/20180718_1155_LST_EMISIVIDAD_ALBEDO_NDVI/TEST_OUTPUT/'
FOLDER_METEO = r'/home/tereza/ownCloud/PhD/staze/valencia2023/work/SEBS/fluxes_Braccagni'

## define input data names available in the INPUT FOLDER
NDVI_name = '1155NDVI.gcu'
LST_name = '1155LST.gcu'
ALBEDO_name = '1155albedo.gcu'
LSE_name = '1155LSE.gcu'
Rnet = '1155_Rnet.gcu'
Fvc = 'fvc.gcu'

## define input meteorological file names available in the FOLDER_METEO
METEO = 'meteo_cornJUN_JUL1.xlsx'

## CONSTANTS
epsg = 'EPSG:32632'
B_constant = scipy.constants.sigma  # W m^-2 K^-4
Z = 10                              # altitude [m]
Z_m = 2                             # height of measurement above ground [m] 
R = 287.05                          # gas constant, [J/kg-K], https://designbuilder.co.uk/helpv3.4/Content/Calculation_of_Air_Density.htm
Cp = 0.001013                       # specific heat at constant pressure [MJ kg-1 °C-1]
l_lambda =  2.45                    # latent heat of vaporization [MJ kg-1]
e =  0.622                          # ratio molecular weight of water vapour/dry air 
k = 0.41                            # von karmann
g = 9.81                            # grativational constant
z_pbl = 1000                        # height of planetary boundary layer [m]

# PSI_ coeffs       ?????
coeff_a = 1
coeff_b = 0.667
coeff_c = 5.0
coeff_d = 0.35

psy_c = 0.33
psy_d = 0.057
psy_n = 0.78

alfa = 0.12
beta = 125.0

ignore_zero = np.seterr(all="ignore")

##open LST raster
with rio.open(os.path.join(INPUT_FOLDER, LST_name)) as lst_raster:
    lst_b1 = lst_raster.read(1, masked = False) # Kelvin
    profile = lst_raster.profile

## open NDVI
with rio.open(os.path.join(INPUT_FOLDER, NDVI_name)) as ndvi_raster:
    ndvi_b1 = ndvi_raster.read(1, masked = False)
    profile = ndvi_raster.profile

## open emissivity
with rio.open(os.path.join(INPUT_FOLDER, LSE_name)) as emissivity_raster:
    lse_b1 = emissivity_raster.read(1, masked = False) # "W/m^2"
    profile = emissivity_raster.profile

## open albedo
with rio.open(os.path.join(INPUT_FOLDER, ALBEDO_name)) as albedo_raster:
    albedo_b1 = albedo_raster.read(1, masked = False)
    profile = albedo_raster.profile

# open meteorology
rows_to_read = [0,13681]
## 0 - dry bulb     2 - wind        4 - SWIN
## 1 - wet bulb     3 - pressure    5- LWIN
#skiprows = lambda x: x not in rows_to_read
meteo_df = pd.read_excel(os.path.join(FOLDER_METEO, 'meteo_cornJUN_JUL1.xlsx'), sheet_name='meteo_cornJUN_JUL', 
                   usecols = ['TIMESTAMP', 'Tair_dry4', 'Tair_wet4', 'wind4', 'pressure', 'SWIN', 'LWIN'], index_col= 'TIMESTAMP')

# get T_AIR_WET
my_t_ref = meteo_df.iloc[13681,1] # Tair_wet4, 18/7/20218 12:00 # °C
my_t_ref_K = my_t_ref + 273.15 # K
pprint(f"Wet Bulb Temperature: {my_t_ref} °C.")

# get T_AIR_DRY
tair_dry4 = meteo_df.iloc[13681,0] # °C
pprint(f"Dry Bulb Temperature: {tair_dry4} °C.")
tair_dry4_max = meteo_df['Tair_dry4'].max()
pprint(f"Maximum dry bulb temperature Pressure: {tair_dry4_max} °C.")
tair_dry4_min = meteo_df['Tair_dry4'].min()
pprint(f"Minimum dry bulb temperature Pressure: {tair_dry4_min} °C.")


# get W
my_w_ref = meteo_df.iloc[13681,2] # m/s
pprint(f"Wind Speed: {my_w_ref} m/s.")

# Pressure
my_press_ref = 100858.1117 # Pa
my_press_ref_kpa = my_press_ref / 1000
pprint(f"Surface Pressure: , {my_press_ref} Pa.")

my_press_surf = meteo_df.iloc[13681,3] # kPa
pprint(f"Surface Pressure: {my_press_surf} kPa.") # ???? UNITS

press_surf_array = np.full_like(lst_b1, my_press_surf)

# create array in shape of LST with values of Tair_wet4
t_ref_array = np.full_like(lst_b1, my_t_ref)
w_ref_array = np.full_like(lst_b1, my_w_ref)
press_ref_array = np.full_like(lst_b1, my_press_ref) # Pa

# pressure ### WHICH UNITS???
sat_pressure_0c = 611.2 # Pa

sat_vap_press = es(tair_dry4) # kPa
pprint(pprint(f"Saturated Vapour Pressure: {sat_vap_press} kPa."))

slope_vap_press =  es_slope(my_t_ref)  # kPa/°C
pprint(f"Slope of saturation vapour pressure curve : {slope_vap_press} kPa/°C.")

my_ea_ref = psychrometric_vapor_pressure_wet(sat_vap_press, my_press_ref_kpa, tair_dry4, my_t_ref)
my_ea_ref_array = np.full_like(lst_b1, my_ea_ref) # Pa
saveimg(os.path.join(INPUT_FOLDER, LST_name), my_ea_ref_array, os.path.join(OUTPUT_FOLDER, 'psychro_wet_1155.GCU'))
pprint(f"Wet Psychrometric Vapour Pressure: {my_ea_ref} kPa.")

# Warter vapour pressure
water_vapour = e0(es(my_t_ref), my_t_ref, tair_dry4, my_press_ref_kpa) # kPa
pprint(f"Actual Water Vapour pressure: {water_vapour} kPa.")

# Humidity
RH = relative_humidity(sat_vap_press, water_vapour) # %
pprint(f"Relative Humidity: {RH} %.")
#dew point
dew = dewpoint(my_t_ref, RH) # °C
pprint(f"Dewpoint temperature: {dew} °C.")
# specific humidity
my_q_ref = SH(my_press_ref_kpa, water_vapour) # g/kg
pprint(f"Specific humidity: {my_q_ref} %.")
q_ref_array = np.full_like(lst_b1, my_q_ref)

# air density
pressure_dry_air = dry_air_pressure(my_press_ref, water_vapour) # kPa
air_density = airdenstiy(pressure_dry_air, R, my_t_ref_K, water_vapour) # kg/m3
pprint(f"Air density: {air_density} kg/m3.")

## Z array
z_ref_array = np.full_like(lst_b1, Z)

#psychrometric
gamma_cons = gamma(my_press_ref_kpa, Cp, l_lambda, e) # kPa °C-1
print(pprint(f"Psychrometric constant: {gamma_cons} kPa °C-1."))

## CALCULATIONS
# FVC
my_fvc = fvc(ndvi_b1)
saveimg(os.path.join(INPUT_FOLDER, LST_name), my_fvc, os.path.join(OUTPUT_FOLDER, 'fvc_1155.GCU'))

# LAI
ndvi_b1[ndvi_b1 <= 0] = 'nan'
my_lai = (ndvi_b1 * (1.0 + ndvi_b1) / (1.0 - ndvi_b1 + 1.0E-6))
saveimg(os.path.join(INPUT_FOLDER, LST_name), my_lai, os.path.join(OUTPUT_FOLDER, 'lai_1155.GCU'))

# Z0M
my_z0m = np.exp(-5.5 + 5.8 * ndvi_b1) #Tomelloso method, Bolle and Streckbach 1993
saveimg(os.path.join(INPUT_FOLDER, LST_name), my_z0m, os.path.join(OUTPUT_FOLDER, 'z0m_1155.GCU'))

# Crop Height
my_hc = my_z0m / 0.136 #Brutsaert, 1982
saveimg(os.path.join(INPUT_FOLDER, LST_name), my_hc, os.path.join(OUTPUT_FOLDER, 'hc_1155.GCU'))

## radiation
my_SWin = meteo_df.iloc[13681,4] # Downwelling shortwave radiation, W/m2
pprint(f"Downwelling shortwave radiation: {my_SWin} W/m2.")
my_LWin = meteo_df.iloc[13681,5] # Downwelling longwave radiation, W/m2
pprint(f"Downwelling longwave radiation: {my_LWin} W/m2.")

# shortwave net
my_SWnet = SWnet(my_SWin, albedo_b1) # W/m2
saveimg(os.path.join(INPUT_FOLDER, LST_name), my_SWnet, os.path.join(OUTPUT_FOLDER, 'my_SWnet_1155.GCU'))

# longwave net
my_LWnet = LWnet(lse_b1, my_LWin, B_constant, lst_b1) # W/m2
saveimg(os.path.join(INPUT_FOLDER, LST_name), my_LWnet, os.path.join(OUTPUT_FOLDER, 'my_LWnet_1155.GCU'))

#net radiation
my_Rn = my_SWnet + my_LWnet # W/m2
saveimg(os.path.join(INPUT_FOLDER, LST_name), my_Rn, os.path.join(OUTPUT_FOLDER, 'my_Rn_1155.GCU'))

# ground heat flux
my_G = G(my_Rn, my_fvc)
saveimg(os.path.join(INPUT_FOLDER, LST_name), my_G, os.path.join(OUTPUT_FOLDER, 'my_G_1155.GCU'))

# zero_plane displacement
zero_d = d(my_hc) # m
saveimg(os.path.join(INPUT_FOLDER, LST_name), zero_d, os.path.join(OUTPUT_FOLDER, 'd_1155.GCU'))

#external friction 
u_ = u_friction(my_w_ref, k, Z_m, zero_d, my_z0m) # m/s
u_ = np.where(u_ < 0, np.mean(u_), u_ )
saveimg(os.path.join(INPUT_FOLDER, LST_name), u_, os.path.join(OUTPUT_FOLDER, 'u_friction_1155.GCU'))

# height of win speed boundary planet
u_planet = windsp_pbl(my_w_ref, 2, my_z0m, 1000, zero_d) # m/s
saveimg(os.path.join(INPUT_FOLDER, LST_name), u_planet, os.path.join(OUTPUT_FOLDER, 'u_planet_1155.GCU'))
 
# ????????????????????????????
zd0 = z_pbl - zero_d
zdh = zdh0(zd0, my_z0m)
zdm = np.log(zd0 / my_z0m)
RUstar = (0.41 * u_planet) / zdm
saveimg(os.path.join(INPUT_FOLDER, LST_name), RUstar, os.path.join(OUTPUT_FOLDER, 'RUstar_1155.GCU'))

# MO length
L_w = (RUstar ** 3.0) * air_density / (0.61 * k * g * (my_Rn - my_G) / 2450000)
L_w = np.where(L_w > 3000, 3000, L_w)
saveimg(os.path.join(INPUT_FOLDER, LST_name), L_w, os.path.join(OUTPUT_FOLDER, 'L_w2_1155.GCU'))

#stability coeff ????
psi_m_stab = PSI_m_stable(coeff_a, coeff_b, coeff_c, coeff_d, u_planet, L_w)
psi_h_stab = PSI_h_stable(coeff_a, coeff_b, coeff_c, coeff_d, u_planet, L_w)

# air resistance

def airResis(Z, zero_d, z0m, psi_m_stable, z_pbl, psi_h_stable, k, wind_sp):
    ra = (np.log((Z - zero_d) / z0m) - psi_m_stable) * (np.log((z_pbl - zero_d)/(0.1 * z0m)) - psi_h_stable) / (k ** 2 * wind_sp) # s/m
    return ra
ra = airResis(Z, zero_d, my_z0m, psi_m_stab, z_pbl, psi_h_stab, k, my_w_ref)
#ra = (np.log((Z - zero_d) / my_z0m) - psi_m_stab) * (np.log((z_pbl - zero_d)/(0.1*my_z0m)) - psi_h_stab) / (k ** 2 * my_w_ref) # s/m
saveimg(os.path.join(INPUT_FOLDER, LST_name), ra, os.path.join(OUTPUT_FOLDER, 'ra_1155.GCU'))

    
# conditional masks?
C0 = (alfa / beta) * z_pbl
C1 = np.nanmin(my_z0m * 0.1) / L_w
C11 = -alfa * z_pbl / L_w
C21 = z_pbl / (beta * my_z0m)
C22 = -beta * my_z0m / L_w

psi_C0 = PSIh_y(C0, psy_c, psy_d, psy_n)
psi_C1 = PSIh_y(C1, psy_c, psy_d, psy_n)
psi_C11 = PSIh_y(C11, psy_c, psy_d, psy_n)
psi_C22 = PSIh_y(C22, psy_c, psy_d, psy_n)

C_wet2 = np.nanmin(psi_h_stab)
C_wet1 = Cw(alfa, psi_C0, psi_C1, psi_C11, C21, psi_C22, my_z0m)
C_wet = C_wet1 + C_wet2

re_w = (zdh - C_wet) / (k * RUstar)
re_w1 = zdh / (k * RUstar)
re_w2 = re_w
rew = re_w1 + re_w2

#sensible heat flux
flux_H = H(my_Rn, air_density, rew, water_vapour, sat_vap_press, slope_vap_press, gamma_cons)
saveimg(os.path.join(INPUT_FOLDER, LST_name), flux_H, os.path.join(OUTPUT_FOLDER, 'H_1155.GCU'))

# latent heat flux
delta = delta_ea_T(lst_b1, my_t_ref)
Latent = LatentH(delta, my_Rn, my_G, air_density, Cp, sat_vap_press, water_vapour, ra, gamma_cons)
saveimg(os.path.join(INPUT_FOLDER, LST_name), Latent, os.path.join(OUTPUT_FOLDER, 'LE_1155.GCU'))

# actual ET
ET_act = ET_actual(my_Rn, my_G, my_hc, l_lambda)
saveimg(os.path.join(INPUT_FOLDER, LST_name), ET_act, os.path.join(OUTPUT_FOLDER, 'ActualET_1155.GCU'))

# daily ET
ET_day = ET_daily(ET_act, l_lambda)
saveimg(os.path.join(INPUT_FOLDER, LST_name), ET_day, os.path.join(OUTPUT_FOLDER, 'DailyET_1155.GCU'))
