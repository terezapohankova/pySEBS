import os
import sys

import numpy as np
import pint
import rasterio as rio
from affine import Affine
from metpy.units import units
from osgeo import gdal

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
from metpy.calc import (saturation_mixing_ratio,
                        specific_humidity_from_mixing_ratio, vapor_pressure)


def saveimg(image_georef, img_new, outputPath, epsg):
    step1 = gdal.Open(image_georef, gdal.GA_ReadOnly) 
    GT_input = step1.GetGeoTransform()
    afn = Affine.from_gdal(* GT_input)
    new_dataset = rio.open(outputPath, "w", 
        driver = "ENVI",
        height = img_new.shape[0],
        width = img_new.shape[1],
        count = 1,
        nodata = -9999, # optinal value for nodata
        dtype = img_new.dtype,
        crs = epsg, # driver for coordinate system code
        
        # upper left, pixel size
        # rasterio.transform.from_origin(west, north, xsize, ysize) -> affine transformation 
        transform=afn)
        #transform = rasterio.transform.from_origin(656265, 5503485, 30, 30)) 
    new_dataset.write(img_new, 1)
    new_dataset.close()
    return


def fvc(ndvi):
    fvc = ndvi - 0.05
    return fvc

def e0(es_t_wet, t_wet, t_dry, pressure):
    #e0 = (0.000662*pressure) * 2.7183 ** (17.27 * temperature / (temperature + 237.3))
    e0 = es_t_wet - 0.000662 * (pressure) * (t_dry - t_wet) 
    return e0

def es(temperature):
    #satVapPress = (e0_temp_max + e0_temp_min) / 2
    satVapPress = 0.6108 * (2.7183 ** ((17.27 * temperature) / (temperature + 237.3)))
    return satVapPress

def dry_air_pressure(pressure, water_vapour):
    dry_air_press = pressure - water_vapour
    return dry_air_press

def es_slope(temperature):
    slopevappress =  (4098 * (0.6108 * 2.7183 ** ((17.27 * temperature) / (temperature + 237.3)))/(temperature + 237.3) ** 2)
    return slopevappress

def psychrometric_vapor_pressure_wet(satvappres, my_press_ref, tair_dry4, my_t_ref, psychrometer_coefficient = None):
    if psychrometer_coefficient is None:
        psychrometer_coefficient = 6.21e-4
        return satvappres - psychrometer_coefficient *((my_press_ref / 1000)) *(tair_dry4 - my_t_ref)

def relative_humidity(sat_vap_press, vapor_pressure):
    RH = 100 * (sat_vap_press / vapor_pressure)
    return RH

def dewpoint(temperature, rel_humidity):
    # Magnus-Tetens formula
    # 17.27 - Magnus coefficient
    # 243.5 - Magnus Coefficient -> https://journals.ametsoc.org/view/journals/apme/35/4/1520-0450_1996_035_0601_imfaos_2_0_co_2.xml
    dew = temperature - ((100 - rel_humidity)/5.)
    return  dew

#specific humidity
def SH(pressure, vapor_pressure):
    spec_hum = (.622 * pressure / (pressure - vapor_pressure) * 100)
    return spec_hum

def SWnet(SWIN, albedo):
    swnet = (1 - albedo) * SWIN
    return swnet

def LWnet(lse, LWIN, B_constant, LST):
    lwnet = (lse * LWIN) - (lse * B_constant * (LST ** 4))
    return lwnet

def G(Rn, fvc):
    G0 = Rn * (0.05 + (1 - fvc) * (0.315 - 0.05))
    return G0

def airdenstiy(dry_air_press, gas_constant, temperature_K, water_vapour):
    airdens =  dry_air_press / (gas_constant * temperature_K) + (water_vapour / (461.495  * temperature_K)) 
    return airdens

#psychrometric const
def gamma(press, Cp, l_lambda, e):
    psychro = (Cp  * press) / (e * l_lambda)
    return psychro

# zero plane displacement
def d(cropheight):
    d_hc = 0.667 * cropheight
    return d_hc

# friction velocity, # https://inis.iaea.org/collection/NCLCollectionStore/_Public/37/118/37118528.pdf
def u_friction(wind_sp, karmann, measurement_height, d, z0m):
    u_ = (wind_sp * karmann) / np.log((measurement_height-d)/z0m) 
    return u_

def u_pbl(wind_sp, height_measure, z0m, z_pbl, zero_d):  # wind speed, height of wind speed measurement, z0m, z_pbl
    """Calculates Planetary Boundary Layer wind speed [m s-1] from Fcover"""
    
    u_c = np.log((z_pbl - zero_d) / z0m) / np.log((height_measure - zero_d) / z0m)
    upbl = wind_sp * u_c
    u_pbl = np.where(upbl < 0, 0,upbl) 
    return u_pbl

def PSIh_y(Y, psy_c, psy_d, psy_n):
        # constants (Brutsaert, 1999)
        Y = abs(Y)
        PSIh_y = (1.0 - psy_d) / psy_n * np.log((psy_c + Y ** psy_n) / psy_c)
        return PSIh_y


def Cw(alfa, psi_C0, psi_C1, psi_C11, C21, psi_C22, z0m):

    ### creating conditional mask
       
    Mask = np.copy(z0m)
    Mask = np.where(Mask <= psi_C0, -9999.0, np.where(Mask != -9999.0, 0, np.where(Mask == -9999.0, 1, Mask)))

    C_1 = np.log(alfa) + psi_C11 - psi_C1
    C_2 = np.log(C21) + psi_C22 - psi_C1
    C_1_cond = np.where(C_1 == -np.inf, 0, np.where(C_1 == np.inf, 0, np.where(C_1 == None, np.nan_to_num(C_1 * Mask), C_1)))
    C_2_cond = np.where(C_2 == -np.inf, 0, np.where(C_2 == np.inf, 0, np.where(C_2 == None, np.nan_to_num(C_2 * Mask), C_2)))

    # SUM CONDITIONAL
    C = C_1_cond + C_2_cond
    C[C < 0.0] = 0
    return C

def zdh0(zd0, z0m):
    zdh = np.log(zd0 / (0.1 * z0m))

    return zdh