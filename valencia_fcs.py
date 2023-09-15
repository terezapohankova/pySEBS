import os
import sys

import numpy as np
import rasterio as rio
from affine import Affine
from metpy.units import units
from osgeo import gdal


def saveimg(image_georef, img_new, outputPath, epsg = 'EPSG:32632'):
    """ Save created image in chosen format to chosen path.

    Args:
        image_georef (string):      path to sample georeferenced image with parameters to be copied to new image
        img_new (numpy.array):      newly created image to be saved
        outputPath (str):           path including new file name to location for saving
        epsg (str):                 EPSG code for SRS (e.g. 'EPSG:32632')
    """    
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
        transform=afn)
    new_dataset.write(img_new, 1)
    new_dataset.close()
    return


def fvc(ndvi):
    """ Calculate fraction of vegetation cover based on NDVI.

    Args:
        ndvi (numpy.array): NDVI array

    Returns:
        numpy.array: fraction of vegetation cover
    """    
    fvc = ndvi - 0.05
    return fvc

def e0(es_t_wet, t_wet, t_dry, pressure):
    """ Water Vapour Pressure (actual water vapour pressure) [kPa]

    Args:
        es_t_wet (float):   Saturated vapour pressure [kPa]
        t_wet (float):      Wet Bulb Air Temperature  [°C]
        t_dry (float):      Dry Bulb Air Temperature  [°C]
        pressure (float):   Atmospheric Pressure [kPa]

    Returns:
        float: value of water vapour pressure [kPa]
    """    
    e0 = es_t_wet - 0.000662 * (pressure) * (t_dry - t_wet) 
    return e0

def es(temperature):
    """ Saturated Water Vapour Pressure [kPa]
        Based on FAO56 https://www.fao.org/3/x0490e/x0490e07.htm#TopOfPage

    Args:
        temperature (float): Air Temperature [°C]

    Returns:
        float: value of saturated water vapour pressure [kPa]
    """    
    satVapPress = 0.6108 * (2.7183 ** ((17.27 * temperature) / (temperature + 237.3)))
    return satVapPress

def dry_air_pressure(pressure, water_vapour):
    """Pressure of dry air

    Args:
        pressure (float):       Atmospheric Pressure [kPa]
        water_vapour (float):   Actual Water Vapour Pressure [kPa]

    Returns:
        float: pressure of dry air [kPa]
    """    
    dry_air_press = pressure - water_vapour
    return dry_air_press

def es_slope(temperature):
    """ Slope of the saturation vapour pressure curve [kPa]

    Args:
        temperature (float): Air Temperature [°C]

    Returns:
        float: Value  of Slope of the saturation vapour pressure curve [kPa]
    """    
    slopevappress =  (4098 * (0.6108 * 2.7183 ** ((17.27 * temperature) / (temperature + 237.3)))/(temperature + 237.3) ** 2)
    return slopevappress

def psychrometric_vapor_pressure_wet(satvappres, pressure, t_dry, t_wet, psychrometer_coefficient = None):
    """ 

    Args:
        satvappres (float):     Saturated water vapour pressure [kPa]
        pressure (float):       Atmospheric Pressure [kPa]
        t_dry (float):          Dry Bulb Air Temperature  [°C]
        t_wet (float):          Dry Bulb Air Temperature  [°C]
        psychrometer_coefficient (dimenstionless, optional): _description_. Defaults to None. ?????????????

    Returns:
        float: psychrometric vapor pressure for wet air [[kPa]
    """    
    if psychrometer_coefficient is None:
        psychrometer_coefficient = 6.21e-4
        return satvappres - psychrometer_coefficient *((pressure / 1000)) *(t_dry - t_wet)

def relative_humidity(satvappres, vappress):
    """ Relative Humidity [%]

    Args:
        satvappres (float): saturated water vapour pressure [kPa]
        vappres (float):    actual water vapour pressure [kPa]
    
    Returns:
        float: value of relative humidity [%]
    """      
    RH = 100 * (satvappres / vappress)
    return RH

def dewpoint(temperature, relhum):
    """ Dew Point based on Magnus-Tetens formula
        Based on Alduchov, O. A., and R. E. Eskridge, 1996: Improved Magnus Form Approximation of Saturation Vapor Pressure. 
        J. Appl. Meteor. Climatol., 35, 601–609, 
        https://doi.org/10.1175/1520-0450(1996)035<0601:IMFAOS>2.0.CO;2.

    Args:
        temperature (float):    Air Temperature [°C]
        relhum (float) :        Relative air humidity [%]

    Returns:
        float: dew point temeprature [°C]
    """    
    dew = temperature - ((100 - relhum)/5.)
    return  dew

def SH(pressure, vappres):
    """Specific Humidity [%]

    Args:
        pressure (float):   Atmospheric Pressure [kPa]
        vappres (float):    Actual water vapour pressure [kPa]

    Returns:
       float : Specific Air Humidity [%]
    """    
    SH = (.622 * pressure / (pressure - vappres) * 100)
    return SH

def SWnet(SWIN, albedo):
    """ Shortwave Net Radiation [W/m2]

    Args:
        SWIN (float): 
        albedo (numpy.array): albedo [%]

    Returns:
        numpy.array: Shortwave Net Radiation [W/m2]
    """    
    swnet = (1 - albedo) * SWIN
    return swnet

def LWnet(lse, LWIN, B_constant, LST):
    """ Longwave Net Radiation [W/m2]

    Args:
        lse (numpy.array):      Land Surface Emissivity [dimensionless]
        LWIN (numpy.array):     Incoming Longwave Radiation [W/m2]
        B_constant (_float):    Stefan-Boltzmann constant. Deafults to 5.670374419e-08 [W m^-2 K^-4]
        LST (numpy.array):      Land Surface Temperature [K]

    Returns:
        numpy.array: Longwave Net Radiation [W/m2]
    """    
    lwnet = (lse * LWIN) - (lse * B_constant * (LST ** 4))
    return lwnet

def G(Rn, fvc):
    """ Soil/Ground Heat Flux [W/m2]
    Args:
        Rn (numpy.array):   Net Solar Radiation [W/m2]
        fvc (numpy.array):  Fraction of Vegetation Cover [dimensionless]

    Returns:
        numpy.array: Soil/Ground Heat Flux [W/m2]
    """    
    G0 = Rn * (0.05 + (1 - fvc) * (0.315 - 0.05))
    return G0

def airdenstiy(dry_air_press, gas_constant, t_wet_K, vappres):
    """ Density of Air [kg/m3]
        Based on https://designbuilder.co.uk/helpv3.4/Content/Calculation_of_Air_Density.htm
    Args:
        dry_air_press (float):  Pressure of dry air [kPa]
        gas_constant (float):   Molar Gas Constant. Defaults to 287.05 [J/kg-K] 
        t_wet_K (float):        (Wet Bulb) Air Temperature [K]
        vappres (float):        Actual water vapour pressure [kPa]

    Returns:
        float: Density of Air [kg/m3]
    """    
    airdens =  dry_air_press / (gas_constant * t_wet_K) + (vappres / (461.495  * t_wet_K)) 
    return airdens


def gamma(pressure, Cp, l_lambda, e):
    """ Psychometric constant [kPa/°C]

    Args:
        pressure (float):   Atmospheric Pressure [kPa]
        Cp (float):         Specific heat at constant pressure. Deafults to 0.001013 [MJ kg-1 °C-1]
        l_lambda (float):   Latent heat of vaporization. Defaults to 2.45 [MJ kg-1]
        e (float):          Ratio molecular weight of water vapour/dry air. Deafults to 0.622 [dimensionless]

    Returns:
        float: Psychometric constant [kPa/°C]
    """    
    psychro = (Cp  * pressure) / (e * l_lambda)
    return psychro

def d(cropheight):
    """ Zero Plane Displacement [m]

    Args:
        cropheight (numpy.array): Height of Crop [m]

    Returns:
        numpy.array: Zero Plane Displacement [m]
    """    
    d_hc = 0.667 * cropheight
    return d_hc

def u_friction(wind_sp, karmann, measur_height, d_hc, z0m):
    """ Friction Velocity [s/m]
        Based on https://inis.iaea.org/collection/NCLCollectionStore/_Public/37/118/37118528.pdf

    Args:
        wind_sp (float):        Wind Speed [m/s]
        karmann (float):        Von Karmann COnstant. Defaults to 0.41
        measur_height (float):  Height of measurement above the ground [m]
        d_hc (numpy.array):        Zero Plane Displacement [m]
        z0m (numpy.array):      Roughness length governing momentum transfer [m]

    Returns:
        numpy.array: Friction Velocity [s/m]
    """    
    u_ = (wind_sp * karmann) / np.log((measur_height-d_hc)/z0m) 
    return u_

def windsp_pbl(wind_sp, measur_height, z0m, z_pbl, d_hc):
    """ Wind speed at the planetary boundara layer [m/s]

    Args:
        wind_sp (float):            Wind Speed [m/s]
        measur_height (float):      Height of wind speed measurement above the ground [m]
        z0m (numpy.array):          Roughness length governing momentum transfer [m]
        z_pbl (float):              Height of Planetary Boundary Layer. Defaults to 1000 [m]
        d_hc (numpy.array):         Zero Plane Displacement [m]

    Returns:
        numpy.array: Wind speed at the planetary boundara layer [m/s]
    """    
    
    u_c = np.log((z_pbl - d_hc) / z0m) / np.log((measur_height - d_hc) / z0m)
    upbl = wind_sp * u_c
    u_pbl = np.where(upbl < 0, 0,upbl) 
    return u_pbl 

def PSIh_y(Y, psy_c, psy_d, psy_n): 
    """ Stability parameter for heat transfer [dimensionless] ????????????????????????????
    Based on Beljaars et Holstag (1991) and Brutsaert, 1999

    Args:
        Y (numpy.array): Conditional mask 
        psy_c (float): Coefficient. Defaults to 0.33
        psy_d (float): Coefficient. Defaults to 0.057
        psy_n (float): Coefficient. Defaults to 0.78

    Returns:
        numpy.array: Stability parameter for heat transfer [dimensionless]
    """        
    Y = abs(Y)
    PSIh_y = (1.0 - psy_d) / psy_n * np.log((psy_c + Y ** psy_n) / psy_c)
    return PSIh_y


def Cw(alfa, psi_C0, psi_C1, psi_C11, C21, psi_C22, z0m):
    """ Conditional masks [dimensionless] ????????????????????????????

    Args:
        alfa (float):           Coefficient. Defaults to 0.12
        psi_C0 (numpy.array):   Stability parameter for heat transfer for C0 mask.
        psi_C1 (numpy.array):   Stability parameter for heat transfer for C1 mask.
        psi_C11 (numpy.array):  Stability parameter for heat transfer for C11 mask.
        C21 (numpy.array):      Conditional mask C21
        psi_C22 (numpy.array):  Stability parameter for heat transfer for C22 mask.
        z0m (numpy.array):      Roughness length governing momentum transfer [m]

    Returns:
        numpy.array:  Conditional masks [dimensionless]
    """
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
    """_summary_ ?????????????????????????????

    Args:
        zd0 (_type_): _description_
        z0m (_type_): _description_

    Returns:
        _type_: _description_
    """    
    zdh = np.log(zd0 / (0.1 * z0m))

    return zdh

def PSI_m_stable(coeff_a, coeff_b, coeff_c, coeff_d, u_planet, MoninObuk_L):
    """_summary_                            ?????????????????????
 
    Args:
        coeff_a (_type_): _description_
        coeff_b (_type_): _description_
        coeff_c (_type_): _description_
        coeff_d (_type_): _description_
        u_planet (_type_): _description_
        MoninObuk_L (_type_): _description_

    Returns:
        _type_: _description_
    """    
    coeff_dzeta = u_planet / MoninObuk_L
    psi_m_stab = -(coeff_a * coeff_dzeta + coeff_b * (coeff_dzeta - coeff_c / coeff_d) \
                   * np.exp((-coeff_d) * coeff_dzeta) + coeff_b * coeff_c / coeff_d)
    return psi_m_stab

def PSI_h_stable(coeff_a, coeff_b, coeff_c, coeff_d, u_planet, MoninObuk_L):
    """_summary_                        ????????????????????????????????????

    Args:
        coeff_a (_type_): _description_
        coeff_b (_type_): _description_
        coeff_c (_type_): _description_
        coeff_d (_type_): _description_
        u_planet (_type_): _description_
        MoninObuk_L (_type_): _description_

    Returns:
        _type_: _description_
    """    
    coeff_dzeta = u_planet / MoninObuk_L   
    psi_h_stab = -((1 + 2 * coeff_a * coeff_dzeta / 3) ** 1.5 + coeff_b * (coeff_dzeta - coeff_c / coeff_d) \
                   * np.exp((-coeff_d) * coeff_dzeta)+ (coeff_b * coeff_c/coeff_d - 1))

    return psi_h_stab

def airResis(Z, zero_d, z0m, psi_m_stable, z_pbl, psi_h_stable, k, wind_sp):
    ra = (np.log((Z - zero_d) / z0m) - psi_m_stable) * (np.log((z_pbl - zero_d)/(0.1 * z0m)) - psi_h_stable) / (k ** 2 * wind_sp) # s/m
    return ra

def H(Rn, airdensity, rew, vappress, satvappress, slopevappress, gamma_cons):
    """ Sensible Heat Flux Aerodynamic Method [W/m2]

    Args:
        Rn (numpy.array):       Net Solar Radiation [W/m2]
        airdensity (float):     Density of Air [kg/m3]
        rew (_type_): _description_             ???????????????????????????
        vappress (float):       Water Vapour Pressure [kPa]
        satvappress (float):    Saturated Water Vapour Pressure [kPa]
        slopevappress (float):  Slope Vapour Pressure [kPa]
        gamma_cons (float):     Psychrometric Constant [kPa/°C]

    Returns:
        numpy.array: Sensible Heat Flux Aerodynamic Method [W/m2]
    """    
    H0 = (Rn - (airdensity / rew) * ((vappress - satvappress) / gamma_cons)) / (1.0 + slopevappress / gamma_cons)
    return H0

def LatentH(delta, Rn, G, airdensity, Cp, satvappress, water_vapour, airresis,  gamma_cons):
    """ Latent Heat Flux Aerodynamic Method [W/m2]

    Args:
        delta (numpy.array):    Slope of water vapour pressure gradient to temperature gradient [kPa/K]
        Rn (numpy.array):       Net Solar Radiation [W/m2]
        G (numpy.array):        Ground Heat Flux [W/ms]
        airdensity (float):     Density of Air [kg/m3]
        Cp (float):             Specific heat at constant pressure. Defaults to 0.001013 [J kg-1 °C-1]
        satvappress (float):    Saturated Water Vapour Pressure [kPa]
        water_vapour (float):   Water Vapour Pressure [kPa]
        airresis (numpy.array): Resistance of Air [s/m]
        gamma_cons (float):     Psychrometric Constant [kPa/°C]

    Returns:
        numpy.array: Latent Heat Flux Aerodynamic Method [W/m2]
    """    
    LHF = (delta * (Rn - G) + airdensity * Cp * (satvappress - water_vapour) / airresis) / (delta + gamma_cons)
    return LHF

def ET_actual(Rn, G, H, l_lambda):
    """ Actual EVapotranspiration [mm]

    Args:
        Rn (numpy.array):       Net Solar Radiation [W/m2]
        G (numpy.array):        Ground Heat Flux [W/ms]
        H (numpy.array):        Sensible Heat Flux [W/ms]
        l_lambda (float):       Latent heat of vaporization. Defaults to 2.45 [MJ kg-1]
    Returns:
        numpy.array: Actual Evapotranspiration [mm]
    """    
    ETA = (Rn - G - H) / l_lambda
    return ETA

def ET_daily(ET_act, l_lambda):
    """ Daily Evapotranspiration [mm/day]

    Args:
        ET_act (numpy.array): Actual Evapotranspiration [mm]
        l_lambda (float):     Latent heat of vaporization. Defaults to 2.45 [MJ kg-1]

    Returns:
        numpy.array: Daily Evapotranspiration [mm/day]
    """    
    ET_day = (ET_act / (l_lambda * 1000000)) * 0.3 * 86400 # daily ET
    return ET_day

def delta_ea_T(lst_K, temperature):
    """ Slope of water vapour pressure gradient to temperature gradient [kPa/K] based on Jackson etal. (1998).

    Args:
        lst_K (numpy.array): Land Surface Temperature [K]
        temperature (float): Air Temperature [°C]
    """    
    
    T = ((lst_K - 273.15) + temperature) / 2
    delta = (45.03 + 3.014 * T + 0.05345 * T ** 2 + 0.00224 * T ** 3) * 0.001
    return delta