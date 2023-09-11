from osgeo import gdal
from affine import Affine
import rasterio as rio
import numpy as np
import os, sys
from metpy.units import units
from metpy.units import units

from metpy.calc import specific_humidity_from_mixing_ratio, saturation_mixing_ratio

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

def saturation_vapor_pressure(sat_pressure_0c, temperature):
    satVapPress = sat_pressure_0c * np.exp(17.67 * (temperature - units.Quantity(273.15, 'kelvin')) / (temperature - units.Quantity(29.65, 'kelvin')))
    
    return satVapPress

def psychrometric_vapor_pressure_wet(satvappres, my_press_ref, tair_dry4, my_t_ref_unit, psychrometer_coefficient = None):
    if psychrometer_coefficient is None:
        psychrometer_coefficient = units.Quantity(6.21e-4, '1/K')
        return satvappres - psychrometer_coefficient * my_press_ref * (tair_dry4 - my_t_ref_unit).to('kelvin')
    
def dewpoint(my_ea_ref, sat_pressure_0c):
    val = np.log(my_ea_ref / sat_pressure_0c)
    return  (units.Quantity(0., 'degC') + units.Quantity(243.5, 'delta_degC') * val / (17.67 - val))


def specific_humidity_from_dewpoint(pressure, dewpoint):
    mixing_ratio = saturation_mixing_ratio(pressure, dewpoint)
    spedicif_hum_dew = specific_humidity_from_mixing_ratio(mixing_ratio)
    return spedicif_hum_dew
