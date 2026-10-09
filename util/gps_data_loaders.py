import datetime
import os
from typing import Union

import astropy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import make_splrep

from util.datasets import get_time_length
import datetime as dt


def get_gps_data_loader(dataset_name):
    if 'campi_flegrei' in dataset_name:
        return CampiFlegreiGPSData()
    elif 'laguna_del_male' in dataset_name or 'laguna_del_maule' in dataset_name:
        return LagunaDelMaleGPSData()
    else:
        raise ValueError('Dataset name {} not recognized'.format(dataset_name))


def coord_to_pixel(clon, clat, plon, plat, XCoord_geo, YCoord_geo):
    XCoord_pix = int((XCoord_geo - clon) // plon)
    YCoord_pix = int((YCoord_geo - clat) // plat)

    # XCoord_pix2geo = clon + ((XCoord_pix+1) * plon)
    # YCoord_pix2geo = clat + ((YCoord_pix+1) * plat)

    return XCoord_pix, YCoord_pix


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GPSDataLoader(metaclass=Singleton):
    def __init__(self, path: Union[str, os.PathLike], plot_data: bool = False):
        pass

    def stations(self) -> list[str]:
        raise NotImplementedError

    def station_coords(self, station: str) -> np.ndarray:
        raise NotImplementedError

    # Processed values, in mm
    def value(self, station: str, date: Union[str, datetime.date]) -> Union[np.float64, None]:
        if type(date) is datetime.date:
            date = date.strftime('%Y%m%d')
        return self._value(station, date)

    def _value(self, station: str, date: str) -> Union[np.float64, None]:
        raise NotImplementedError

    def value_original(self, station: str, date: Union[str, datetime.date]) -> Union[np.float64, None]:
        if type(date) is datetime.date:
            date = date.strftime('%Y%m%d')
        return self._value_original(station, date)

    def _value_original(self, station: str, date: str) -> Union[np.float64, None]:
        raise NotImplementedError


class CampiFlegreiGPSData(GPSDataLoader):
    def __init__(self, path: Union[str, os.PathLike] = 'datasets/campi_flegrei/gps_data', plot_data: bool = False):
        super().__init__(path, plot_data)
        self.path = path

        self.gps_data = {
            'ACAE':	{
                'location': np.array([40.820, 14.141]),
                'coords': np.array([257, 252]),
                'data': {},
                'data_orig': {},
            },
            'ARFE':	{
                'location': np.array([40.832, 14.099]),
                'coords': np.array([244, 209]),
                'data': {},
                'data_orig': {},
            },
            'ASTR':	{
                'location': np.array([40.840, 14.159]),
                'coords': np.array([236, 270]),
                'data': {},
                'data_orig': {},
            },
            'BAGN':	{
                'location': np.array([40.804, 14.174]),
                'coords': np.array([273, 285]),
                'data': {},
                'data_orig': {},
            },
            'BAIA':	{
                'location': np.array([40.809, 14.080]),
                'coords': np.array([268, 190]),
                'data': {},
                'data_orig': {},
            },


            # 'CMIS':	{
            #     'location': np.array([40.778, 14.089]),
            #     'coords': np.array([299, 199]),
            #     'data': {},
            #     'data_orig': {},
            # },


            'CUMA':	{
                'location': np.array([40.848, 14.058]),
                'coords': np.array([228, 168]),
                'data': {},
                'data_orig': {},
            },
            'FRUL':	{
                'location': np.array([40.877, 14.225]),
                'coords': np.array([199, 336]),
                'data': {},
                'data_orig': {},
            },
            'IPPO':	{
                'location': np.array([40.837, 14.167]),
                'coords': np.array([239, 278]),
                'data': {},
                'data_orig': {},
            },
            'ISMO':	{
                'location': np.array([40.826, 14.191]),
                'coords': np.array([251, 302]),
                'data': {},
                'data_orig': {},
            },
            'LICO':	{
                'location': np.array([40.876, 14.049]),
                'coords': np.array([200, 159]),
                'data': {},
                'data_orig': {},
            },
            'MAFE':	{
                'location': np.array([40.847, 14.258]),
                'coords': np.array([229, 369]),
                'data': {},
                'data_orig': {},
            },
            'MORU':	{
                'location': np.array([40.859, 14.082]),
                'coords': np.array([217, 192]),
                'data': {},
                'data_orig': {},
            },


            # 'NAMM':	{
            #     'location': np.array([40.836, 14.254]),
            #     'coords': np.array([240, 365]),
            #     'data': {},
            #     'data_orig': {},
            # },


            'NISI':	{
                'location': np.array([40.797, 14.163]),
                'coords': np.array([280, 274]),
                'data': {},
                'data_orig': {},
            },
            'PIS1':	{
                'location': np.array([40.830, 14.145]),
                'coords': np.array([246, 256]),
                'data': {},
                'data_orig': {},
            },
            'QUAR':	{
                'location': np.array([40.877, 14.143]),
                'coords': np.array([199, 254]),
                'data': {},
                'data_orig': {},
            },
            'RITE':	{
                'location': np.array([40.823, 14.126]),
                'coords': np.array([254, 236]),
                'data': {},
                'data_orig': {},
            },
            'SOLO':	{
                'location': np.array([40.830, 14.135]),
                'coords': np.array([246, 245]),
                'data': {},
                'data_orig': {},
            },
            'STRZ':	{
                'location': np.array([40.835, 14.111]),
                'coords': np.array([241, 221]),
                'data': {},
                'data_orig': {},
            },
            'VICA':	{
                'location': np.array([40.855, 14.120]),
                'coords': np.array([221, 230]),
                'data': {},
                'data_orig': {},
            },
        }

        files = os.listdir(self.path)

        for file in files:
            if '_los.csv' not in file:
                continue

            station_name = file.split('_')[0]
            if station_name not in self.gps_data:
                continue

            gps_file = pd.read_csv(os.path.join(self.path, file))

            gps_file_date = np.array(gps_file['date'])
            gps_file_deform = np.array(gps_file['deform'] * 1000)  # From m to mm

            def _from_mjd_to_string(d):
                d = astropy.time.Time(d, format='mjd')
                d = str(d.iso).split(' ')[0]
                d = d.translate({ord(c): None for c in '-'})
                return d

            data_orig = {}

            for i in range(len(gps_file_date)):
                date = gps_file_date[i]
                val = gps_file_deform[i]

                data_orig[date] = val

            gps_file_deform = gps_file_deform / 10  # Transforming to cm for the spline calculations

            xs = np.copy(gps_file_date)
            ys = np.copy(gps_file_deform)

            xs_int = np.array([get_time_length(f'{xs[0]}_{date}') for date in xs])

            ys2 = make_splrep(xs_int, ys, s=len(xs_int) // 10)(xs_int)
            ys_mask = np.abs(ys - ys2) < 5

            x_new = np.arange(xs_int[0], xs_int[-1])
            y_new = make_splrep(xs_int[ys_mask], ys[ys_mask], s=len(xs_int[ys_mask]) // 9)(x_new)

            # We transform back to mm
            ys = ys * 10
            y_new = y_new * 10

            if plot_data:
                plt.scatter(xs_int, ys, alpha=0.2, label=file.split('_')[1])
                plt.plot(x_new, y_new, c='red')
                plt.legend()
                plt.show()

            date0 = dt.datetime.strptime(str(xs[0]), '%Y%m%d').date()
            data = {}
            for i in range(len(x_new)):
                date = date0 + dt.timedelta(int(x_new[i]))
                date = str(date)
                date = date.translate({ord(c): None for c in '-'})
                val = y_new[i]

                data[date] = val

            self.gps_data[station_name]['data'] = data
            self.gps_data[station_name]['data_orig'] = data_orig

    def stations(self) -> list[str]:
        keys = self.gps_data.keys()
        keys = [str(x) for x in keys]
        return keys

    def station_coords(self, station: str) -> np.ndarray:
        coords = self.gps_data[station]['coords']
        c = np.array([coords[0], coords[1]])
        return c

    def _value(self, station: str, date: Union[str, datetime.date]) -> Union[np.float64, None]:
        if date not in self.gps_data[station]['data']:
            return None
        return self.gps_data[station]['data'][date]

    def _value_original(self, station: str, date: Union[str, datetime.date]) -> Union[np.float64, None]:
        if date not in self.gps_data[station]['data_orig']:
            return None
        return self.gps_data[station]['data_orig'][date]


class LagunaDelMaleGPSData(GPSDataLoader):
    def __init__(self, path: Union[str, os.PathLike] = 'datasets/laguna_del_male/GPS_Data_csv', plot_data: bool = False):
        super().__init__(path, plot_data)
        self.path = path

        self.gps_data = {
            'MAU2': {
                'location': np.array([-36.0679, -70.5336]),
                'coords_off_centered': np.array([567, 466]),
                'coords': np.array([259, 208]),
                'data': {},
                'data_orig': {},
            },
            'COLO': {
                'location': np.array([-36.1349, -70.4702]),
                'coords_off_centered': np.array([634, 529]),
                'coords': np.array([326, 271]),
                'data': {},
                'data_orig': {},
            },
            'LDMP': {
                'location': np.array([-36.0133, -70.5604]),
                'coords_off_centered': np.array([513, 439]),
                'coords': np.array([205, 181]),
                'data': {},
                'data_orig': {},
            },
            'NIEB': {
                'location': np.array([-36.1014, -70.5734]),
                'coords_off_centered': np.array([601, 426]),
                'coords': np.array([293, 168]),
                'data': {},
                'data_orig': {},
            },
            'PUEL': {
                'location': np.array([-36.0600, -70.4554]),
                'coords_off_centered': np.array([559, 544]),
                'coords': np.array([251, 286]),
                'data': {},
                'data_orig': {},
            },
        }

        for file in os.listdir(self.path):
            if '_los.csv' not in file:  # export_{STA}_los.csv; export_{STA}.csv holds the ENU series
                continue
            gps_file = pd.read_csv(os.path.join(self.path, file))

            gps_file_date = np.array(gps_file['fecha'])
            gps_file_deform = np.array(gps_file['0']) * 10 # From cm to mm

            gps = [gps_file_date, gps_file_deform]
            data_orig = {}

            for i in range(len(gps[0])):
                date = gps[0][i]
                date = date.translate({ord(c): None for c in '-'})
                val = gps[1][i]

                data_orig[date] = val

            gps_file_deform = gps_file_deform / 10 # We keep it to cm here for the spline calculations

            xs = np.copy(gps_file_date)
            xs = [date.translate({ord(c): None for c in '-'}) for date in xs]
            ys = np.copy(gps_file_deform)

            xs_int = np.array([get_time_length(f'{xs[0]}_{date}') for date in xs])

            ys2 = make_splrep(xs_int, ys, s=len(xs_int) // 3)(xs_int)
            ys_mask = np.abs(ys - ys2) < 1.5

            x_new = np.arange(xs_int[0], xs_int[-1])
            y_new = make_splrep(xs_int[ys_mask], ys[ys_mask], s=len(xs_int[ys_mask]) // 2.5)(x_new)

            # We transform back to mm
            ys = ys * 10
            y_new = y_new * 10

            if plot_data:
                plt.scatter(xs_int, ys, alpha=0.2, label=file.split('_')[1])
                plt.plot(x_new, y_new, c='red')
                plt.legend()
                plt.show()

            date0 = dt.datetime.strptime(str(xs[0]), '%Y%m%d').date()
            data = {}
            for i in range(len(x_new)):
                date = date0 + dt.timedelta(int(x_new[i]))
                date = str(date)
                date = date.translate({ord(c): None for c in '-'})
                val = y_new[i]

                data[date] = val

            for key in self.gps_data.keys():
                if key in file:
                    self.gps_data[key]['data'] = data
                    self.gps_data[key]['data_orig'] = data_orig
                    break

    def stations(self) -> list[str]:
        keys = self.gps_data.keys()
        keys = [str(x) for x in keys]
        return keys

    def station_coords(self, station: str) -> np.ndarray:
        coords = self.gps_data[station]['coords']
        c = np.array([coords[0], coords[1]])
        return c

    def station_coords_off_centered(self, station: str) -> np.ndarray:
        coords = self.gps_data[station]['coords_off_centered']
        c = np.array([coords[0], coords[1]])
        return c

    def _value(self, station: str, date: Union[str, datetime.date]) -> Union[np.float64, None]:
        if date not in self.gps_data[station]['data']:
            return None
        return self.gps_data[station]['data'][date]

    def _value_original(self, station: str, date: Union[str, datetime.date]) -> Union[np.float64, None]:
        if date not in self.gps_data[station]['data_orig']:
            return None
        return self.gps_data[station]['data_orig'][date]

# spline_smooth parameters (s divisor, outlier threshold in cm, second s divisor) used by the existing loaders
CF_SMOOTHING = (10, 5, 9)
LDM_SMOOTHING = (3, 1.5, 2.5)


def spline_smooth(dates, values_mm, s_div=10, outlier_cm=5., s_div_outliers=9):
    """
    The existing loaders' smoothing, with its parameters exposed: a first spline (s = n // s_div) flags outliers
    further than outlier_cm from it, a second spline through the rest (s = n // s_div_outliers) is sampled daily from
    the first date up to, not including, the last one. Returns {YYYYMMDD: mm}.
    """
    ys = np.asarray(values_mm) / 10  # spline parameters are tuned in cm
    xs_int = np.array([get_time_length(f'{dates[0]}_{date}') for date in dates])

    ys2 = make_splrep(xs_int, ys, s=len(xs_int) // s_div)(xs_int)
    ys_mask = np.abs(ys - ys2) < outlier_cm

    x_new = np.arange(xs_int[0], xs_int[-1])
    y_new = make_splrep(xs_int[ys_mask], ys[ys_mask], s=len(xs_int[ys_mask]) // s_div_outliers)(x_new)
    y_new = y_new * 10

    date0 = dt.datetime.strptime(str(dates[0]), '%Y%m%d').date()
    return {(date0 + dt.timedelta(int(x))).strftime('%Y%m%d'): val for x, val in zip(x_new, y_new)}
