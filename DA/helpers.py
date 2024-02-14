import pickle
import numpy as np
import pandas as pd
from datetime import timedelta

def save_dict_to_pickle(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)
    print(f"Dictionary saved to {filename}")

def load_dict_from_pickle(filename):
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    print(f"Dictionary loaded from {filename}")
    return loaded_dict

def haversine_distance(loc1, loc_array):
    """
    Calculate the Haversine distance between a point and an array of points on the Earth
    given their latitude and longitude in decimal degrees.

    Parameters:
    - loc1: Tuple containing the latitude and longitude of the first point (in decimal degrees).
    - loc_array: Array of tuples, each containing the latitude and longitude of a point (in decimal degrees).

    Returns:
    - Array of distances between loc1 and each point in loc_array (in kilometers).
    """
    if np.isnan(loc1[0]) and np.isnan(loc1[1]):
        distances = np.zeros(len(loc_array))
    else:
        # Radius of the Earth in kilometers
        R = 6371.0

        # Convert decimal degrees to radians
        lat1_rad, lon1_rad = np.radians(loc1)
        lat2_rad, lon2_rad = np.radians(np.array(loc_array).T)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distances = R * c
    return distances


def GC(r, c):
    '''
    Gaspari-Cohn localization function
    r: np.array containing the distance
    c: cutoff length
    '''
    abs_r = np.abs(r)
    if np.isnan(c):
        result = np.ones_like(abs_r, dtype=float)
    else:
        condition1 = (0 <= abs_r) & (abs_r <= c)
        condition2 = (c <= abs_r) & (abs_r <= 2 * c)

        result = np.zeros_like(abs_r, dtype=float)

        result[condition1] = -1/4 * (abs_r[condition1] / c) ** 5 + 1/2 * (abs_r[condition1] / c) ** 4 + 5/8 * (abs_r[condition1] / c) ** 3 - \
                            5/3 * (abs_r[condition1] / c) ** 2 + 1
        result[condition2] = 1/12 * (abs_r[condition2] / c) ** 5 - 1/2 * (abs_r[condition2] / c) ** 4 + 5/8 * (abs_r[condition2] / c) ** 3 + \
                            5/3 * (abs_r[condition2] / c) ** 2 - 5 * (abs_r[condition2] / c) + 4 - 2/3 * (c / abs_r[condition2])

    return result


def bin_dates_by_restart_dates(date_results,date_restarts_in,spinup=False,avoid_big_bins=False):
    '''
    Function that bins date_results (dates for which results are written) into bins defined by date_restarts
    Each bin end correspons to the next bin start (date_results_binned[0][-1] == date_results_binned[1][0])
    Warning: last bin can contain a different amount of result files, since the total time does not necessarily bin into n integer intervals
    '''
    date_restarts = date_restarts_in.copy()
    if avoid_big_bins:
        # set to true if wanting to avoid that extra dates are contained in the last bin
        # e.g. bin[0]: 2019-01-01 to 2020-01-01 (i.e. 1 year)
        #      bin[1]: 2020-01-01 to 2021-04-20 (i.e. >1 year)
        # will be:
        # e.g. bin[0]: 2019-01-01 to 2020-01-01 (i.e. 1 year)
        #      bin[1]: 2020-01-01 to 2021-01-01 (i.e. 1 year)
        #      bin[2]: 2021-01-01 to 2021-04-20 (i.e. <1 year)
        if date_results[-1] > date_restarts[-1]:
            date_restarts.append(date_results[-1])
        if date_results[0] != date_restarts[0]:
            date_restarts.insert(0,date_results[0])
            
    date_results_binned = [[]] #bin the simulation dates into the periods defined by date_restarts
    c = 0
    date_new_bin = date_restarts[c+1]
    for date_ in date_results:
        if date_ >= date_new_bin and date_ < date_restarts[-1] and date_ < date_results[-1]:
            date_results_binned[c].append(date_) #add the first date of the next bin to the date array to have 'overlapping' dates for the restart
            c+=1
            date_new_bin = date_restarts[c+1]
            date_results_binned.append([])
        date_results_binned[c].append(date_)    
   
    if spinup:
        assert date_end == date_results[-1], 'Choose an output frequency that fits n (integer) times inside of the restart frequency, e.g. 1 year sim with restart "AS" and output "MS". %s' % date_results
        assert len(date_results_binned) == 1
        date_results_binned = [date_results_binned[0]]*spinup

    return date_results_binned


def date_range_noleap(*args, **kwargs):
    '''
    Input parameters: see pd.date_range 
    Uses pd.date_range to generate a date range
    Checks if there are any leap days, and filters them out
    '''
    date_range = pd.date_range(*args, **kwargs)

    dt = date_range[1] - date_range[0]
    
    if dt > timedelta(days=1):
        if 'freq' in kwargs and 'd' in kwargs['freq']: #frequency is based on the amount of days
            # first generate daily data, then simply take every nth element
            kwargs['freq'] = '1d'
            date_range = pd.date_range(*args, **kwargs)

            # Filter out leap days (February 29)
            filtered_date_range_d = date_range[~((date_range.month == 2) & (date_range.day == 29))]

            filtered_date_range = filtered_date_range_d[::dt.days]
        else: #frequency is likely based on months/years, keep it as is
            filtered_date_range = pd.date_range(*args, **kwargs)
            
    elif dt <= timedelta(days=1):
        assert ((timedelta(days=1).total_seconds() / dt.total_seconds()) % 1) == 0, 'frequency should fit an integer amount of times in one day'
        # simply remove leap day entries
        filtered_date_range = date_range[~((date_range.month == 2) & (date_range.day == 29))]
    
    return filtered_date_range

###############################################################################################
### Below some ParFlow helper functions, imported from SLOTH https://github.com/HPSCTerrSys/SLOTH/
###############################################################################################

def readSa(file):
    """
    Reads data from a file in ASCI format and returns a NumPy array.

    Parameters
    ----------
    file : str
        The file path to read the data from.

    Returns
    -------
    numpy.ndarray
        A NumPy array containing the data read from the file.

    Example
    -------
    >>> data = readSa('data.txt')
    >>> print(data)
    [[1.2 3.4]
     [5.6 7.8]]

    """
    with open(file, 'r') as f:
        header = f.readline()
        nx, ny, nz = (int(item) for item in header.split(' '))

        data = np.genfromtxt(f, dtype=float)
        data = data.reshape((nz, ny, nx))

        return data
    
def writeSa(file, data):
    """
    Writes data to a file in ASCI format.

    Parameters
    ----------
    file : str
        The file path to write the data to.
    data : numpy.ndarray
        The NumPy array containing the data to be written.

    Returns
    -------
    None

    Example
    -------
    >>> data = np.array([[1.2, 3.4], [5.6, 7.8]])
    >>> writeSa('output.txt', data)

    """
    nz, ny, nx = data.shape
    with open(file, 'w') as f:
        f.write(f'{nx} {ny} {nz}\n')
        # Below should be more easy with a flatten array...
        # But how to flatt? C or F order? 
        # If knowen and tested change below.
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f'{data[k,j,i]}\n')
                    