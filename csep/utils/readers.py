import datetime
import math
import re
import warnings
import enum
import csv
from itertools import zip_longest
import os

import numpy

from csep.utils.time_utils import strptime_to_utc_datetime, strptime_to_utc_epoch
from csep.core.exceptions import CSEPIOException

def read_ndk(filename):
    """
    Reads an NDK file to a :class:`~csep.core.Catalogs.ZMAPCatalog` object. This code was modified from the obspy implementation
    to work with CSEP Catalog objects. If a more verbose representation of the catalog is required, including picks and
    other various data, please use the obspy catalog.

    Args:
        filename: file or file-like object
    """
    # this function first parses the data into a human readable dict with appropriate values and then finally returns a
    # CSEP catalog object.

    out = []

    if not hasattr(filename, "read"):
        # Check if it exists, otherwise assume its a string.
        try:
            with open(filename, "rt") as fh:
                data = fh.read()
        except Exception:
            try:
                data = filename.decode()
            except Exception:
                data = str(filename)
            data = data.strip()
    else:
        data = filename.read()
        if hasattr(data, "decode"):
            data = data.decode()

    # Create iterator that yields lines.
    def lines_iter():
        prev_line = -1
        while True:
            next_line = data.find("\n", prev_line + 1)
            if next_line < 0:
                break
            yield data[prev_line + 1: next_line]
            prev_line = next_line
        if len(data) > prev_line + 1:
            yield data[prev_line + 1:]

    # Loop over 5 lines at once.
    for _i, lines in enumerate(zip_longest(*[lines_iter()] * 5)):
        if None in lines:
            msg = "Skipped last %i lines. Not a multiple of 5 lines." % (
                lines.count(None))
            warnings.warn(msg, RuntimeWarning)
            continue

        # Parse the lines to a human readable dictionary.
        try:
            record = _read_lines(*lines)
        # need to handle the exception here.
        except (ValueError, IOError):
            # exc = traceback.format_exc()
            msg = (
                "Could not parse event %i (faulty file?). Will be "
                "skipped." % (_i + 1))
            warnings.warn(msg, RuntimeWarning)
            continue

        # Assemble the time for the reference origin.
        try:
            date_time_dict = _parse_datetime_to_zmap(record["date"], record["time"])
        except ValueError:
            msg = ("Invalid time in event %i. '%s' and '%s' cannot be "
                   "assembled to a valid time. Event will be skipped.") % \
                  (_i + 1, record["date"], record["time"])
            warnings.warn(msg, RuntimeWarning)
            continue

        # we are stripping off a significant amount of information from the gCMT catalog
        # if more information is required please use the obspy implementation
        out_tup = (record['hypo_lng'],
                   record['hypo_lat'],
                   date_time_dict['year'],
                   date_time_dict['month'],
                   date_time_dict['day'],
                   record["Mw"],
                   record["hypo_depth_in_km"],
                   date_time_dict['hour'],
                   date_time_dict['minute'],
                   date_time_dict['second'])
        out.append(out_tup)
    return out

def _read_lines(line1, line2, line3, line4, line5):
    # First line: Hypocenter line
    # [1-4]   Hypocenter reference catalog (e.g., PDE for USGS location,
    #         ISC for #ISC catalog, SWE for surface-wave location,
    #         [Ekstrom, BSSA, 2006])
    # [6-15]  Date of reference event
    # [17-26] Time of reference event
    # [28-33] Latitude
    # [35-41] Longitude
    # [43-47] Depth
    # [49-55] Reported magnitudes, usually mb and MS
    # [57-80] Geographical location (24 characters)
    rec = {}
    rec["hypocenter_reference_catalog"] = line1[:4].strip()
    rec["date"] = line1[5:15].strip()
    rec["time"] = line1[16:26]
    rec["hypo_lat"] = float(line1[27:33])
    rec["hypo_lng"] = float(line1[34:41])
    rec["hypo_depth_in_km"] = float(line1[42:47])
    rec["mb"], rec["MS"] = map(float, line1[48:55].split())
    rec["location"] = line1[56:80].strip()

    # Second line: CMT info (1)
    # [1-16]  CMT event name. This string is a unique CMT-event identifier.
    #         Older events have 8-character names, current ones have
    #         14-character names.  See note (1) below for the naming
    #         conventions used.
    # [18-61] Data used in the CMT inversion. Three data types may be used:
    #         Long-period body waves (B), Intermediate-period surface waves
    #         (S), and long-period mantle waves (M). For each data type,
    #         three values are given: the number of stations used, the
    #         number  of components used, and the shortest period used.
    # [63-68] Type of source inverted for:
    #         "CMT: 0" - general moment tensor;
    #         "CMT: 1" - moment tensor with constraint of zero trace
    #             (standard);
    #         "CMT: 2" - double-couple source.
    # [70-80] Type and duration of moment-rate function assumed in the
    #         inversion.  "TRIHD" indicates a triangular moment-rate
    #         function, "BOXHD" indicates a boxcar moment-rate function.
    #         The value given is half the duration of the moment-rate
    #         function. This value is assumed in the inversion, following a
    #         standard scaling relationship (see note (2) below), and is
    #         not derived from the analysis.
    rec["cmt_event_name"] = line2[:16].strip()

    data_used = line2[17:61].strip()
    # Use regex to get the data used in case the data types are in a
    # different order.
    data_used = re.findall(r"[A-Z]:\s*\d+\s+\d+\s+\d+", data_used)
    rec["data_used"] = []
    for data in data_used:
        data_type, count = data.split(":")
        if data_type == "B":
            data_type = "body waves"
        elif data_type == "S":
            data_type = "surface waves"
        elif data_type == "M":
            data_type = "mantle waves"
        else:
            msg = "Unknown data type '%s'." % data_type
            raise ValueError(msg)

        sta, comp, period = count.strip().split()

        rec["data_used"].append({
            "wave_type": data_type,
            "station_count": int(sta),
            "component_count": int(comp),
            "shortest_period": float(period)
        })

    source_type = line2[62:68].strip().upper().replace(" ", "")
    if source_type == "CMT:0":
        rec["source_type"] = "general"
    elif source_type == "CMT:1":
        rec["source_type"] = "zero trace"
    elif source_type == "CMT:2":
        rec["source_type"] = "double couple"
    else:
        msg = "Unknown source type."
        raise ValueError(msg)

    mr_type, mr_duration = [i.strip() for i in line2[69:].split(":")]
    mr_type = mr_type.strip().upper()
    if mr_type == "TRIHD":
        rec["moment_rate_type"] = "triangle"
    elif mr_type == "BOXHD":
        rec["moment_rate_type"] = "box car"
    else:
        msg = "Moment rate function '%s' unknown." % mr_type
        raise ValueError(msg)

    # Specified as half the duration in the file.
    rec["moment_rate_duration"] = float(mr_duration) * 2.0

    # Third line: CMT info (2)
    # [1-58]  Centroid parameters determined in the inversion. Centroid
    #         time, given with respect to the reference time, centroid
    #         latitude, centroid longitude, and centroid depth. The value
    #         of each variable is followed by its estimated standard error.
    #         See note (3) below for cases in which the hypocentral
    #         coordinates are held fixed.
    # [60-63] Type of depth. "FREE" indicates that the depth was a result
    #         of the inversion; "FIX " that the depth was fixed and not
    #         inverted for; "BDY " that the depth was fixed based on
    #         modeling of broad-band P waveforms.
    # [65-80] Timestamp. This 16-character string identifies the type of
    #         analysis that led to the given CMT results and, for recent
    #         events, the date and time of the analysis. This is useful to
    #         distinguish Quick CMTs ("Q-"), calculated within hours of an
    #         event, from Standard CMTs ("S-"), which are calculated later.
    if line3[0:9] != "CENTROID:":
        raise IOError("parse error: file should have CENTROID ")
    numbers = [line3[10:18], line3[18:22], line3[22:29], line3[29:34],
               line3[34:42], line3[42:47], line3[47:53], line3[53:58]]
    rec["centroid_time"], rec["centroid_time_error"], \
        rec["centroid_latitude"], rec["centroid_latitude_error"], \
        rec["centroid_longitude"], rec["centroid_longitude_error"], \
        rec["centroid_depth_in_km"], rec["centroid_depth_in_km_error"] = \
        map(float, numbers)
    type_of_depth = line3[59:63].strip().upper()

    if type_of_depth == "FREE":
        rec["type_of_centroid_depth"] = "from moment tensor inversion"
    elif type_of_depth == "FIX":
        rec["type_of_centroid_depth"] = "from location"
    elif type_of_depth == "BDY":
        rec["type_of_centroid_depth"] = "from modeling of broad-band P " \
                                        "waveforms"
    else:
        msg = "Unknown type of depth '%s'." % type_of_depth
        raise ValueError(msg)

    timestamp = line3[64:].strip().upper()
    rec["cmt_timestamp"] = timestamp
    if timestamp.startswith("Q-"):
        rec["cmt_type"] = "quick"
    elif timestamp.startswith("S-"):
        rec["cmt_type"] = "standard"
    # This is invalid but occurs a lot so we include it here.
    elif timestamp.startswith("O-"):
        rec["cmt_type"] = "unknown"
    else:
        msg = "Invalid CMT timestamp '%s' for event %s." % (
            timestamp, rec["cmt_event_name"])
        raise ValueError(msg)

    # Fourth line: CMT info (3)
    # [1-2]   The exponent for all following moment values. For example, if
    #         the exponent is given as 24, the moment values that follow,
    #         expressed in dyne-cm, should be multiplied by 10**24.
    # [3-80]  The six moment-tensor elements: Mrr, Mtt, Mpp, Mrt, Mrp, Mtp,
    #         where r is up, t is south, and p is east. See Aki and
    #         Richards for conversions to other coordinate systems. The
    #         value of each moment-tensor element is followed by its
    #         estimated standard error. See note (4) below for cases in
    #         which some elements are constrained in the inversion.
    # Exponent converts to dyne*cm. To convert to N*m it has to be decreased
    # seven orders of magnitude.
    exponent = int(line4[:2]) - 7
    # Directly set the exponent instead of calculating it to enhance
    # precision.
    rec["m_rr"], rec["m_rr_error"], rec["m_tt"], rec["m_tt_error"], \
        rec["m_pp"], rec["m_pp_error"], rec["m_rt"], rec["m_rt_error"], \
        rec["m_rp"], rec["m_rp_error"], rec["m_tp"], rec["m_tp_error"] = \
        map(lambda x: float("%sE%i" % (x, exponent)), line4[2:].split())

    # Fifth line: CMT info (4)
    # [1-3]   Version code. This three-character string is used to track
    #         the version of the program that generates the "ndk" file.
    # [4-48]  Moment tensor expressed in its principal-axis system:
    #         eigenvalue, plunge, and azimuth of the three eigenvectors.
    #         The eigenvalue should be multiplied by 10**(exponent) as
    #         given on line four.
    # [50-56] Scalar moment, to be multiplied by 10**(exponent) as given on
    #         line four.
    # [58-80] Strike, dip, and rake for first nodal plane of the
    #         best-double-couple mechanism, repeated for the second nodal
    #         plane.  The angles are defined as in Aki and Richards. The
    #         format for this string should not be considered fixed.
    rec["version_code"] = line5[:3].strip()
    rec["scalar_moment"] = float(line5[49:56]) * (10 ** exponent)
    # Calculate the moment magnitude.
    rec["Mw"] = 2.0 / 3.0 * (math.log10(rec["scalar_moment"]) - 9.1)

    principal_axis = line5[3:48].split()
    rec["principal_axis"] = []
    for axis in zip(*[iter(principal_axis)] * 3):
        rec["principal_axis"].append({
            # Again set the exponent directly to avoid even more rounding
            # errors.
            "length": "%sE%i" % (axis[0], exponent),
            "plunge": float(axis[1]),
            "azimuth": float(axis[2])
        })

    nodal_planes = map(float, line5[57:].strip().split())
    rec["nodal_plane_1"] = {
        "strike": next(nodal_planes),
        "dip": next(nodal_planes),
        "rake": next(nodal_planes)
    }
    rec["nodal_plane_2"] = {
        "strike": next(nodal_planes),
        "dip": next(nodal_planes),
        "rake": next(nodal_planes)
    }

    return rec

def _parse_datetime_to_zmap(date, time):
    """ Helping function to return datetime in zmap format.

    Args:
        date: string record from .ndk file
        time: string record from .ndk file

    Returns:
        out: dictionary following
            out_dict = {'year': year, 'month': month, 'day': day',
                        'hour': hour, 'minute': minute, 'second': second}
    """

    add_minute = False
    if ":60.0" in time:
        time = time.replace(":60.0", ":0.0")
        add_minute = True
    try:
        dt = strptime_to_utc_datetime(date + " " + time, format="%Y/%m/%d %H:%M:%S.%f")
    except (TypeError, ValueError):
        msg = ("Could not parse date/time string '%s' and '%s' to a valid "
               "time" % (date, time))
        raise RuntimeError(msg)

    if add_minute:
        dt += datetime.timedelta(minutes=1)

    out = {}
    out['year'] = dt.year
    out['month'] = dt.month
    out['day'] = dt.day
    out['hour'] = dt.hour
    out['minute'] = dt.minute
    out['second'] = dt.second
    return out

def read_zmap_ascii(fname, delimiter=None):
    """
    Reads csep1 ascii format into numpy structured array. this can be passed into a catalog object constructor. Using

    $ catalog = csep.core.catalogs.CSEPCatalog(catalog=read_zmap_ascii(fname), **kwargs)

    Many of the catalogs from the CSEP1 testing center were empty indicating that no observed earthquakes were available
    during the time period of the catalog. In the case of an empty catalog, this function will return an empty numpy array. The
    catalog object should still be created, but it will contain zero events. Therefore it can still be used for evaluations
    and plotting as normal.

    The CSEP Format has the following dtype:

    dtype = numpy.dtype([('longitude', numpy.float32),
                        ('latitude', numpy.float32),
                        ('year', numpy.int32),
                        ('month', numpy.int32),
                        ('day', numpy.int32),
                        ('magnitude', numpy.float32),
                        ('depth', numpy.float32),
                        ('hour', numpy.int32),
                        ('minute', numpy.int32),
                        ('second', numpy.int32)])

    Args:
        fname: absolute path to csep1 catalog file

    Returns:
        list: list of tuples representing above type, empty if no events were found
    """

    class ColumnIndex(enum.Enum):
        Longitude = 0
        Latitude = 1
        DecimalYear = 2
        Month = 3
        Day = 4
        Magnitude = 5
        Depth = 6
        Hour = 7
        Minute = 8
        Second = 9

        # Error columns
        HorizontalError = 10
        DepthError = 11
        MagnitudeError = 12

        NetworkName = 13
        NumColumns = 14

    # short-circuit for empty file
    if os.stat(fname).st_size == 0:
        return []

    # arrange file into list of tuples
    out = []
    csep1_zmap = numpy.loadtxt(fname, delimiter=delimiter)
    for line in csep1_zmap:
        event_tuple = (line[ColumnIndex.Longitude],
                       line[ColumnIndex.Latitude],
                       line[ColumnIndex.DecimalYear],
                       line[ColumnIndex.Month],
                       line[ColumnIndex.Day],
                       line[ColumnIndex.Magnitude],
                       line[ColumnIndex.Depth],
                       line[ColumnIndex.Hour],
                       line[ColumnIndex.Minute],
                       line[ColumnIndex.Second])
        out.append(event_tuple)
    return out

def read_csep_ascii(fname, return_catalog_id=False):
    """ Reads single catalog in CSEP ascii format.

    Args:
        fname (str): filename of catalog
        return_catalog_id (bool): return the catalog id

    Returns:
        list of tuples containing event information or (eventlist, catalog_id)
    """

    def is_header_line(line):
        if line[0] == 'lon':
            return True
        else:
            return False

    def parse_datetime(dt_string):
        try:
            origin_time = strptime_to_utc_epoch(dt_string, format='%Y-%m-%dT%H:%M:%S.%f')
            return origin_time
        except:
            pass
        try:
            origin_time = strptime_to_utc_epoch(dt_string, format='%Y-%m-%dT%H:%M:%S')
            return origin_time
        except:
            pass
        raise CSEPIOException("Supported time-string formats are '%Y-%m-%dT%H:%M:%S.%f' and '%Y-%m-%dT%H:%M:%S'")

    with open(fname, 'r', newline='') as input_file:
        catalog_reader = csv.reader(input_file, delimiter=',')
        # csv treats everything as a string convert to correct types
        is_first_event = True
        events = []
        for line in catalog_reader:
            # skip header line on first read if included in file
            if is_first_event and is_header_line(line):
                continue
            # convert to correct types
            lon = float(line[0])
            lat = float(line[1])
            magnitude = float(line[2])
            # maybe fractional seconds are not included
            origin_time = parse_datetime(line[3])
            depth = float(line[4])
            catalog_id = line[5]
            event_id = line[6]
            events.append((lon, lat, magnitude, origin_time, depth, event_id))

        if not return_catalog_id:
            return events
        else:
            return events, catalog_id



def read_ingv_rcmt_csv(fname):
    """
    Reader for the INGV (Istituto Nazionale di Geofisica e Vulcanologia - Italy)  European-
    Mediterranean regional Centroid Moment Tensor Catalog.
    It reads a catalog in .csv format, directly downloaded from http://rcmt2.bo.ingv.it/ using the Catalog Search (Beta
    version).

    The ZMAP Format has the following dtype:

    dtype = numpy.dtype([('longitude', numpy.float32),
                        ('latitude', numpy.float32),
                        ('year', numpy.int32),
                        ('month', numpy.int32),
                        ('day', numpy.int32),
                        ('magnitude', numpy.float32),
                        ('depth', numpy.float32),
                        ('hour', numpy.int32),
                        ('minute', numpy.int32),
                        ('second', numpy.int32)])
    """

    ind = {'date': 1,
           'time': 2,
           'sec_dec': 3,
           'lat': 4,
           'lon': 5,
           'depth': 6,
           'Mw': 61}

    out = []
    with open(fname) as file_:
        reader = csv.reader(file_)

        for line in reader:

            try:
                date_time_dict = _parse_datetime_to_zmap(line[ind['date']].replace('-', '/'),
                                                         line[ind['time']].replace(' ', '0') +
                                                         '.' + line[ind['sec_dec']].replace(' ', ''))
            except ValueError:
                msg = ("Could not parse date/time string '%s' and '%s' to a valid "
                       "time" % (line[ind['date']], line[ind['time']]))
                warnings.warn(msg, RuntimeWarning)
                continue
            event_tuple = (float(line[ind['lon']]),
                           float(line[ind['lat']]),
                           int(date_time_dict['year']),
                           int(date_time_dict['month']),
                           int(date_time_dict['day']),
                           float(line[ind['Mw']]),
                           float(line[ind["depth"]]),
                           int(date_time_dict['hour']),
                           int(date_time_dict['minute']),
                           int(date_time_dict['second']))
            out.append(event_tuple)
    return out


def read_jma_csv(fname):
    raise NotImplementedError("not implemented yet!")