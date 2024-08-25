import streamlit as st
import time
import numpy as np
from utils import *
import pandas as pd
import os
import scipy.io as sio
from common import *
from PIL import Image
from utils import *
from plotter import *

st.set_page_config(page_title="DST Demo", page_icon="📈")
add_logo()
st.markdown("# DST Demo")
st.sidebar.header("DST Demo")

st.write("Please select the options that apply to your use case.")
map_type = st.selectbox(
    'What is the type of your georeferenced map?',
    ["fusion", "lidar"])
am_bin = st.checkbox('I have a measurement file for the known transmitters in the region within the frequency band I am interested in.')
um_bin = st.checkbox('I have unassociated measurements within the frequency band I am interested in.')
mode_bin = st.selectbox(
    'What is the type of your georeferenced map?',
    ["none", "fspl", "TIREM"])


map_file = st.file_uploader("Choose a map file")



if map_file is not None:
    x = sio.loadmat(map_file)
    map_struct = x['SLC']
    SLC = map_struct[0][0]
    #st.write(SLC)
    column_map = dict(zip([name for name in SLC.dtype.names], [i for i in range(len(SLC.dtype.names))]))
    #st.write(column_map)
    en = SLC[column_map["axis"]]
    #st.write(en[0].shape)
    zonenum = SLC[column_map["utmZoneNum"]]
    zoneltr = SLC[column_map["utmZoneLtr"]]
    #st.write(zonenum)
    #st.write(zoneltr)
    lon1, lat1 = utm_to_wgs84(en[0,0], en[0,2], zonenum, zoneltr[0])
    #st.write(lon1)
    #st.write(lat1)
    lon2, lat2 = utm_to_wgs84(en[0,0], en[0,3], zonenum, zoneltr[0])
    #st.write(lon2)
    #st.write(lat2)
    lon3, lat3 = utm_to_wgs84(en[0,1], en[0,2], zonenum, zoneltr[0])
    #st.write(lon3)
    #st.write(lat3)
    lon4, lat4 = utm_to_wgs84(en[0,1], en[0,3], zonenum, zoneltr[0])
    #st.write(lon4)
    #st.write(lat4)
    num_samples_per_side = 100
    lats_top = np.linspace(lat1, lat2, num_samples_per_side)
    lons_top = np.linspace(lon1, lon2, num_samples_per_side)

    lats_right = np.linspace(lat1, lat3, num_samples_per_side)
    lons_right = np.linspace(lon1, lon3, num_samples_per_side)

    lats_bottom = np.linspace(lat3, lat4, num_samples_per_side)
    lons_bottom = np.linspace(lon3, lon4, num_samples_per_side)

    lats_left = np.linspace(lat4, lat2, num_samples_per_side)
    lons_left = np.linspace(lon4, lon2, num_samples_per_side)

    # Combine the points from all sides to form the rectangle
    rectangle_points = np.vstack((
        np.column_stack((lats_top, lons_top)),
        np.column_stack((lats_right, lons_right)),
        np.column_stack((lats_bottom, lons_bottom)),
        np.column_stack((lats_left, lons_left))
    ))



    df = pd.DataFrame(
       #np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        #np.array([[lat1, lon1],[lat2, lon2],[lat3, lon3], [lat4, lon4]]),
        rectangle_points,
    columns=['lat', 'lon'])

    resize_map(df, 700, 400)
    #st.map(df, size=40, color='#0044ff', use_container_width=False)

    if map_type == "fusion":
        map_ = SLC[column_map["dem"]] + 0.3048 * SLC[column_map["hybrid_bldg"]]
    elif map_type == "lidar":
        map_ = SLC[column_map["data"]]

    plotter(map_, "Digital Elevation Map")





meas_file= None
if am_bin:
    meas_file = st.file_uploader("Choose a measurement file")

if meas_file is not None and map_file is not None:

    N1 = map_.shape[0]
    N2 = map_.shape[1]

    UTM_long = np.linspace(en[0, 2], en[0, 3] - SLC[column_map["cellsize"]], N1)
    UTM_lat = np.linspace(en[0, 0], en[0, 1] - SLC[column_map["cellsize"]], N2)

    x = sio.loadmat(meas_file)

    coordinates = np.array(x["endpoint_info"][0][0][0])
    names = np.array(x["endpoint_info"]["names"])
    names = ["cbrssdr1-hospital-comp", "cbrssdr1-smt-comp", "cbrssdr1-ustar-comp", "cbrssdr1-bes-comp",
                          "cbrssdr1-honors-comp", "cbrssdr1-fm-comp", "ebc-nuc1-b210", "web-nuc1-b210",
                          "sagepoint-nuc1-b210", "law73-nuc1-b210", "garage-nuc1-b210", "madsen-nuc1-b210",
                          "cnode-wasatch-dd-b210", "cnode-moran-dd-b210", "cnode-ustar-dd-b210"]

    options = st.multiselect(
        'Which of the known transmitters are active?',
        names)
    measurements = np.array(x["endpoint_info"]["measurements"][0][0])
    x_list = []
    y_list = []
    end_x_list = []
    end_y_list = []
    for option in options:
        endpoint_name = option
        idx = names.index(endpoint_name)
        endpoint_coords = coordinates[:, idx]
        x_e,y_e, _ = lon_lat_to_grid_xy(endpoint_coords[1], endpoint_coords[0], SLC, column_map)
        end_x_list.append(x_e)
        end_y_list.append(y_e)
        data = measurements
        meas_data = data[:, 3 * idx:3 * idx + 3]
        meas_data = meas_data[meas_data[:, 0] != 0, :]
        for i in range(len(meas_data[:,2])):

            x_,y_,_ = lon_lat_to_grid_xy(meas_data[i, 2], meas_data[i, 1], SLC, column_map)
            x_list.append(x_)
            y_list.append(y_)
    plotter3(map_, x_list, y_list, UTM_long, UTM_lat, end_x_list, end_y_list, options, "Active Endpoints & Measurement Locations", 3)

    for option in options:
        endpoint_name = option
        bs_is_tx = st.selectbox(
            f"Is {endpoint_name} a TX or RX?",
            ["TX", "RX"]
        )
        tx_height = st.number_input('TX height: ')
        rx_height = st.number_input('RX height: ')

        idx = names.index(endpoint_name)
        endpoint_coords = coordinates[:, idx]

        data = measurements
        meas_data = data[:, 3 * idx:3 * idx + 3]
        meas_data = meas_data[meas_data[:, 0] != 0, :]

        x = np.zeros((len(meas_data[:, 2]), 1), dtype=int)
        y = np.zeros((len(meas_data[:, 2]), 1), dtype=int)
        valid_idx = []
        invalid_idx = []

        for i in range(len(meas_data[:, 2])):
            [a, b, indx] = lon_lat_to_grid_xy(meas_data[i, 2], meas_data[i, 1], SLC, column_map)
            x[i] = int(a)
            y[i] = int(b)

            if a >= 0 and b >= 0 and a < SLC[column_map["ncols"]] and b < SLC[column_map["nrows"]]:
                valid_idx.append(i)
            else:
                invalid_idx.append(i)

        x = x[valid_idx]
        y = y[valid_idx]
        meas_data = meas_data[valid_idx, :]

        [BS_x, BS_y, indx] = lon_lat_to_grid_xy(endpoint_coords[1], endpoint_coords[0], SLC,
                                                column_map)  # establish basestation pixel location.


        key = np.hstack([x, y])
        value = 10 ** (meas_data[:, 0] / 10)  # Must be averaged in linear space

        rows_to_be_deleted = dict()
        for i in range(key.shape[0]):
            rows_to_be_deleted[i] = []
            for j in range(key.shape[0]):

                if i != j and (key[i, :] == key[j, :]).all():
                    rows_to_be_deleted[i].append(j)

        del_list1 = []
        del_list2 = []

        new_vals = np.zeros((value.shape))
        for i in range(key.shape[0]):
            new_vals[i] = value[i]

            if len(rows_to_be_deleted[i]) != 0:
                for j in rows_to_be_deleted[i]:
                    if j in del_list1:
                        break
                    else:
                        new_vals[i] = float(np.sum([value[rows_to_be_deleted[i]]]) + value[i]) / (
                                len(rows_to_be_deleted[i]) + 1)

                        for elem in zip(rows_to_be_deleted[i]):
                            del_list1.extend(elem)
                            del_list2.extend(elem)
                            del_list1.append(i)

        key_del = np.delete(key, del_list2, 0)
        value_del = np.delete(new_vals, del_list2, 0)
        value_lin = 10 * np.log10(value_del)
        meas_data_unique = np.column_stack((value_lin, key_del))

        ## Generate Range Map
        x = x.flatten()
        y = y.flatten()

        [X, Y] = np.meshgrid(range(1, map_.shape[1] + 1), range(1, map_.shape[0] + 1))

        X0 = X - BS_x
        Y0 = Y - BS_y
        X = X0 * SLC[column_map["cellsize"]]
        Y = Y0 * SLC[column_map["cellsize"]]
        B = np.sqrt(X ** 2 + Y ** 2)

        H = np.zeros(map_.shape)  # height difference between antenna locations
        if bs_is_tx == "TX":
            h_tx = map_[BS_y, BS_x] + tx_height  # map elevation plus antenna height
            H = h_tx - (map_ + rx_height)  # height of TX minus pixel elev and Rx antenna height
        else:
            h_rx = map_[BS_y, BS_x] + rx_height  # map elevation plus antenna height
            H = h_rx - (map_ + tx_height)  # height of TX minus pixel elev  and Rx antenna height

        R = np.sqrt(H ** 2 + B ** 2)

        # Distance to the BS feature
        dist_to_BS = np.log10(1 + R)

        ## Get range value for unique x/y pairs

        # for all meas_data data
        meas_data_r = np.zeros((meas_data.shape[0], 1))

        for i in range(meas_data.shape[0]):
            meas_data_r[i] = R[y[i] - 1, x[i] - 1]

        meas_data_ru = np.zeros((meas_data_unique.shape[0], 1))

        for i in range(meas_data_unique.shape[0]):
            meas_data_ru[i] = R[meas_data_unique[i, 2].astype(int) - 1, meas_data_unique[i, 1].astype(int) - 1]

        ## Pull corresponding unique TIREM Values

        if mode_bin == "none" or mode_bin == "fspl":
            tirem_rssi = np.zeros(tirem_rssi.shape)

        plotter(tirem_rssi, "Initial TIREM Predictions w/o Antenna Pattern")


        ## Radiation Pattern Generation
        if add_radiation_patt:

            directory = os.listdir(radiation_folderdir)

            if radiation_filename not in directory:
                errorMessage = 'Error: The radiation pattern file does not exist in the folder:\n ' + radiation_folderdir
                warnings.warn(errorMessage)

            print('Now reading ' + radiation_filename + "\n")

            tirem_preds = tirem_rssi
            map = map_
            map_res = SLC[column_map["cellsize"]]
            tx_antenna_raster_idx = [BS_y, BS_x]
            tx_antenna_height = tx_height
            rx_antenna_height = rx_height
            antenna_threeD_gain = read_antenna_pattern_file(radiation_folderdir, radiation_filename,
                                                            radiation_file_extns)

            tirem_preds_pattern = add_rad_patt(tirem_preds, map, map_res, tx_antenna_raster_idx, tx_antenna_height,
                                               rx_antenna_height,
                                               antenna_0_az_bearing_angle, antenna_0_el_deviation_angle_from_zenith,
                                               antenna_threeD_gain,
                                               antenna_inclined_tow_bearing_angle)

            ## calculate original error between measured and tirem data
            tirem_rssi = tirem_preds_pattern

        plotter(tirem_rssi, "Initial TIREM Predictions with Antenna Pattern")

        tirem_unique = np.zeros((meas_data_unique.shape[0], 2))
        for i in range(tirem_unique.shape[0]):
            tirem_unique[i, 0] = tirem_rssi[
                meas_data_unique[i, 2].astype(int) - 1, meas_data_unique[i, 1].astype(int) - 1]
            tirem_unique[i, 1] = R[meas_data_unique[i, 2].astype(int) - 1, meas_data_unique[i, 1].astype(int) - 1]

        tirem_error = tirem_unique[:, 0] - meas_data_unique[:, 0]

        ## Select/Cull results by range and noise floor
        '''
        This code further reduces the measured data set to only those values which
        1. fall within the range (can be adjusted)
        2. values that aren't excessively below the noise floor. Noise floor is
        estimated from the values beyond the range limit

        Variables get a pinch weird here. _limit1 suffix denotes the set is range
        limited. _limit2 suffix denotes range and noise floor limited.
        '''

        # max_range = 690; %meters

        # find size of new set
        N_limit = sum(tirem_unique[:, 1] <= max_range)

        # initialize range limited rasters
        tirem_limit = np.zeros((N_limit, 2))
        meas_data_limit = np.zeros((N_limit, 3))
        noise_floor = np.zeros((tirem_unique.shape[0], 1))  # this is used solely to estimate noise floor

        # best to select values at once to maintain "match" between this point.
        # Easier to do if tirem and meas data are combined to n x 5 matrix.
        j = 0
        for i in range(meas_data_unique.shape[0]):
            if tirem_unique[i, 1] <= max_range:

                tirem_limit[j, 0] = tirem_unique[i, 0]
                tirem_limit[j, 1] = tirem_unique[i, 1]

                meas_data_limit[j, 0] = meas_data_unique[i, 0]
                meas_data_limit[j, 1] = meas_data_unique[i, 1]
                meas_data_limit[j, 2] = meas_data_unique[i, 2]

                j = j + 1
            else:
                noise_floor[i] = meas_data_unique[i, 0]

        meas_data_ru1 = meas_data_ru[meas_data_ru < max_range]

        # Calculate the average noise floor value [Should this be a linear average?]

        nz_nf = noise_floor[(noise_floor != 0).flatten()]
        nf_sd = np.std(nz_nf)
        noise_floor = np.median(nz_nf)  # median removes some bias from interference values

        # Remove points lower than the noise floor as well

        N_limit = sum(meas_data_limit[:, 0] >= noise_floor)

        # initialize
        tirem_limit2 = np.zeros((N_limit, 2))
        meas_data_limit2 = np.zeros((N_limit, 3))
        meas_data_ru2 = np.zeros((N_limit, 1))
        # best to select values at once to maintain "match" between this point

        j = 0
        for i in range(meas_data_limit.shape[0]):
            if meas_data_limit[i, 0] >= noise_floor:
                tirem_limit2[j, 0] = tirem_limit[i, 0]
                tirem_limit2[j, 1] = tirem_limit[i, 1]

                meas_data_limit2[j, 0] = meas_data_limit[i, 0]
                meas_data_limit2[j, 1] = meas_data_limit[i, 1]
                meas_data_limit2[j, 2] = meas_data_limit[i, 2]
                meas_data_ru2[j, 0] = meas_data_ru1[i]

                j = j + 1

        meas_data_limit3 = meas_data_limit2
        meas_data_ru3 = meas_data_ru2
        tirem_limit3 = tirem_limit2
        meas_data_ru3_flat = meas_data_ru3.flatten()
        b = np.polyfit(np.log10(meas_data_ru3_flat), meas_data_limit3[:, 0], 1)
        path_loss_exponent = abs(b[0] / 10)



