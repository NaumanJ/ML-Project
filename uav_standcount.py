# --------------------------------------------------------
# Name: CornStandCount
# Purpose: Counting the corn stands using plot-level image.
#
# --------------------------------------------------------


import numpy as np
import cv2
import os
import scipy.signal as signal
import peakutils
# import matplotlib.pyplot as plt
# from operator import itemgetter
# from itertools import groupby
import pandas as pd

"""
elv = 40m: min_dist == 15 for peak detection
elv = 60m: min_dist == 7 for peak detection
"""


class CornStandCount:
    def __init__(self):
        self.path = r'/Users/nan.an/Documents/Projects/StandCount/ILMN-CMY1-CMY2-CMY4_20170531_RGB/CMY1/PlotImages/'
        self.filename = ''
        self.maskname = ''
        self.sum_row = []
        self.splitpoint = 0
        self.num_plant = 0
        self.range = ''
        self.column = ''
        self.fieldID = ''
        self.stand_table = ['AbsR' + ',' + 'AbsC' + ',' + 'UAV']
        self.plant_table = ['AbsR' + ',' + 'AbsC' + ',' + 'LocX' + ',' + 'LocY']
        self.plot_img_list = []
        self.plot_rgb_list = []
        self.plot_mask_list = []
        self.peak_idx = []
        self.plant_center_y = 0
        self.min_peak_interval = 15  # this is related to flight elv and resolution

    def get_img_file_names(self):
        files_list = os.listdir(self.path)
        self.plot_img_list = [files_list[i] for i in range(0, len(files_list)) if files_list[i][0] == 'C']
        # remove the file extension
        self.plot_img_list = [os.path.splitext(self.plot_img_list[i])[0] for i in range(0, len(self.plot_img_list))]
        # list plot mask images
        #self.plot_mask_list = [self.plot_img_list[i] for i in range(0, len(self.plot_img_list)) if
        #                       self.plot_img_list[i].split('_')[2] == 'RGB-mask']
        self.plot_mask_list = [self.plot_img_list[i] for i in range(0, len(self.plot_img_list)) if
                               self.plot_img_list[i].split('_')[2] == 'mask']

        # list plot rgb images
        self.plot_rgb_list = [self.plot_img_list[i] for i in range(0, len(self.plot_img_list)) if
                              self.plot_img_list[i].split('_')[2] == 'RGB']

    def meta_parser(self, filename):
        self.filename = filename
        #self.maskname = filename + '-mask'
        self.maskname = self.filename.split('_')[0] + '_' + self.filename.split('_')[1] + '_mask'
        self.column = self.filename.split('_')[0][1:]
        self.range = self.filename.split('_')[1][1:]

    def list_duplicates(self, seq, item):
        start_at = -1
        locs = []
        while True:
            try:
                loc = seq.index(item, start_at + 1)
            except ValueError:
                break
            else:
                locs.append(loc)
                start_at = loc
        return locs

    def plant_count(self):
        self.img = cv2.imread(os.path.join(self.path, self.maskname) + '.tif', 0)
        self.rgb_img = cv2.imread(os.path.join(self.path, self.filename) + '.tif', 1)

        # New canopy coverage mask can be a 0 - 1 binary image. Coverting it to 255.
        if self.img.max() == 1:
            self.img = self.img * 255
        else:
            pass

        if self.img.max() == 0:
            print(self.filename + 'is empty.')
            pass
        else:
            """This is the section for splitting the two-row plot images to single row"""
            rows, cols = self.img.shape
            if rows > cols:
                """The plot image is vertical"""
                self.plot_img = np.rot90(self.img, k=1)  # k == the Number of times the array is rotated by 90 degrees.
                self.plot_rgb_img = np.rot90(self.rgb_img, k=1)
            else:
                """The plot image is horizontal"""
                self.plot_img = self.img
                self.plot_rgb_img = self.rgb_img

            # make the subplot folder for one-row images
            subplot_dir = self.path + 'subplot/'
            if not os.path.exists(subplot_dir):
                os.makedirs(subplot_dir)

            cv2.imwrite(os.path.join(subplot_dir, self.maskname) + '.tif', self.plot_img)
            cv2.imwrite(os.path.join(subplot_dir, self.filename) + '.tif', self.plot_rgb_img)

            # re-read the rotated plot image to avoid OpenCV bug
            self.plot_img = cv2.imread(os.path.join(subplot_dir, self.maskname) + '.tif', 0)

            rows_rot, cols_rot = self.plot_img.shape
            # construct the row-side pixel count curve for finding the split point
            self.sum_row = np.sum(self.plot_img, axis=1)
            # finding the center of the space between two rows
            # peakind = signal.find_peaks_cwt(self.sum_row, widths=np.arange(1, 20))
            self.sum_row = self.sum_row.astype(int)
            peakind = peakutils.indexes(self.sum_row,
                                        thres=0.2,
                                        min_dist=50)  # threshold here is for preventing low LAI CV on each row

            if len(peakind) == 2 and peakind[0] > 20:  # This is for 2-row plots and the first row is not cut into half or more rows in the plot than the desired number.
                self.splitpoint = int(np.mean(peakind))

                # crop plot image
                self.row_1 = self.plot_img[0:self.splitpoint, 0:cols_rot]
                # cv2.imwrite(os.path.join(subplot_dir, self.filename) + "_row1.tif", self.row_1)
                self.row_2 = self.plot_img[self.splitpoint:rows_rot, 0:cols_rot]
                # cv2.imwrite(os.path.join(subplot_dir, self.filename) + "_row2.tif", self.row_2)

                """This is the section for counting the plants"""
                # initialize the numbers
                num_row = 1
                self.num_plant = 0

                for singlerow_img in [self.row_1, self.row_2]:
                    rows, cols = singlerow_img.shape

                    # this is the index of the peak of each row
                    j = 0

                    # create a colorized image for plant labeling
                    singlerow_color = np.dstack((singlerow_img, singlerow_img, singlerow_img))
                    plot_color = np.dstack((self.plot_img, self.plot_img, self.plot_img))

                    # construct the col-side pixel count curve for finding the split point
                    sum_col = np.sum(singlerow_img, axis=0)

                    # smooth the BV curve
                    curve_smooth = signal.savgol_filter(sum_col, window_length=31, polyorder=5, deriv=0)

                    # finding the peaks of the space between two rows -- method 1
                    # peak_idx = signal.find_peaks_cwt(curve_smooth, widths=np.arange(1, 20))

                    # find the peaks of the curve -- method 2
                    # peaks_detect = np.r_[True, curve_smooth[1:] > curve_smooth[:-1]] & np.r_[curve_smooth[:-1] > curve_smooth[1:], True]
                    # peak_idx = []
                    # [peak_idx.append(i) for i in range(0, len(peaks_detect)) if peaks_detect[i] == True and curve_smooth[i] > 175]

                    # find the peaks of the curve -- method 3
                    self.peak_idx = peakutils.indexes(curve_smooth, min_dist=self.min_peak_interval)

                    # the number of peaks is the estimation of the plants
                    num_peaks = len(self.peak_idx)
                    # print('the single row count is %d')%num_peaks

                    # sum the two rows
                    self.num_plant += num_peaks
                    print(self.num_plant)

                    # show the peak
                    for i in range(0, len(self.peak_idx)):
                        # plt.plot(peak_idx[i], curve_smooth[peak_idx[i]], '-o')

                        # extract the pixel digial numbers for each peak column
                        # print(self.peak_idx[i])

                        col_px_dn = singlerow_img[:, self.peak_idx[i]]

                        # refine the row center in each single row image
                        row_y_profile = np.sum(singlerow_img, axis=1)
                        row_y_profile = row_y_profile.astype(int)
                        row_y_peak = peakutils.indexes(row_y_profile, min_dist=50)

                        self.plant_center_y = row_y_peak[0]

                        # show the plant location in coordinates
                        # print(self.peak_idx[i], self.plant_center_y)

                        if num_row == 1:
                            plant_loc = (self.peak_idx[i], self.plant_center_y)
                        elif num_row == 2:
                            plant_loc = (self.peak_idx[i], self.plant_center_y + self.splitpoint)

                        output_line_entry = self.range + ',' + self.column + ',' + str(plant_loc[0]) + ',' + str(
                            plant_loc[1])
                        self.plant_table.append(output_line_entry)

                    j += 1
                    # cv2.imwrite(subplot_dir + self.filename + '_' + str(num_row) + '.tif', singlerow_color)
                    num_row += 1

                    """
                    # show the pixel BV curve
                    plt.plot(range(0, len(sum_col)), sum_col, 'r', range(0, len(sum_col)), curve_smooth, 'b--')
                    plt.xlim(0, cols)
                    plt.show()

                    # show image
                    cv2.imshow('image for plant locations', singlerow_color)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    """
            else:
                pass

    def generate_stand_table(self):
        single_entry = self.range + ',' + self.column + ',' + str(self.num_plant)
        self.stand_table.append(single_entry)

    def generate_stand_csv(self):
        csv_file = open(self.path + 'uav_stand.csv', 'wb')
        for item in self.stand_table:
            csv_file.write('%s\n' % item)
        print('The stand count table is generated.')

    def generate_plant_csv(self):
        plant_csv = open(self.path + 'plant_location.csv', 'wb')
        for item in self.plant_table:
            plant_csv.write('%s\n' % item)
        print('The plant location table is generated.')

    def plant_labeling(self):

        path = self.path + 'subplot/'
        plot_name_list = os.listdir(path)

        for plotname in plot_name_list:

            img_color = cv2.imread(path + plotname, 1)
            basename = plotname.split('.')[0]
            col = plotname.split('_')[0][1:]
            ran = plotname.split('_')[1][1:]

            # convert the plant location table to a pandas dataframe
            df = pd.DataFrame([line.split(',') for line in stand.plant_table],
                              columns=['AbsR', 'AbsC', 'LocX', 'LocY'])
            # drop the title row
            df.drop(df.index[0])
            df_plot = df.loc[(df['AbsC'] == col) & (df['AbsR'] == ran)]

            for i in range(len(df_plot)):
                plant_loc = (int(df_plot.iloc[i]['LocX']), int(df_plot.iloc[i]['LocY']))

                cv2.circle(img_color,
                           plant_loc,
                           radius=10,
                           color=(0, 0, 255),
                           thickness=1,
                           lineType=8,
                           shift=0
                           )
            cv2.imwrite(path + basename + '_stand.tif', img_color)


if __name__ == "__main__":
    stand = CornStandCount()
    stand.get_img_file_names()

    for filename in stand.plot_rgb_list:
        stand.meta_parser(filename)
        print(filename)
        stand.plant_count()
        # print(stand.num_plant)
        stand.generate_stand_table()

    stand.generate_stand_csv()
    stand.generate_plant_csv()

    print('Labeling plant images.....')
    stand.plant_labeling()
    print('Plant labeling is completed.')
