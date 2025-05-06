import numpy as np
import mne
from scipy.io import loadmat
import os


class BCICompetition4Set2B:

    def __init__(self, filename, labels_filename=None):
        self.filename = filename
        self.labels_filename = labels_filename

    def load(self):
        cnt = self.extract_data()
        events, artifact_trial_mask = self.extract_events(cnt)
        cnt.info['temp'] = {'events': events, 'artifact_trial_mask': artifact_trial_mask}
        return cnt

    def extract_data(self):
        raw_gdf = mne.io.read_raw_gdf(self.filename, stim_channel="auto", verbose='ERROR',
                                      exclude=(['EOG:ch01', 'EOG:ch02', 'EOG:ch03']))

        raw_gdf.rename_channels(
            {'EEG:C3': 'C3', 'EEG:Cz': 'Cz',  'EEG:C4': 'C4', }
                                )


        raw_gdf.load_data()
        # correct nan values
        data = raw_gdf.get_data()

        for i_chan in range(data.shape[0]):
            # first set to nan, than replace nans by nanmean.
            this_chan = data[i_chan]
            data[i_chan] = np.where(
                this_chan == np.min(this_chan), np.nan, this_chan
            )
            mask = np.isnan(data[i_chan])
            chan_mean = np.nanmean(data[i_chan])
            data[i_chan, mask] = chan_mean

        gdf_events = mne.events_from_annotations(raw_gdf)
        raw_gdf = mne.io.RawArray(data, raw_gdf.info, verbose="ERROR")
        # remember gdf events
        # raw_gdf.info["gdf_events"] = gdf_events
        raw_gdf.info['temp'] = {'gdf_events': gdf_events}
        return raw_gdf

    def extract_events(self, raw_gdf):
        # all events
        # events, name_to_code = raw_gdf.info["gdf_events"]
        events, name_to_code = raw_gdf.info['temp']['gdf_events']

        if "769" and "770"  in name_to_code :
            train_set = True
        else:
            train_set = False
            assert (
                # "cue unknown/undefined (used for BCI competition) "
                "783" in name_to_code
            )

        if train_set:
            if self.filename[-10:] == "B0102T.gdf":
                trial_codes = [4,5]
            else:
                trial_codes = [10,11]
            trial_mask = [ev_code in trial_codes for ev_code in events[:, 2]]
            trial_events = events[trial_mask]

            # assert len(trial_events) == 120, "Got {:d} markers".format(
            #     len(trial_events)
            # )

        else:
            if self.filename[-10:] == "B0504E.gdf":
                trial_codes = [5]
            else:
                trial_codes = [11]  # "unknown" class
            trial_mask = [ev_code in trial_codes for ev_code in events[:, 2]]
            trial_events = events[trial_mask]

            # assert len(trial_events) == 160, "Got {:d} markers".format(
            #     len(trial_events)
            # )

        # print('self.filename[-8:-5]: ', self.filename[-8:-5])
        if train_set :
            if self.filename[-10:] == "B0102T.gdf":
                trial_events[:, 2] = trial_events[:, 2] - 3
            else:
                trial_events[:, 2] = trial_events[:, 2] - 9

        else:
            if self.filename[-10:] == "B0504E.gdf":
                trial_events[:, 2] = trial_events[:, 2] - 4
            else:
                trial_events[:, 2] = trial_events[:, 2] - 10

        # possibly overwrite with markers from labels file
        if self.labels_filename is not None:
            classes = loadmat(self.labels_filename)["classlabel"].squeeze()
            if train_set:  # 确保另外给的train_label和train data中的label一样
                np.testing.assert_array_equal(trial_events[:, 2], classes)
            trial_events[:, 2] = classes
        unique_classes = np.unique(trial_events[:, 2])
        assert np.array_equal(
            [1, 2,], unique_classes
        ), "Expect 1,2, as class labels, got {:s}".format(
            str(unique_classes)
        )

        # now also create 0-1 vector for rejected trials
        if train_set:
            if self.filename[-10:] == "B0102T.gdf":
                trial_start_events = events[events[:, 2] == 3]
            else:
                trial_start_events = events[events[:, 2] == 9]
        else:
            if self.filename[-10:] == "B0504E.gdf":
                trial_start_events = events[events[:, 2] == 3]
            else:
                trial_start_events = events[events[:, 2] == 9]  # 768 start a trail
        assert len(trial_start_events) == len(trial_events)
        artifact_trial_mask = np.zeros(len(trial_events), dtype=np.uint8)
        artifact_events = events[events[:, 2] == 1]

        for artifact_time in artifact_events[:, 0]:
            i_trial = trial_start_events[:, 0].tolist().index(artifact_time)
            artifact_trial_mask[i_trial] = 1

        return trial_events, artifact_trial_mask


def extract_segment_trial(raw_gdb, baseline=(-0.5, 0), duration=4):
    '''
    get segmented data and corresponding labels from raw_gdb.
    :param raw_gdb: raw data
    :param baseline: unit: second. baseline for the segment data. The first value is time before cue.
                     The second value is the time after the Mi duration. Positive values represent the time delays,
                     negative values represent the time lead.
    :param duration: unit: seconds. mi duration time
    :return: array data: trial data, labels
    '''
    events = raw_gdb.info['temp']['events']
    raw_data = raw_gdb.get_data()
    freqs = raw_gdb.info['sfreq']
    mi_duration = int(freqs * duration)
    duration_before_mi = int(freqs * baseline[0])
    duration_after_mi = int(freqs * baseline[1])

    labels = np.array(events[:, 2])

    trial_data = []
    for i_event in events:  # i_event [time, 0, class]
        segmented_data = raw_data[:,
                         int(i_event[0]) + duration_before_mi:int(i_event[0]) + mi_duration + duration_after_mi]
        assert segmented_data.shape[-1] == mi_duration - duration_before_mi +duration_after_mi
        trial_data.append(segmented_data)
    trial_data = np.stack(trial_data, 0)

    return trial_data, labels



if __name__ == '__main__':

    train_filepath = ''
    test_filepath = ''


    train_label_filepath = train_filepath.replace(".gdf", ".mat")
    test_label_filepath = test_filepath.replace(".gdf", ".mat")

    train_loader = BCICompetition4Set2B(
        train_filepath, labels_filename=train_label_filepath
    )
    test_loader = BCICompetition4Set2B(
        test_filepath, labels_filename=test_label_filepath
    )
    train_cnt = train_loader.load()
    test_cnt = test_loader.load()
    print('shae===',type(train_cnt))

    print(train_cnt.get_data().shape)
    print(test_cnt.get_data().shape)


    train_data, train_label = extract_segment_trial(train_cnt)
    test_data, test_label = extract_segment_trial(test_cnt)

    merged_data = np.concatenate((train_data, test_data), axis=0)
    merged_label = np.concatenate((train_label, test_label), axis=0)
    print(merged_data.shape)
    print(merged_label.shape)



