
from T1_32 import TCI
from EEGNet import MaxNormDefaultConstraint,weights_init
from experiment import EEGDataLoader, Experiment, setup_seed
# dataloader and preprocess
from BCI2B.b_dataloader import BCICompetition4Set2B, extract_segment_trial
from BCI2A.data_preprocess import preprocess4mi, mne_apply, bandpass_cnt
# tools for pytorch
from torch.utils.data import DataLoader
import torch
# tools for numpy as scipy and sys
import logging
import os
import time
import datetime
# tools for plotting confusion matrices and t-SNE
from torchsummary import summary
import numpy as np


# ========================= BCIIV2A data =====================================

def bci4_2b(subject_id):
    dataset = ''
    base_path = ''


    train_filepath_1 = f'{base_path}/{subject_id}01T.gdf'
    train_label_filepath_1 = train_filepath_1.replace(".gdf", ".mat")

    train_filepath_2 = f'{base_path}/{subject_id}02T.gdf'
    train_label_filepath_2 = train_filepath_2.replace(".gdf", ".mat")

    train_filepath_3 = f'{base_path}/{subject_id}03T.gdf'
    train_label_filepath_3 = train_filepath_3.replace(".gdf", ".mat")

    test_filepath_4 = f'{base_path}/{subject_id}04E.gdf'
    test_label_filepath_4 = test_filepath_4.replace(".gdf", ".mat")

    test_filepath_5 = f'{base_path}/{subject_id}05E.gdf'
    test_label_filepath_5 = test_filepath_5.replace(".gdf", ".mat")

    # 加载训练集数据并进行滤波
    train_loader_1 = BCICompetition4Set2B(train_filepath_1, labels_filename=train_label_filepath_1)
    train_cnt_1 = train_loader_1.load()
    train_cnt_1 = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz=4, high_cut_hz=38,
                                                   filt_order=200, fs=250, zero_phase=False),
                            train_cnt_1)

    train_loader_2 = BCICompetition4Set2B(train_filepath_2, labels_filename=train_label_filepath_2)
    train_cnt_2 = train_loader_2.load()
    train_cnt_2 = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz=4, high_cut_hz=38,
                                                   filt_order=200, fs=250, zero_phase=False),
                            train_cnt_2)

    train_loader_3 = BCICompetition4Set2B(train_filepath_3, labels_filename=train_label_filepath_3)
    train_cnt_3 = train_loader_3.load()
    train_cnt_3 = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz=4, high_cut_hz=38,
                                                   filt_order=200, fs=250, zero_phase=False),
                            train_cnt_3)

    # 加载测试集数据并进行滤波
    test_loader_4 = BCICompetition4Set2B(test_filepath_4, labels_filename=test_label_filepath_4)
    test_cnt_4 = test_loader_4.load()
    test_cnt_4 = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz=4, high_cut_hz=38,
                                                  filt_order=200, fs=250, zero_phase=False),
                           test_cnt_4)

    test_loader_5 = BCICompetition4Set2B(test_filepath_5, labels_filename=test_label_filepath_5)
    test_cnt_5 = test_loader_5.load()
    test_cnt_5 = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz=4, high_cut_hz=38,
                                                  filt_order=200, fs=250, zero_phase=False),
                           test_cnt_5)


    train_data_1, train_label_1 = extract_segment_trial(train_cnt_1)
    train_data_2, train_label_2 = extract_segment_trial(train_cnt_2)
    train_data_3, train_label_3 = extract_segment_trial(train_cnt_3)

    test_data_4, test_label_4 = extract_segment_trial(test_cnt_4)
    test_data_5, test_label_5 = extract_segment_trial(test_cnt_5)


    train_data = np.concatenate((train_data_1, train_data_2, train_data_3), axis=0)
    train_label = np.concatenate((train_label_1, train_label_2, train_label_3), axis=0)
    test_data = np.concatenate((test_data_4, test_data_5), axis=0)
    test_label = np.concatenate((test_label_4, test_label_5), axis=0)


    train_label = train_label - 1
    test_label = test_label - 1


    preprocessed_train = preprocess4mi(train_data)
    preprocessed_test = preprocess4mi(test_data)


    train_loader = EEGDataLoader(preprocessed_train, train_label)
    test_loader = EEGDataLoader(preprocessed_test, test_label)
    train_dl = DataLoader(train_loader, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
    test_dl = DataLoader(test_loader, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    valid_dl = None


    model_id = f'{subject_id}_{share_model_name}'
    folder_path = f'./{dataset}/{subject_id}/'
    os.makedirs(folder_path, exist_ok=True)
    output_file = os.path.join(folder_path, f'{model_id}.log')
    fig_path = folder_path + str(model_id)

    logging.basicConfig(
        datefmt='%Y/%m/%d %H:%M:%S',
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.INFO,
        filename=output_file,
    )
    logging.info("****************  %s for %s! ***************************", model_id, subject_id)

    # 定义模型
    Net = TCI(num_classes=2, chans=3, samples=1125).to(device)
    Net.apply(weights_init)
    logging.info(summary(Net, input_size=(1, 3, 1125)))

    model_optimizer = torch.optim.AdamW(Net.parameters(), lr=lr_model)
    model_constraint = MaxNormDefaultConstraint()
    return train_dl, valid_dl, test_dl, Net, model_optimizer, model_constraint, fig_path


if __name__ == "__main__":
    mi_class = 2
    channels = 3
    samples = 1125
    sample_rate = 250
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 超参数设置
    lr_model = 1e-3
    step_one_epochs = 2000
    batch_size = 16
    kwargs = {'num_workers': 1, 'pin_memory': True}

    print('* * ' * 20)
    print(f"使用设备: {device}")
    print('* * ' * 20)


    for subject_id in [f'B0{i}' for i in range(7, 10)]:
        print(f'start: {subject_id}')
        start_time = time.time()
        setup_seed(521)
        share_model_name = 'BCI2B'


        train_dl, valid_dl, test_dl, Net, model_optimizer, model_constraint, fig_path = bci4_2b(subject_id)


        exp = Experiment(
            model=Net,
            device=device,
            optimizer=model_optimizer,
            train_dl=train_dl,
            test_dl=test_dl,
            val_dl=valid_dl,
            fig_path=fig_path,
            model_constraint=model_constraint,
            step_one=step_one_epochs,
            classes=mi_class,
        )

        # 运行实验
        exp.run()
        end_time = time.time()
        logging.info('Done! Running time %.5f', end_time - start_time)
        print(f'finished {subject_id} ')
        print('- - ' * 20)



