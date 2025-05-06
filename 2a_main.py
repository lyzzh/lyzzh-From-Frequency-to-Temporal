from model import TCI
# from my_model import TCI
from EEGNet import MaxNormDefaultConstraint,weights_init
from experiment import EEGDataLoader, Experiment, setup_seed
# dataloader and preprocess
from BCI2A.data_loader import BCICompetition4Set2A, extract_segment_trial
from BCI2A.data_preprocess import preprocess4mi, mne_apply, bandpass_cnt
# tools for pytorch
from torch.utils.data import DataLoader
import torch
# tools for numpy as scipy and sys
import logging
import os
import time
# tools for plotting confusion matrices and t-SNE
from torchsummary import summary


# ========================= BCIIV2A data ========= ============================

def bci4_2a():
    dataset = ''
    data_path = ""

    train_filename = "{}T.gdf".format(subject_id)
    test_filename = "{}E.gdf".format(subject_id)
    train_filepath = os.path.join(data_path, train_filename)
    test_filepath = os.path.join(data_path, test_filename)
    train_label_filepath = train_filepath.replace(".gdf", ".mat")
    test_label_filepath = test_filepath.replace(".gdf", ".mat")

    train_loader = BCICompetition4Set2A(
        train_filepath, labels_filename=train_label_filepath
    )
    test_loader = BCICompetition4Set2A(
        test_filepath, labels_filename=test_label_filepath
    )
    train_cnt = train_loader.load()
    test_cnt = test_loader.load()

    # band-pass before segment trials
    # train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    # test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)

    train_cnt = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz=4, high_cut_hz=38,
                                                 filt_order=200, fs=250, zero_phase=False),
                          train_cnt)

    test_cnt = mne_apply(lambda a: bandpass_cnt(a, low_cut_hz=4, high_cut_hz=38,
                                                filt_order=200, fs=250, zero_phase=False),
                         test_cnt)

    train_data, train_label = extract_segment_trial(train_cnt)
    test_data, test_label = extract_segment_trial(test_cnt)

    train_label = train_label - 1
    test_label = test_label - 1

    preprocessed_train = preprocess4mi(train_data)
    preprocessed_test = preprocess4mi(test_data)

    train_loader = EEGDataLoader(preprocessed_train, train_label)
    test_loader = EEGDataLoader(preprocessed_test, test_label)

    train_dl = DataLoader(train_loader, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
    test_dl = DataLoader(test_loader, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    valid_dl = None

    model_id = '%s' % share_model_name
    folder_path = './%s/%s/' % (dataset, subject_id)  # mkdir in current folder, and name it by target's num
    folder = os.path.exists(folder_path)
    if not folder:
        os.makedirs(folder_path)
    output_file = os.path.join(folder_path, '%s.log' % (model_id))
    fig_path = folder_path + str(model_id)   # 用来代码命名
    logging.basicConfig(
        datefmt='%Y/%m/%d %H:%M:%S',
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.INFO,
        filename=output_file,
    )

    logging.info("****************  %s for %s! ***************************", model_id, subject_id)

    if share_model_name == 'TCI':
        Net = TCI(num_classes = mi_class, chans = channels, samples = samples).to(device)



    Net.apply(weights_init)
    Net.apply(weights_init)

    logging.info(summary(Net, input_size=(1, 22, 1125)))

    model_optimizer = torch.optim.AdamW(Net.parameters(), lr=lr_model)
    model_constraint = MaxNormDefaultConstraint()
    return train_dl, valid_dl, test_dl, Net, model_optimizer, model_constraint, fig_path


if __name__ == "__main__":
    mi_class = 4
    channels = 22
    samples = 1125
    sample_rate = 250
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 超参数设置
    lr_model = 1e-3
    step_one_epochs = 1000
    batch_size = 16
    kwargs = {'num_workers': 1, 'pin_memory': True}

    print('* * ' * 20)
    print(f"使用设备: {device}")
    print('* * ' * 20)


    for subject_id in [f'A0{i}' for i in range(1, 10)]:
        print(f'start: {subject_id}')
        start_time = time.time()
        setup_seed(521)
        share_model_name = 'TCI'


        train_dl, valid_dl, test_dl, Net, model_optimizer, model_constraint, fig_path = bci4_2a()

        # 创建实验对象
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


        exp.run()
        end_time = time.time()
        logging.info('Done! Running time %.5f', end_time - start_time)
        print(f'finish {subject_id} ')
        print('- - ' * 20)


