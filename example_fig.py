from common.utils import load_train_res, train_results_plots

res_1 = load_train_res('./LSTM/save/12_ax/LSTM_1.npy')
res_2 = load_train_res('./LSTM/save/12_ax/LSTM_2.npy')
res_3 = load_train_res('./LSTM/save/12_ax/LSTM_3.npy')

train_results_plots(dir='./fig/', figname='12_ax', names=['LSMT_1', 'LSTM_2', 'LSTM_3'], \
                    numbers=[res_1, res_2, res_3])