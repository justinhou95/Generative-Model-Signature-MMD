import numpy as np
import torch
torch.set_float32_matmul_precision('high')
from train import *

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

from pdv import PDV4

# samples
batch_size = 38 # number of samples in each batch
sample_len = 300 # length of each sample
sample_model = 'Realdt' # GBM, Heston, OU, RealData, Realdt, spx_rates
lead_lag = True # whether to use lead lag transformation
lags = [1] # number of lags to use for lead lag transformation: int or list[int]
seed = 42

# real data parameters
stride = 50 # for real data
start_date = '1995-01-01' # start date for real data
end_date = '2018-09-18' # end date for real data

t = pd.date_range(start=start_date, end=end_date,freq = 'D')
n_timestep = len(t)
df = pd.DataFrame({'Date': t, 'spx': t})
df.set_index('Date', inplace=True)
dt = 1/365

model = PDV4()
prices, log_return, sigma, r10, r11, r20, r21, r1, r2 = model.simulate(1,n_timestep-1)
path = prices[0,:,0].astype(np.float64) * 450
df['spx'] = path
df.to_csv('data/pdv.csv')
df = pd.read_csv('data/pdv.csv', index_col=0, parse_dates=True)
df.info()


# signature kernel
static_kernel_type = 'rq' # type of static kernel to use - rbf, rbfmix, rq, rqmix, rqlinear for
n_levels = 10 # number of levels in the truncated signature kernel

# generator
seq_dim = 1 # dimension of sequence vector
activation = 'Tanh' # pytorch names e.g. Tanh, ReLU. NOTE: does NOT change transformer layers'
hidden_size = 64
n_lstm_layers = 1 # number of LSTM layers
conditional = True # feed in history for LSTM generators
hist_len = 50

noise_dim = 4 # dimension of noise vector
ma = True # whether to use MA noise generator fitted to log returns gaussianized by Lambert W transformation
ma_p = 20

epochs = 10000 # number of batches
start_lr = 0.001 # starting learning rate
patience = 100 # number of epochs to wait before reducing lr
lr_factor = 0.5 # factor to multiply lr by for scheduler
early_stopping = patience*3 # number of epochs to wait before no improvement
kernel_sigma = 0.1 # starting kernel_sigma
num_losses = 20


# save all parameters to a dictionary
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

data_params, model_params, train_params = get_params_dicts(vars().copy())

# save parameters to tensorboard
writer = start_writer(data_params, model_params, train_params)


dataloader = get_dataloader(**{**data_params, **model_params, 'pdv':True})
kernel = get_signature_kernel(**{**model_params, **train_params})
generator = get_generator(**{**model_params, **data_params})
generator.to(device)

if __name__ == '__main__':
    train(generator, kernel, dataloader, rng, writer, device, **{**train_params, **model_params, **data_params})