import numpy as np
import torch
torch.set_float32_matmul_precision('high')
from train import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def simulate_BM(n_sample, dt, n_timestep):
    noise = torch.randn(size=(n_sample, n_timestep))
    paths_incr = noise * torch.sqrt(torch.tensor(dt))
    paths = torch.cumsum(paths_incr, axis=1)
    BM_paths = torch.cat([torch.zeros((n_sample, 1)), paths], axis=1)
    BM_paths = BM_paths[..., None]
    return BM_paths


def simulate_BS(n_sample, dt, n_timestep, mu, sigma):
    time_grid = torch.linspace(0, dt * n_timestep, n_timestep + 1)
    time_paths = time_grid.expand([n_sample, n_timestep + 1])[..., None]
    BM_paths = simulate_BM(n_sample, dt, n_timestep)
    BS_paths = torch.exp(sigma * BM_paths + (mu - 0.5 * sigma**2) * time_paths)
    return BS_paths


start_date = '2000-01-01'
end_date = '2023-01-01'
t_date = pd.date_range(start=start_date, end=end_date)
n_timestep = len(t_date )
dt = 1/12
mu = 0.1
sigma = 0.2

def get_bs_df():
    
    df = pd.DataFrame({'Date': t_date , 'spx': t_date })
    df.set_index('Date', inplace=True)

    BS_paths = simulate_BS(1, dt, n_timestep-1, mu, sigma)
    path = BS_paths[0,:,0].numpy().astype(np.float64)
    df['spx'] = path
    
    return df

df = get_bs_df()
df.to_csv('data/bs.csv')

df = pd.read_csv('data/bs.csv', index_col=0, parse_dates=True)
df.info()
    
# samples
batch_size = 64 # number of samples in each batch
sample_len = 61 # length of each sample
stride = 60

sample_model = 'Realdt' # GBM, Heston, OU, RealData, Realdt, spx_rates
lead_lag = False # whether to use lead lag transformation
lags = [1] # number of lags to use for lead lag transformation: int or list[int]
seed = 42


# real data parameters
# stride = 50 # for real data
# start_date = '1995-01-01' # start date for real data
# end_date = '2918-09-18' # end date for real data


# signature kernel
static_kernel_type = 'rq' # type of static kernel to use - rbf, rbfmix, rq, rqmix, rqlinear for
n_levels = 5 # number of levels in the truncated signature kernel

# generator
seq_dim = 1 # dimension of sequence vector
activation = 'Tanh' # pytorch names e.g. Tanh, ReLU. NOTE: does NOT change transformer layers'
hidden_size = 64
n_lstm_layers = 1 # number of LSTM layers
conditional = False # feed in history for LSTM generators
hist_len = 11 


noise_dim = 1 # dimension of noise vector
ma = True # whether to use MA noise generator fitted to log returns gaussianized by Lambert W transformation
ma_p = 5


epochs = 300 # number of batches
start_lr = 0.001 # starting learning rate
patience = 100 # number of epochs to wait before reducing lr
lr_factor = 0.5 # factor to multiply lr by for scheduler
early_stopping = patience*3 # number of epochs to wait before no improvement
kernel_sigma = 0.2 # starting kernel_sigma
num_losses = 20


# save all parameters to a dictionary
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

data_params, model_params, train_params = get_params_dicts(vars().copy())

# save parameters to tensorboard
writer = start_writer(data_params, model_params, train_params)


dataloader = get_dataloader(**{**data_params, **model_params, 'bs':True})
kernel = get_signature_kernel(**{**model_params, **train_params})
generator = get_generator(**{**model_params, **data_params})
generator.to(device)

if __name__ == "__main__":
    train(generator, kernel, dataloader, rng, writer, device, **{**train_params, **model_params, **data_params})