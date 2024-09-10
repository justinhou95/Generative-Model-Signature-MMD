import numpy as np

class PDV4:
    def __init__(
        self,
        beta0=0.04,
        beta1=-0.13,
        beta2=0.65,
        lamb10=55,
        lamb11=10,
        theta1=0.25,
        lamb20=20,
        lamb21=3,
        theta2=0.5,
        ddt=1 / 365,
        day_timestep=1,
        mu=0.1,
        *args,
        **kwargs,
    ) -> None:
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.lamb10 = lamb10
        self.lamb11 = lamb11
        self.theta1 = theta1
        self.lamb20 = lamb20
        self.lamb21 = lamb21
        self.theta2 = theta2
        self.ddt = ddt
        self.day_timestep = day_timestep
        self.dt = self.ddt / self.day_timestep
        self.mu = mu

    def simulate(
        self,
        n_sample: int,
        n_timestep: int,
        r10_init=0.078,
        r11_init=0.16,
        r20_init=0.074,
        r21_init=0.016,
    ):
        """
        prices: L+1
        rxx: L+1
        r: L
        sigma: L

        """
        n_timestep = n_timestep * self.day_timestep

        r10 = np.ones(shape=[n_sample, n_timestep + 1, 1]) * r10_init
        r11 = np.ones(shape=[n_sample, n_timestep + 1, 1]) * r11_init
        r20 = np.ones(shape=[n_sample, n_timestep + 1, 1]) * r20_init
        r21 = np.ones(shape=[n_sample, n_timestep + 1, 1]) * r21_init
        log_return = np.zeros(shape=[n_sample, n_timestep + 1, 1])

        sigma = np.ones(shape=[n_sample, n_timestep, 1])
        r1 = np.ones(shape=[n_sample, n_timestep, 1])
        r2 = np.ones(shape=[n_sample, n_timestep, 1])

        for t in range(n_timestep):
            # compute sigma
            r1[:, t] = (1 - self.theta1) * r10[:, t] + self.theta1 * r11[:, t]
            r2[:, t] = (1 - self.theta2) * r20[:, t] + self.theta2 * r21[:, t]
            sigma[:, t] = (
                self.beta0 + self.beta1 * r1[:, t] + self.beta2 * np.sqrt(r2[:, t])
            )
            # update R_{i,j}
            dw = np.random.normal(loc=0, scale=np.sqrt(self.dt), size=[n_sample, 1])
            r10[:, t + 1] = r10[:, t] + self.lamb10 * (
                sigma[:, t] * dw - r10[:, t] * self.dt
            )
            r11[:, t + 1] = r11[:, t] + self.lamb11 * (
                sigma[:, t] * dw - r11[:, t] * self.dt
            )
            r20[:, t + 1] = (
                r20[:, t] + self.lamb20 * (sigma[:, t] ** 2 - r20[:, t]) * self.dt
            )
            r21[:, t + 1] = (
                r21[:, t] + self.lamb21 * (sigma[:, t] ** 2 - r21[:, t]) * self.dt
            )
            # update logreturn
            log_return[:, t + 1] = sigma[:, t] * dw + self.mu * self.dt

        log_return = log_return[:, :: self.day_timestep]
        sigma = sigma[:, :: self.day_timestep]
        r10 = r10[:, :: self.day_timestep]
        r11 = r11[:, :: self.day_timestep]
        r20 = r20[:, :: self.day_timestep]
        r21 = r21[:, :: self.day_timestep]
        r1 = r1[:, :: self.day_timestep]
        r2 = r2[:, :: self.day_timestep]

        prices = np.cumprod(np.exp(log_return), axis=1)

        return prices, log_return, sigma, r10, r11, r20, r21, r1, r2