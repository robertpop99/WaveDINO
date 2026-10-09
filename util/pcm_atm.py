import torch
import torchvision


def _pcm_setup(rows, cols, psizex, psizey, device='cpu'):
    n = rows * cols
    #     calculate distance between points in a matrix
    yy, xx = torch.meshgrid(torch.arange(0, rows), torch.arange(0, cols), indexing='xy')
    xxv = xx.reshape(n, 1).to(device)
    yyv = yy.reshape(n, 1).to(device)
    xxv = xxv * psizex
    yyv = yyv * psizey
    dx = xxv.repeat(1, n) - xxv.T.repeat(n, 1)
    dy = yyv.repeat(1, n) - yyv.T.repeat(n, 1)
    rgrid = torch.sqrt(dx ** 2 + dy ** 2).to(device)

    return rgrid, n


def _pcm_calc(rgrid, n, rows, cols, maxvar, alpha, covmodel_type, N, device='cpu'):
    if covmodel_type == 0:
        vcm = maxvar * torch.exp(-alpha * rgrid)
    else:
        raise NotImplementedError

    Z = torch.randn(N, n).to(device)
    V = torch.linalg.cholesky(vcm, upper=True)
    X = torch.matmul(Z, V)
    atm_pets = X.view(N, rows, cols)

    return atm_pets


def pcmc_atm(rows, cols, maxvar, alpha, covmodel_type, N, psizex, psizey, device='cpu'):
    rgrid, n = _pcm_setup(rows, cols, psizex, psizey, device)
    atm_pets = _pcm_calc(rgrid, n, rows, cols, maxvar, alpha, covmodel_type, N, device)

    return atm_pets


class PcmcAtm:
    def __init__(self, rows=100, cols=100, psizex=1, psizey=1, device='cpu'):
        self.rows = rows
        self.cols = cols
        self.device = device

        self.rgrid, self.n = _pcm_setup(rows, cols, psizex, psizey, device)

        self.max_vars = [5.5, 6.5, 7.5, 8.25, 9.]
        self.alphas = [0.004, 0.006, 0.008, 0.012, 0.016]

    def generate_atm(self, maxvar, alpha, N=1, covmodel_type=0):
        atm_pets = _pcm_calc(self.rgrid, self.n, self.rows, self.cols, maxvar, alpha, covmodel_type, N, self.device)

        return atm_pets

    def generate_noise(self, x):
        ind = torch.randint(low=0, high=5, size=(2,))

        noise = self.generate_atm(self.max_vars[ind[0]], self.alphas[ind[1]], x.shape[0])

        # noise[noise > 30] = 30
        # noise[noise < -30] = -30
        # noise = (noise + 30) / 30 - 1

        transformer = torchvision.transforms.Resize((x.shape[-2], x.shape[-1]))
        noise = transformer(noise)
        noise = noise.view(x.shape)

        return noise

