# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05a_baseline_algos.ipynb (unless otherwise specified).

__all__ = ['Clamp', 'VanillaCF', 'DiverseCF', 'ProtoCF', 'VAE_CF']

# Cell
from .import_essentials import *
from .utils import *
# from counterfactual.train import *
from .training_module import *
from .net import *
from .interface import ABCBaseModule, LocalExplainerBase, GlobalExplainerBase

from torch.nn.parameter import Parameter
from torchmetrics.functional.classification import accuracy

# Cell

class Clamp(torch.autograd.Function):
    """
    Clamp parameter to [0, 1]
    code from: https://discuss.pytorch.org/t/regarding-clamped-learnable-parameter/58474/4
    """
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

# Cell

class VanillaCF(LocalExplainerBase):
    def __init__(self, x: torch.tensor, model: BaselineModel, n_iters: int = 1000):
        """vanilla version of counterfactual generation
            - link: https://doi.org/10.2139/ssrn.3063289

        Args:
            x (torch.tensor): input instance
            model (BaselineModel): black-box model
        """
        super().__init__(x, model)
        self.cf = nn.Parameter(self.x.clone(), requires_grad=True)
        self.n_iters = n_iters

    def forward(self):
        cf = self.cf * 1.0
        return cat_normalize(cf, self.model.cat_arrays, len(self.model.continous_cols), False)
        # return cf

    def configure_optimizers(self):
        return torch.optim.RMSprop([self.cf], lr=0.001)

    def compute_regularization_loss(self):
        cat_idx = len(self.model.continous_cols)
        regularization_loss = 0.
        for col in self.model.cat_arrays:
            cat_idx_end = cat_idx + len(col)
            regularization_loss += torch.pow((torch.sum(self.cf[cat_idx: cat_idx_end]) - 1.0), 2)
        return regularization_loss

    def _loss_functions(self, x, c):
        # target
        y_pred = self.model.predict(x)
        y_prime = torch.ones(y_pred.shape) - y_pred

        c_y = self.model(c)
        l_1 = F.binary_cross_entropy(c_y, y_prime.float())
        l_2 = F.mse_loss(x, c)
        return l_1, l_2

    def _loss_compute(self, l_1, l_2):
        return 1.0 * l_1 + 0.5 * l_2

    def generate_cf(self, debug: bool = False):
        optim = self.configure_optimizers()
        clamp = Clamp()
        for i in range(self.n_iters):
            c = self()
            l_1, l_2 = self._loss_functions(self.x, c)
            loss = self._loss_compute(l_1, l_2)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if debug and i % 100 == 0:
                print(f"iter: {i}, loss: {loss.item()}")

            # contrain to [0,1]
            clamp.apply(self.cf)

        cf = self.cf * 1.0
        clamp.apply(self.cf)
        return cat_normalize(cf, self.model.cat_arrays, len(self.model.continous_cols), True)

# Cell

class DiverseCF(LocalExplainerBase):
    def __init__(self, x: torch.tensor, model: CounterfactualTrainingModule, n_iters = 1000):
        """diverse counterfactual explanation
            - link: https://doi.org/10.1145/3351095.3372850

        Args:
            x (torch.tensor): input instance
            model (CounterfactualTrainingModule): black-box model
        """
        self.n_cfs = 5
        super().__init__(x, model)
        # self.cf = nn.Parameter(self.x.repeat(self.n_cfs, 1), requires_grad=True)
        self.cf = nn.Parameter(torch.rand(self.n_cfs, self.x.size(1)), requires_grad=True)
        self.n_iters = n_iters

    def forward(self):
        cf = self.cf * 1.0
        return torch.clamp(cf, 0, 1)

    def configure_optimizers(self):
        return torch.optim.RMSprop([self.cf], lr=0.001)

    def _compute_dist(self, x1, x2):
        return torch.sum(torch.abs(x1 - x2), dim = 0)

    def _compute_proximity_loss(self):
        """Compute the second part (distance from x1) of the loss function."""
        proximity_loss = 0.0
        for i in range(self.n_cfs):
            proximity_loss += self.compute_dist(self.cf[i], self.x1)
        return proximity_loss/(torch.mul(len(self.minx[0]), self.total_CFs))

    def _dpp_style(self, cf):
        det_entries = torch.ones(self.n_cfs, self.n_cfs)
        for i in range(self.n_cfs):
            for j in range(self.n_cfs):
                det_entries[i, j] = self._compute_dist(cf[i], cf[j])

        # implement inverse distance
        det_entries = 1.0 / (1.0 + det_entries)
        det_entries += torch.eye(self.n_cfs) * 0.0001
        return torch.det(det_entries)

    def _compute_diverse_loss(self, c):
        return self._dpp_style(c)

    def _compute_regularization_loss(self):
        cat_idx = len(self.model.continous_cols)
        regularization_loss = 0.
        for i in range(self.n_cfs):
            for col in self.model.cat_arrays:
                cat_idx_end = cat_idx + len(col)
                regularization_loss += torch.pow((torch.sum(self.cf[i][cat_idx: cat_idx_end]) - 1.0), 2)
        return regularization_loss

    def _loss_functions(self, x, c):
        # target
        y_pred = self.model.predict(x)
        y_prime = torch.ones(y_pred.shape) - y_pred

        c_y = self.model(c)
        # yloss
        l_1 = hinge_loss(input=c_y, target=y_prime.float())
        # proximity loss
        l_2 = l1_mean(x, c)
        # diverse loss
        l_3 = self._compute_diverse_loss(c)
        # categorical penalty
        l_4 = self._compute_regularization_loss()
        return l_1, l_2, l_3, l_4

    def _compute_loss(self, *loss_f):
        return sum(loss_f)

    def generate_cf(self, debug: bool = False):
        optim = self.configure_optimizers()
        for i in range(self.n_iters):
            c = self()

            l_1, l_2, l_3, l_4 = self._loss_functions(self.x, c)
            loss = self._compute_loss(l_1, l_2, l_3, l_4)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if  debug and i % 100 == 0:
                print(f"iter: {i}, loss: {loss.item()}")

            # contrain to [0,1]
            self.clamp.apply(self.cf)

        cf = self.cf * 1.0
        cf = torch.clamp(cf, 0, 1)
        # return cf[0]
        return cat_normalize(cf[0].view(1, -1), self.model.cat_arrays, len(self.model.continous_cols), True)

# Cell

class ProtoCF(LocalExplainerBase):
    def __init__(self, x: torch.tensor, model: pl.LightningModule, train_loader: DataLoader, ae: AE, n_iters: int = 1000):
        """vanilla version of counterfactual generation
            - link: https://doi.org/10.2139/ssrn.3063289

        Args:
            x (torch.tensor): input instance
            model (pl.LightningModule): black-box model
        """
        super().__init__(x, model)
        self.cf = nn.Parameter(self.x.clone(), requires_grad=True)
        self.sampled_data, _ = next(iter(train_loader))
        self.sampled_label = self.model.predict(self.sampled_data)
        self.ae = ae
        self.ae.freeze()
        self.n_iters = n_iters

    def forward(self):
        cf = self.cf * 1.0
        # return cat_normalize(cf, self.model.cat_arrays, len(self.model.continous_cols), False)
        return cf

    def configure_optimizers(self):
        return torch.optim.RMSprop([self.cf], lr=0.001)

    def compute_regularization_loss(self):
        cat_idx = len(self.model.continous_cols)
        regularization_loss = 0.
        for col in self.model.cat_arrays:
            cat_idx_end = cat_idx + len(col)
            regularization_loss += torch.pow((torch.sum(self.cf[cat_idx: cat_idx_end]) - 1.0), 2)
        return regularization_loss

    def proto(self, data):
        return self.ae.encoded(data).mean(axis=0).view(1, -1)

    def _loss_functions(self, x, c):
        # target
        y_pred = self.model.predict(x)
        y = torch.ones(y_pred.shape) - y_pred

        data = self.sampled_data[self.sampled_label == y]

        l_1 = F.binary_cross_entropy(self.model(c), y)
        l_2 = 0.1 * F.l1_loss(x, c) + F.mse_loss(x, c)
        l_3 = F.mse_loss(self.ae.encoded(c), self.proto(data))

        return l_1, l_2, l_3

    def _loss_compute(self, l_1, l_2, l_3):
        return l_1 + l_2 + l_3 #+ self.compute_regularization_loss()

    def generate_cf(self, debug: bool = False):
        optim = self.configure_optimizers()
        for i in range(self.n_iters):
            c = self()

            l_1, l_2, l_3 = self._loss_functions(self.x, c)
            loss = self._loss_compute(l_1, l_2, l_3)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if debug and i % 100 == 0:
                print(f"iter: {i}, loss: {loss.item()}")

            # contrain to [0,1]
            self.clamp.apply(self.cf)

        cf = self.cf * 1.0
        self.clamp.apply(self.cf)
        # return cf
        return cat_normalize(cf, self.model.cat_arrays, len(self.model.continous_cols), True)

# Cell
class VAE_CF(CounterfactualTrainingModule):
    def __init__(self, config: Dict, model: pl.LightningModule):
        """
        config: basic configs
        model: the black-box model to be explained
        """
        super().__init__(config)
        self.model = model
        self.model.freeze()
        self.vae = VAE(input_dims=self.enc_dims[0])
        # validity_reg set to 42.0
        # according to https://interpret.ml/DiCE/notebooks/DiCE_getting_started_feasible.html#Generate-counterfactuals-using-a-VAE-model
        self.validity_reg = config['validity_reg'] if 'validity_reg' in config.keys() else 1.0

    def model_forward(self, x):
        """lazy implementation since this method is actually not needed"""
        recon_err, kl_err, x_true, x_pred, cf_label = self.vae.compute_elbo(x, 1 - self.model.predict(x), self.model)
        # return y, c
        return cf_label, x_pred

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr)

    def predict(self, x):
        return self.model.predict(x)

    def compute_loss(self, out, x, y):
        em = out['em']
        ev = out['ev']
        z = out['z']
        dm = out['x_pred']
        mc_samples = out['mc_samples']
        #KL Divergence
        kl_divergence = 0.5*torch.mean(em**2 + ev - torch.log(ev) - 1, axis=1)

        #Reconstruction Term
        #Proximity: L1 Loss
        x_pred = dm[0]
        cat_idx = len(self.continous_cols)
        # recon_err = - \
        #     torch.sum(torch.abs(x[:, cat_idx:-1] -
        #                         x_pred[:, cat_idx:-1]), axis=1)
        recon_err = - torch.sum(torch.abs(x - x_pred), axis=1)

        # Sum to 1 over the categorical indexes of a feature
        for col in self.cat_arrays:
            cat_end_idx = cat_idx + len(col)
            temp = - \
                torch.abs(1.0 - x_pred[:, cat_idx: cat_end_idx].sum(axis=1))
            recon_err += temp

        #Validity
        c_y = self.model(x_pred)
        validity_loss = torch.zeros(1, device=self.device)
        validity_loss += hinge_loss(input=c_y, target=y.float())

        for i in range(1, mc_samples):
            x_pred = dm[i]

            # recon_err += - \
            #     torch.sum(torch.abs(x[:, cat_idx:-1] -
            #                         x_pred[:, cat_idx:-1]), axis=1)
            recon_err += - torch.sum(torch.abs(x - x_pred), axis=1)

            # Sum to 1 over the categorical indexes of a feature
            for col in self.cat_arrays:
                cat_end_idx = cat_idx + len(col)
                temp = - \
                    torch.abs(1.0 - x_pred[:, cat_idx: cat_end_idx].sum(axis=1))
                recon_err += temp

            #Validity
            c_y = self.model(x_pred)
            validity_loss += hinge_loss(c_y, y.float())

        recon_err = recon_err / mc_samples
        validity_loss = -1 * self.validity_reg * validity_loss / mc_samples

        return -torch.mean(recon_err - kl_divergence) - validity_loss


    def training_step(self, batch, batch_idx):
        # batch
        x, _ = batch
        # prediction
        y_hat = self.model.predict(x)
        # target
        y = 1.0 - y_hat

        out = self.vae(x, y)
        loss = self.compute_loss(out, x, y)

        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # batch
        x, _ = batch
        # prediction
        y_hat = self.model.predict(x)
        # target
        y = 1.0 - y_hat

        out = self.vae(x, y)
        loss = self.compute_loss(out, x, y)

        _, _, _, x_pred, cf_label = self.vae.compute_elbo(x, y, self.model)

        cf_proximity = torch.abs(x - x_pred).sum(dim=1).mean()
        cf_accuracy = accuracy(cf_label, y)

        self.log('val/val_loss', loss)
        self.log('val/proximity', cf_proximity)
        self.log('val/cf_accuracy', cf_accuracy)

        return loss

    def validation_epoch_end(self, val_outs):
        return

    def generate_cf(self, x):
        self.vae.freeze()
        y_hat = self.model.predict(x)
        recon_err, kl_err, x_true, x_pred, cf_label = self.vae.compute_elbo(x, 1.-y_hat, self.model)
        return self.model.cat_normalize(x_pred, hard=True)