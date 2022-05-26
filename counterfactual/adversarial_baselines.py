# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/06a_adversarial_baselines.ipynb (unless otherwise specified).

__all__ = ['LimeExplanation', 'LimeTabularExplainer', 'LocalApprox', 'ROAR']

# Cell
from .import_essentials import *
from .utils import *
from .training_module import *
from .net import *
from .interface import ABCBaseModule, LocalExplainerBase, GlobalExplainerBase
from .train import train_model

from torch.nn.parameter import Parameter
from torchmetrics.functional.classification import accuracy

import sklearn
from lime.lime_base import LimeBase
# import gurobipy as grb
from functools import partial

# Cell
class LimeExplanation(object):
    def __init__(self, intercept={}, local_exp={}, score={}, local_pred={}):
        self.intercept = intercept
        self.local_exp = local_exp
        self.score = score
        self.local_pred = local_pred

    def __str__(self):
        return str({
            'intercept': self.intercept,
            'local_exp': self.local_exp,
            'score': self.score,
            'local_pred': self.local_pred
        })

class LimeTabularExplainer(object):
    def __init__(self, training_data):
        freq = np.sum(training_data, axis=0)
        freq = freq / len(training_data)
        self.freq = freq
        # kernel_width = None
        kernel_width = np.sqrt(training_data.shape[1]) * .75
        kernel_width = float(kernel_width)

        def kernel(d, kernel_width):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.base = LimeBase(kernel_fn)

    def generate_neighbors(self, x, cat_arrays, cat_idx, num_samples):
        neighbors = np.zeros((num_samples, x.shape[-1]))
        cont_perturbed = x[:, :cat_idx] + np.random.normal(0, 0.1, size=(num_samples, cat_idx))
        cont_perturbed = np.clip(cont_perturbed, 0., 1.)
        _cat_idx = cat_idx
        neighbors[:, :cat_idx] = cont_perturbed
        for col in cat_arrays:
            cat_end_idx = cat_idx + len(col)
            # one_hot_idx = np.random.randint(0, len(col), size=(num_samples,))
            one_hot_idx = np.random.choice(range(len(col)), size=(num_samples,), p=self.freq[cat_idx: cat_end_idx])
            neighbors[:, cat_idx: cat_end_idx] = np.eye(len(col))[one_hot_idx]
            cat_idx = cat_end_idx
        x = x.reshape(1, -1)
        return np.concatenate((x, neighbors), axis=0)

    def explain_instance(self,
                         x,
                         predict_fn,
                         cat_arrays,
                         cat_idx,
                         labels=(1,),
                         top_labels=None,
                         num_features=200,
                         num_samples=5000):
        neighbors = self.generate_neighbors(
            x, cat_arrays=cat_arrays, cat_idx=cat_idx, num_samples=num_samples)
        yss = predict_fn(neighbors) + 1e-6
        # map to regression model
        yss = - np.log(1 / yss - 1)
        distances = sklearn.metrics.pairwise_distances(
                neighbors, neighbors[0].reshape(1, -1), metric="euclidean"
        ).ravel()

        self.class_names = [str(x) for x in range(yss[0].shape[0])]

        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]

        intercept, local_exp, score, local_pred = {}, {}, {}, {}
        for label in labels:
            (intercept[label],
             local_exp[label],
             score[label],
             local_pred[label]) = self.base.explain_instance_with_data(
                 neighbors, yss, distances, label, num_features,
                 model_regressor=sklearn.linear_model.Ridge(alpha=1, fit_intercept=False,)
             )
        return LimeExplanation(
            intercept=intercept,
            local_exp=local_exp,
            score=score,
            local_pred=local_pred
        )

# Cell
class LocalApprox(object):
    def __init__(self, train_X, predict_fn, cat_arrays, cat_idx):
        # self.explainer = LimeTabularExplainer(train_X, class_names=['0', '1'], discretize_continuous=False)
        self.explainer = LimeTabularExplainer(train_X)
        self.predict_fn = predict_fn
        self.cat_arrays = cat_arrays
        self.cat_idx = cat_idx

    def extract_weights(self, x_0, shift=0.1):
        # exp = self.explainer.explain_instance(x_0, self.predict_fn, top_labels=1, num_features=200, num_samples=1000)
        exp = self.explainer.explain_instance(x_0,
                                              self.predict_fn,
                                              self.cat_arrays,
                                              self.cat_idx,
                                            #   top_labels=1,
                                              num_features=200,
                                              num_samples=5000)
        coefs = exp.local_exp[1]

        intercept = exp.intercept[1]
        coefs = sorted(coefs, key=lambda x: x[0])

        w = np.array([e[1] for e in coefs])
        # b = intercept - max(self.predict_fn(x_0.reshape(1, -1)).squeeze())
        # b = -shift - np.dot(w, x_0)
        # print('w: ', w)
        # print('b: ', b )

        return w, intercept #np.array(b).reshape(1,)

# Cell
class ROAR(LocalExplainerBase):
    def __init__(self, x: torch.Tensor, model: BaselineModel, n_iters: int = 50, max_delta: float = 0.1):
        def pred_fn(x):
            if len(x.shape) == 1:
                x = torch.from_numpy(x).float().unsqueeze(dim=0)
            else:
                x = torch.from_numpy(x).float()
            prob = model(x).view(-1, 1)
            return torch.cat((1-prob, prob), dim=1).cpu().detach().numpy()

        super().__init__(x, model)
        train_X, y = model.train_dataset[:]
        train_X = train_X.cpu().detach().numpy()
        self.x = x.clone()
        self.lime = LocalApprox(train_X, pred_fn, model.cat_arrays, model.cat_idx)
        self.cf = nn.Parameter(x.clone(), requires_grad=True)
        self.max_delta = max_delta
        self.n_iters = n_iters
        self.dim = x.size(-1)

    def forward(self):
        cf = self.cf * 1.0
        return cat_normalize(cf,
                             self.model.cat_arrays,
                             len(self.model.continous_cols),
                             hard=False)

    # def adv_attack(self, coef, cf, target):
    #     # Model initialization
    #     model = grb.Model("qcp")
    #     model.params.NonConvex = 2
    #     model.setParam('OutputFlag', False)
    #     # model.params.threads = 64
    #     model.params.IterationLimit = 1e3

    #     delta = model.addMVar(self.dim,
    #                           lb=float('-inf'), ub=float('inf'),
    #                           vtype=grb.GRB.CONTINUOUS, name="delta")
    #     # Set objective
    #     obj = np.abs(cf @ delta + np.dot(cf, coef) - target)
    #     model.setObjective(obj, grb.GRB.MAXIMIZE)
    #     # model.setObjective(y, grb.GRB.MAXIMIZE)

    #     # set constrains
    #     model.addConstr(-self.max_delta <= delta)
    #     model.addConstr(delta <= self.max_delta)

    #     # optimize
    #     model.optimize()
    #     return [_delta.x for _delta in delta]

    def configure_optimizers(self):
        return torch.optim.RMSprop([self.cf], lr=0.1)

    def adv_attack(self, coef, cf, target, eps=0.1):
        # coef = coef.reshape((-1, 1))
        delta = torch.zeros_like(coef).uniform_(-eps, eps)
        delta.requires_grad = True
        alpha = eps * 1.25

        for _ in range(7):
            pred = cf @ coef + cf @ delta
            y_pred = 1 / (1 + torch.exp(-(pred)))
            loss = F.mse_loss(y_pred, target.reshape(-1, 1))
            loss.backward()
            scaled_g = delta.grad.detach()#.sign()
            delta.data = l_inf_proj(delta + alpha * scaled_g, eps=eps)
            delta.grad.zero_()
            if torch.linalg.norm(scaled_g).item() < 1e-3:
                break
        return delta.detach()

    def generate_cf(self):
        coef, intercept = self.lime.extract_weights(self.x)
        optim = self.configure_optimizers()

        # x_0 = torch.from_numpy(x_0)
        coef = torch.from_numpy(deepcopy(coef)).float()
        coef = coef.reshape((-1, 1))
        coef_ = deepcopy(coef)
        g = 0
        y_pred = 1 / (1 + torch.exp(-(self.x @ coef)))
        # print(f'y_lime: {y_pred}, y_model: {self.model(self.x)}')
        y_target = torch.ones(y_pred.shape) - torch.round(y_pred)

        for i in range(self.n_iters):
            cf = self()
            delta = self.adv_attack(coef, cf.detach(), y_target, self.max_delta)
            coef = coef + delta
            cf.retain_grad()
            # y_pred = cf @ coef
            y_pred = 1 / (1 + torch.exp(-(cf @ coef)))
            # loss = F.mse_loss(cf @ coef, y_target) + 0.5 * F.mse_loss(self.x, cf)
            loss = F.binary_cross_entropy(y_pred, y_target) + 0.5 * F.mse_loss(self.x, cf)
            optim.zero_grad()
            loss.backward()
            optim.step()
            # check lime
            # if i % 10 == 0:
            #     print(f'y_lime: {1 / (1 + torch.exp(-(cf @ coef)))}, y_model: {self.model(cf)}')

        cf = self.cf * 1.0
        return cat_normalize(cf.detach(),
                             self.model.cat_arrays,
                             len(self.model.continous_cols),
                             hard=True)