# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_interface.ipynb (unless otherwise specified).

__all__ = ['Clamp', 'ExplainerBase', 'LocalExplainerBase', 'GlobalExplainerBase', 'ABCBaseModule']

# Cell
from .import_essentials import *

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

class ExplainerBase(ABC):
    @abstractmethod
    def generate_cf(self, x: torch.Tensor, *args, **kargs):
        """generate cf explanation

        Args:
            x (torch.Tensor): input instance
        """
        raise NotImplementedError

# Cell
class LocalExplainerBase(nn.Module, ExplainerBase):
    def __init__(self,
                 x: torch.Tensor,
                 model: nn.Module):
        super().__init__()
        self.model = model
        self.model.freeze()
        self.x = x
        # self.clamp = Clamp()

    def forward(self):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.Adam([self.cf], lr=0.001)

# Cell
class GlobalExplainerBase(ExplainerBase):
    pass

# Cell
class ABCBaseModule(ABC):
    @abstractmethod
    def model_forward(self, *x):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *x):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *x):
        raise NotImplementedError