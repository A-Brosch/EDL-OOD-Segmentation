import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as dist
from mmseg.registry import MODELS
from mmengine.logging import MessageHub

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def safe_one_hot(tensor, num_classes):
    # Zero out any invalid indices
    mask = tensor < num_classes
    tensor = tensor * mask  # Set invalid entries to 0
    
    
    one_hot = F.one_hot(tensor, num_classes=num_classes)
    
    # Zero out rows corresponding to originally invalid entries
    one_hot = one_hot * mask.unsqueeze(-1)
    
    return one_hot

def get_annealing_coef(global_step, start_step=40000, warmup_steps=8000, max_coef=0.15):
    """
    Linearly anneal from 0 to max_coef over warmup_steps.
    After warmup_steps, coefficient stays at max_coef.
    """

    if global_step < start_step:
        return 0.0
    progress = (global_step - start_step) / float(warmup_steps)
    return min(max_coef, progress * max_coef)


def kl_dirichlet(alpha, prior=None, eps=1e-8):
    """
    KL(Dir(alpha) || Dir(prior))
    alpha: [N, C]
    prior: prior concentration, defaults to uniform (all ones)
    """
    if prior is None:
        prior = torch.ones_like(alpha)

    alpha0 = torch.sum(alpha, dim=1, keepdim=True)
    prior0 = torch.sum(prior, dim=1, keepdim=True)

    # log B(alpha) = sum(log Gamma(alpha_i)) - log Gamma(alpha0)
    log_B_alpha = torch.sum(torch.lgamma(alpha + eps), dim=1) - torch.lgamma(alpha0 + eps)
    log_B_prior = torch.sum(torch.lgamma(prior + eps), dim=1) - torch.lgamma(prior0 + eps)

    # Digamma terms
    digamma_alpha = torch.digamma(alpha + eps)
    digamma_alpha0 = torch.digamma(alpha0 + eps)

    t = (alpha - prior) * (digamma_alpha - digamma_alpha0)
    kl = log_B_alpha - log_B_prior + torch.sum(t, dim=1)

    return torch.mean(kl)

def loglikelihood_loss(masked_label, alpha):                                                #Sensoy et al EDL Paper page 5 2nd equality (above Propositions)
    
    S = torch.sum(alpha, dim=1, keepdim=True)                                               # B x 1 x H x W

    Var_numer= alpha * torch.sum((S - alpha), dim=1, keepdim=True)                          # B x C x H x W
    Var_denom = S * S * torch.sum((S + 1), dim=1, keepdim=True)                             # B x 1 x H x W
    
    loglikelihood_err = torch.sum((masked_label - (alpha / S)) ** 2, dim=1, keepdim=True)   # B x 1 X H x W
    loglikelihood_var = Var_numer / Var_denom                                               # B x 1 X H x W
    loglikelihood = (loglikelihood_err + loglikelihood_var).sum(dim=(-1, -2))               # B x C

    return loglikelihood


def wasserstein_dirac_metric(label_one_hot, alpha, reduction='mean'):
       # Dirichlet mean m
    S = alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)  # (B,1,H,W)
    m = alpha / S                                        # (B,C,H,W)

    # Probability assigned to the true class per pixel
    m_true = (m * label_one_hot).sum(dim=1, keepdim=True)  # (B,1,H,W)

    # W1 with Dirac metric: 1 - m_true (per pixel). Sum over H like your other losses.
    loss = (1.0 - m_true).sum(dim=-2)  # (B,1,W)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def wasserstein_dirac_squared_metric(label_one_hot, alpha, reduction='mean'):
       # Dirichlet mean m
    S = alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)  # (B,1,H,W)
    m = alpha / S                                        # (B,C,H,W)

    # Probability assigned to the true class per pixel
    m_true = (m * label_one_hot).sum(dim=1, keepdim=True)  # (B,1,H,W)

    # W1 with Dirac metric: 1 - m_true (per pixel).
    loss = ((1.0 - m_true)**2).sum(dim=-2)  # (B,1,W)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def mse_loss(label_one_hot, strength, p):        #Sensoy et al EDL Paper page 5 3rd equality (above Propositions)
    """
    y : Labels: B X C => B x C x W x H
    alpha : Dirichlet: B X C => B x C x W x H
    """
    
    err = (label_one_hot - p) ** 2                  #B x C x H x W
    var = p * (1 - p) / (strength + 1)              #B x C x H x W
    loss = (err + var).sum(dim=-2)                  #B x C x H(1024)
    loss = loss.mean() 

    return loss
    

def maximum_likelihoodII(label_one_hot, strength, alpha): # Sensoy et al loss fct 1 p.4 equaton (3)
    "y: label tensor (one-hot)"
    "Strength: tensor of sum of all evidences alpha"
    "alpha: evidence tensor +1 "
    "log(strenght) : tensor B x 1 x 512 x 1024 (Bx1xWxH)"
    "log(alpha): tensor Bx C x W x H ( B x 19 x 512 x 1024)"
    "loss : tensor Bx C x H ( B x 19 x 1024)"

    loss = (label_one_hot * (torch.log(strength) - torch.log(alpha))).sum(dim=-2)
    
    return loss

def kl_divergence(alpha, reference):
    "as name suggests, computes KL divergence between the reference distribution (uniform) and the sample distribution and takes the average over the the regarded patch"
    "evidence: tensor of the predicted probabilities B x C x H X W"
    "reference: tensor representing a uniform distribution of classes 1 x C x H x W"


    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(reference).sum(dim=1, keepdim=True)
        - torch.lgamma(reference.sum(dim=1, keepdim=True))
    )
    second_term = (alpha - reference).mul(torch.digamma(alpha) - torch.digamma(sum_alpha)).sum(dim=1, keepdim=True)
    kl = first_term + second_term
    return kl


def kl_dirichlet_regularization(alpha, num_classes, reduction='mean'):
    """
    KL divergence between predicted Dirichlet and uniform Dirichlet prior.
    Encourages the model to be uncertain when evidence is weak.
    """
    prior = torch.ones_like(alpha)
    prior = prior * 0.25
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)

    # Dirichlet KL divergence (per pixel)
    log_term = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    log_term += torch.sum(torch.lgamma(prior), dim=1, keepdim=True) - torch.lgamma(torch.sum(prior, dim=1, keepdim=True))
    kl = log_term + torch.sum((alpha - prior) * (torch.digamma(alpha) - torch.digamma(S_alpha)), dim=1, keepdim=True)

    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    return kl
    

def bayes_risk(label_one_hot, strength, alpha):         # Sensoy et al loss fct 2 p.5 equaton (4)
    "y: label tensor (one-hot)"
    "Strength: tensor of sum of all evidences alpha"
    "alpha: evidence tensor +1 "
    "digamma-strength : tensor B x 1 x 512 x 1024 (Bx1xWxH), digamma function applied to strength"
    "digamma-alpha: tensor Bx C x W x H ( B x 19 x 512 x 1024), digamma function applied to alpha"
    "loss : tensor Bx C x H ( B x 19 x 1024)"

    loss = (label_one_hot * (torch.digamma(strength) - torch.digamma(alpha))).sum(dim=-2)
   
    return loss

def wasserstein_metric(label_one_hot, alpha, reduction='mean'):
    strenght = torch.sum(alpha, dim=1, keepdim=True)  # (B, 1, H, W)
    pred_mean = alpha / strenght  # (B, C, H, W)


    loss = ((pred_mean - label_one_hot)**2).sum(dim=-2) # (B, C, 1024)
   

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss





@MODELS.register_module()
class EDLLoss(nn.Module):
    # (5) in the paper
    def __init__(
        self, num_classes=19, loss_name='loss_edl', use_sigmoid=False, loss_weight=1.0, annealing_step: int = 10, class_weights=None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.class_weights = class_weights
        self._loss_name = loss_name
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
    
    num_classes=19

    def forward(self, pred: torch.Tensor, label: torch.Tensor, kl_weight=1.0, weight=None, ignore_index=-100) -> torch.Tensor:
        """_summary_

        Args:
            output (torch.Tensor): Network prediction in shape B x C X H x W
            label (torch.Tensor): One-hot encoded Groundtruth in shape B x H x W
            label_one_hot (torch.Tensor): One-hot encoded Groundtruth in shape B x C X H x W
            epoch_number (int): Current epoch number

        Returns:
            torch.Tensor: The loss value
        """
        num_classes=19
        step = MessageHub.get_instance('message_hub').get_info('iter', 0)

    

        label_one_hot = safe_one_hot(label, num_classes=self.num_classes)  # (B, H, W, C)
        label_one_hot = label_one_hot.permute(0, 3, 1, 2).float()

        mask = (label_one_hot != 0).any(dim=1)
        mask = mask.unsqueeze(1)
        mask = mask.expand_as(label_one_hot)
        masked_label = label_one_hot * mask
        R = torch.nn.ReLU()
        evidence = R(pred)
        alpha = evidence + 1
        reference = torch.ones([1, num_classes, alpha.shape[2], alpha.shape[3]], dtype=torch.float32)
        reference = reference.to(device)
        strength = torch.sum(alpha, dim=1, keepdim=True)
        p = alpha / strength
        
        loss = wasserstein_dirac_metric(label_one_hot, alpha, reduction='mean') + 0.45 * mse_loss(masked_label, strength, p)
        
        # KL divergence to uniform prior
        kl_reg = kl_dirichlet_regularization(alpha, self.num_classes, reduction='mean')
    
        # Annealing coefficient
        kl_weight = get_annealing_coef(step, self.annealing_step, max_coef=1.0)
        
        total_loss = loss + kl_weight * kl_reg
        return total_loss
    
    @property
    def loss_name(self):
        return self._loss_name