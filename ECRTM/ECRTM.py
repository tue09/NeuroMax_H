import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ECR import ECR
from NeuroMax.CTR import CTR


class ECRTM(nn.Module):
    '''
        Effective Neural Topic Modeling with Embedding Clustering Regularization. ICML 2023

        Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, Anh Tuan Luu.
    '''
    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0., pretrained_WE=None, embed_size=200, is_CTR=False,
                    cluster_distribution=None, cluster_mean=None, cluster_label=None, sinkhorn_alpha = 20.0, weight_CTR=100.0, learn_=0,
                    beta_temp=0.2, weight_loss_ECR=250.0, alpha_ECR=20.0, sinkhorn_max_iter=1000, coef_=0.5, init_2=0, use_MOO=1):
        super().__init__()
        self.coef_ = coef_
        self.use_MOO = use_MOO
        self.learn_ = learn_
        self.lambda_1 = self.coef_
        self.lambda_2 = self.coef_
        self.lambda_3 = self.coef_

        self.num_topics = num_topics
        self.beta_temp = beta_temp
        self.weight_CTR = weight_CTR
        self.is_CTR = is_CTR

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)
        self.fc1_dropout = nn.Dropout(dropout)
        self.theta_dropout = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)
        self.decoder_bn.weight.requires_grad = False

        if pretrained_WE is not None:
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        if init_2 == 1:
            self.topic_embeddings = nn.Parameter(torch.randn((num_topics, self.word_embeddings.shape[1])))
        else:
            self.topic_embeddings = torch.empty((num_topics, self.word_embeddings.shape[1]))
            nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
            self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.encoder1 = nn.Sequential(
            nn.Linear(vocab_size, en_units),
            nn.Softplus(),
            nn.Linear(en_units, en_units),
            nn.Softplus(),
            nn.Dropout(dropout)
        )

        # Add CTR
        self.cluster_mean = nn.Parameter(torch.from_numpy(cluster_mean).float(), requires_grad=False)
        self.cluster_distribution = nn.Parameter(torch.from_numpy(cluster_distribution).float(), requires_grad=False)
        self.cluster_label = cluster_label
        if not isinstance(self.cluster_label, torch.Tensor):
            self.cluster_label = torch.tensor(self.cluster_label, dtype=torch.long, device='cuda')
        else:
            self.cluster_label = self.cluster_label.to(device='cuda', dtype=torch.long)
        
        self.map_t2c = nn.Linear(self.word_embeddings.shape[1], self.cluster_mean.shape[1], bias=False)
        self.CTR = CTR(weight_CTR, sinkhorn_alpha, sinkhorn_max_iter)
        # #

        self.ECR = ECR(weight_loss_ECR, alpha_ECR, sinkhorn_max_iter)

    # Same
    def get_beta(self):
        dist = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    # Same
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    # Same
    def encode(self, input):
        # e1 = F.softplus(self.fc11(input))
        # e1 = F.softplus(self.fc12(e1))
        # e1 = self.fc1_dropout(e1)
        e1 = self.encoder1(input)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)

        loss_KL = self.compute_loss_KL(mu, logvar)

        return theta, loss_KL


    # Same
    def get_theta(self, input):
        theta, loss_KL = self.encode(input)
        if self.training:
            return theta, loss_KL
        else:
            return theta

    # Same
    def compute_loss_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD

    # Same
    def get_loss_ECR(self):
        cost = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        loss_ECR = self.ECR(cost)
        return loss_ECR

    # Same
    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost


    # Thêm
    def get_loss_CTR(self, input, indices):
        bow = input[0]
        theta, _ = self.encode(bow)
        cd_batch = self.cluster_distribution[indices]  
        cost = self.pairwise_euclidean_distance(self.cluster_mean, self.map_t2c(self.topic_embeddings))  
        loss_CTR = self.weight_CTR * self.CTR(theta, cd_batch, cost)  
        return loss_CTR

    def forward(self, indices, input, epoch_id=None):
        # input = input['data']
        bow = input[0]
        theta, loss_KL = self.encode(bow)
        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
        recon_loss = -(bow * recon.log()).sum(axis=1).mean()

        loss_TM = recon_loss + loss_KL

        loss_ECR = self.get_loss_ECR()

        if self.weight_CTR != 0:
            loss_CTR = self.get_loss_CTR(input, indices)
        else:
            loss_CTR = 0.0

        #loss = loss_TM + loss_ECR + loss_CTR
        loss = loss_TM + loss_ECR

        if self.use_MOO == 1:
            if self.weight_CTR != 0:
                rst_dict = {
                    'loss_': loss,
                    'loss_x1': recon_loss + self.coef_ * (loss_TM + loss_ECR + loss_CTR),
                    'loss_x2': loss_KL + self.coef_ * (loss_TM + loss_ECR + loss_CTR),
                    'loss_x3': loss_ECR + self.coef_ * (loss_TM + loss_ECR + loss_CTR),
                    'loss_x4': loss_CTR + + self.coef_ * (loss_TM + loss_ECR + loss_CTR),
                    'recon_loss': recon_loss,
                    'lossKL': loss_KL,
                    'lossECR': loss_ECR,
                    'loss_CTR': loss_CTR
                }
            else:
                if self.learn_ == 0:
                    rst_dict = {
                        'loss_': loss,
                        'loss_x1': recon_loss + self.coef_ * loss,
                        'loss_x2': loss_KL + self.coef_ * loss,
                        'loss_x3': loss_ECR + self.coef_ * loss,
                        'lossrecon': recon_loss,
                        'lossKL': loss_KL,
                        'lossECR': loss_ECR,
                    }
                else:
                    rst_dict = {
                        'loss_': loss,
                        'loss_x1': recon_loss + self.lambda_1 * loss,
                        'loss_x2': loss_KL + self.lambda_2 * loss,
                        'loss_x3': loss_ECR + self.lambda_3 * loss,
                        'losssrecon': recon_loss,
                        'losssKL': loss_KL,
                        'losssECR': loss_ECR,
                    }
        else:
            rst_dict = {
                    'loss_': loss,
                    'lossrecon': recon_loss,
                    'lossKL': loss_KL,
                    'lossECR': loss_ECR,
                    #'lossCTR': loss_CTR
                }


        return rst_dict

