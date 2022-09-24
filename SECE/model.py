
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from .module import model_dict, GAT_pyg
from .utils_loss import nll_loss, plot_loss2
from .utils_dataset import MyDataset, Batch_Data
from .utils import MakeLogClass

import matplotlib.pyplot as plt

class SECE_model(object):
    
    """
    Spatial region-related embedding and Cell type-related embedding of spatial transcriptomics.

    Parameters
    ----------
    adata
        AnnData object object of scanpy package.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    dropout_rate
        Dropout rate for neural networks.
    gene_likelihood
        One of:
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    dropout_rate
        Dropout rate of model1
    dropout_gat_rate
        Dropout rate of model2
    num_workers
        Number of workers in dataloader
    result_path
        Output path of model
    make_log
        Whether to create a log file or not


    Examples
    --------
    >>> adata = sc.read_h5ad(path_to_anndata)
    >>> sece = SECE_model(adata.copy())
    >>> sece.prepare_data()
    >>> sece.train_model1()
    >>> adata1 = sece.predict_model1()
    >>> sece.prepare_graph()
    >>> sece.train_model2()
    >>> adata1 = sece.predict_model2()
    """

    def __init__(
        self,
        adata,
        hvg=False,
        n_hidden: int = 128,
        n_latent: int = 32,
        dropout_rate: float = 0.2,
        dropout_gat_rate: float = 0.5,
        likelihood: str = "nb",
        num_workers: int = 4,
        device = None,
        result_path = None, 
        make_log = True
    ):
        super(SECE_model, self).__init__() 
        
        self.adata = adata


        if device is not None:
            self.device = device
        else:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        if hvg:
            self.n_input = hvg
        else:
            self.n_input = adata.shape[1]
        
        self.hvg = hvg
        self.n_latent = n_latent
        self.likelihood = likelihood
        
        self.model1 = model_dict[self.likelihood](
            input_dim = self.n_input, 
            hidden_dim = n_hidden, 
            latent_dim = n_latent, 
            dropout = dropout_rate
        ).to(self.device)
        
        self.model2 = GAT_pyg(
            latent_dim=self.n_latent, 
            dropout_gat=dropout_gat_rate
            ).to(self.device)
        
        
        self.num_workers = num_workers
        self.result_path = result_path
        self.make_log = make_log
        if self.make_log:
            self.makeLog = MakeLogClass(f"{self.result_path}/log.tsv").make
            self.makeLog(f"Likelihood: {self.likelihood}")
            self.makeLog(f"Input dim: {self.n_input}")
            self.makeLog(f"Latent Dir: {self.n_latent}")
            self.makeLog(f"Model1 dropout: {dropout_rate}")
            self.makeLog(f"Model2 dropout: {dropout_gat_rate}")
    

    def prepare_data(self, lib_size='explog', normalize=True, scale=False):
        
        self.dataset1 = MyDataset(self.adata, size=lib_size, normalize=normalize, hvg=self.hvg, scale=scale)
        if self.hvg:
            self.adata = self.adata[:,:self.hvg]
        if self.make_log:
            self.makeLog(f"Library size: {lib_size}")
            self.makeLog(f"Input normalize: {normalize}")
            self.makeLog(f"Input scale: {scale}")
            self.makeLog(f"Hvg: {self.hvg}")
        print(self.adata.shape)

    
    def train_model1(self, 
                     lr=0.001, 
                     weight_decay=0, 
                     epoch1=40, 
                     batch_size=128,
                     plot=False):
        
        data_loader = DataLoader(self.dataset1, shuffle=True, batch_size=batch_size, num_workers=self.num_workers)
        optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=lr, weight_decay=weight_decay)   
        
        train_loss = []
        for epoch in tqdm(range(0, epoch1)):
            loss_tmp = 0
            for i, (feat_tmp, count_tmp, size_tmp) in enumerate(data_loader):
                feat_tmp = feat_tmp.to(self.device)
                count_tmp = count_tmp.to(self.device)
                size_tmp = size_tmp.to(self.device)
                self.model1.train()
                rate_scaled_tmp, logits_tmp, drop_tmp, z_tmp = self.model1(feat_tmp)
                rate_tmp = rate_scaled_tmp*size_tmp
                mean_tmp = rate_tmp*logits_tmp
                optimizer1.zero_grad()
                loss_train = nll_loss(count_tmp, mean_tmp, rate_tmp, drop_tmp, dist=self.likelihood).mean()
                loss_train.backward()
                optimizer1.step()
                # if i%5 == 0:
                #     print("Epoch:{},  Current loss {}".format(epoch,loss_train.item()))
                loss_tmp += loss_train.item()
            train_loss.append(loss_tmp/len(data_loader))
        
        self.model1.eval()
        
        if self.result_path:
            torch.save(self.model1.state_dict(),f'{self.result_path}/model1.pth')
            if plot:
                plt.plot(train_loss)
                plt.savefig(f'{self.result_path}/model1_loss.png')
                plt.close()
        
        if self.make_log:
            self.makeLog(f"Model1 lr: {lr}")
            self.makeLog(f"Model1 epoch: {epoch1}")
            self.makeLog(f"Model1 batch_size: {batch_size}")


    def predict_model1(self, batch_size=512):
        
        self.model1.eval()
        dataloader = DataLoader(self.dataset1, shuffle=False, batch_size=batch_size, num_workers=self.num_workers)
        
        mean = torch.empty(size=[0,self.n_input])
        z = torch.empty(size=[0,self.n_latent])
        
        with torch.no_grad():
            for i, (feat_tmp, count_tmp, lib_tmp) in enumerate(dataloader):
                # print(i)
                feat_tmp = feat_tmp.to(self.device)
                count_tmp = count_tmp.to(self.device)
                lib_tmp = lib_tmp.to(self.device)
                rate_scaled_tmp, logits_tmp, dropout_tmp, z_tmp = self.model1(feat_tmp)
                rate_tmp = rate_scaled_tmp*lib_tmp
                mean_tmp = rate_tmp*logits_tmp
                mean = torch.cat([mean, mean_tmp.cpu()])
                z = torch.cat([z, z_tmp.cpu()])
        
        self.adata.layers['expr'] = mean.detach().cpu().numpy()
        self.adata.obsm['X_CE'] = z.detach().cpu().numpy()
        
        return self.adata
    
    def prepare_graph(self, 
                      cord_keys=['x','y'], 
                      latent_key = 'X_CE',
                      num_batch_x=1, 
                      num_batch_y=1, 
                      neigh_cal='knn',
                      n_neigh=10, 
                      n_radius=100,
                      kernal_thresh=0.5):

        self.batch_list, self.data_all = Batch_Data(self.adata,
                                                    cord_keys=cord_keys, 
                                                    latent_key = latent_key,
                                                    num_batch_x = num_batch_x, 
                                                    num_batch_y = num_batch_y, 
                                                    neigh_cal = neigh_cal, 
                                                    n_neigh = n_neigh, 
                                                    n_radius = n_radius,
                                                    kernal_thresh = kernal_thresh)
        if self.make_log:
            self.makeLog(f"Graph cal: {neigh_cal}")
            if neigh_cal == 'knn':
                self.makeLog(f"knn: {n_neigh}")
            else:
                self.makeLog(f"radius: {n_radius}")
            self.makeLog(f"kernal_thresh: {kernal_thresh}")


    def train_model2(self, 
                     lr_gat=0.01, 
                     weight_decay_gat=0, 
                     epoch2=40, 
                     re_weight=1, 
                     si_weight=0.08, 
                     plot=False):
        
        optimizer2 = torch.optim.Adam(self.model2.parameters(), lr=lr_gat, weight_decay=weight_decay_gat)
        
        loss_list = []
        for epoch in tqdm(range(epoch2)):
            for batch_this in self.batch_list:
                
                data, kernal = batch_this
                self.model2.train()
                
                data = data.to(self.device)
                kernal = kernal.to(self.device)
                zbar, w = self.model2(data)
                
                re_loss = F.mse_loss(data.x, zbar)
                si_loss = F.mse_loss(kernal, torch.mm(zbar,zbar.T))
                
                loss = re_weight * re_loss + si_weight * si_loss
                loss_list.append([loss.item(), re_loss.item(), si_loss.item()])
                # print("Epoch:{}  loss:{}  re_loss:{}  si_loss:{}".format(epoch, loss.item(), re_loss.item(), si_loss.item()))
               
                loss.backward()
                optimizer2.step()
                optimizer2.zero_grad()
        
        self.model2.eval()
       
        if self.result_path:
            torch.save(self.model2.state_dict(),f'{self.result_path}/model2.pth')
            if plot:
                plot_loss2(loss_list, self.result_path, labels = ['loss', 'loss_re', 'loss_kernal'])
    
        if self.make_log:
            self.makeLog(f"Model2 lr: {lr_gat}")
            self.makeLog(f"Model2 epoch: {epoch2}")
            self.makeLog(f"Model2 similar weight: {si_weight}")


    def predict_model2(self):
       
        self.model2.eval()
        w, _ = self.model2(self.data_all.to(self.device))
        
        self.adata.obsm['X_SE'] = w.detach().cpu().numpy()

        return self.adata



