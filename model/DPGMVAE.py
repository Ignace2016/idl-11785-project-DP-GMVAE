"""
---------------------------------------------------------------------
-- Author: Team 60: Haonan Di, Yuelin Liu, Yu Liu
---------------------------------------------------------------------

Dirichlet Process - Gaussian Mixture Variational Autoencoder Networks
GMVAE with DPGMM Priors for Unsupervised Clustering
"""

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from networks.Networks import *
from losses.LossFunctions import *
from metrics.Metrics import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.mixture import BayesianGaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, cosine_similarity

import torch.optim as optim


class DPGMVAE:

  def __init__(self, args):
    self.num_epochs = args.epochs
    self.cuda = args.cuda
    self.verbose = args.verbose
    self.visualize = args.visualize

    self.batch_size = args.batch_size
    self.batch_size_val = args.batch_size_val
    self.learning_rate = args.learning_rate
    self.decay_epoch = args.decay_epoch
    self.lr_decay = args.lr_decay
    self.w_cat = args.w_categ
    self.w_gauss = args.w_gauss    
    self.w_rec = args.w_rec
    self.rec_type = args.rec_type 

    self.num_classes = args.num_classes
    self.gaussian_size = args.gaussian_size
    self.input_size = args.input_size

    # gumbel
    self.init_temp = args.init_temp
    self.decay_temp = args.decay_temp
    self.hard_gumbel = args.hard_gumbel
    self.min_temp = args.min_temp
    self.decay_temp_rate = args.decay_temp_rate
    self.gumbel_temp = self.init_temp

    self.network = GMVAENet(self.input_size, self.gaussian_size, self.num_classes)
    self.losses = LossFunctions()
    self.metrics = Metrics()
    
    
    
    if self.cuda:
      self.network = self.network.cuda() 
  

  def unlabeled_loss(self, data, out_net):
    """Method defining the loss functions derived from the variational lower bound
    Args:
        data: (array) corresponding array containing the input data
        out_net: (dict) contains the graph operations or nodes of the network output

    Returns:
        loss_dic: (dict) contains the values of each loss function and predictions
    """
    # obtain network variables
    z, data_recon = out_net['gaussian'], out_net['x_rec'] 
    logits, prob_cat = out_net['logits'], out_net['prob_cat']
    y_mu, y_var = out_net['y_mean'], out_net['y_var']
    mu, var = out_net['mean'], out_net['var']
    
    # reconstruction loss
    loss_rec = self.losses.reconstruction_loss(data, data_recon, self.rec_type)

    # gaussian loss
    loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)

    # categorical loss
    loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)

    # total loss
    loss_total = self.w_rec * loss_rec + self.w_gauss * loss_gauss + self.w_cat * loss_cat

    # obtain predictions
    _, predicted_labels = torch.max(logits, dim=1)

    loss_dic = {'total': loss_total, 
                'predicted_labels': predicted_labels,
                'reconstruction': loss_rec,
                'gaussian': loss_gauss,
                'categorical': loss_cat}
    return loss_dic
    
    
  def train_epoch(self, optimizer, data_loader):
    """Train the model for one epoch

    Args:
        optimizer: (Optim) optimizer to use in backpropagation
        data_loader: (DataLoader) corresponding loader containing the training data

    Returns:
        average of all loss values, accuracy, nmi
    """
    self.network.train()
    total_loss = 0.
    recon_loss = 0.
    cat_loss = 0.
    gauss_loss = 0.

    accuracy = 0.
    nmi = 0.
    num_batches = 0.
    
    true_labels_list = []
    predicted_labels_list = []

    # iterate over the dataset
    for (data, labels) in tqdm(data_loader):
      if self.cuda == 1:
        data = data.cuda()

      optimizer.zero_grad()

      # flatten data
      data = data.view(data.size(0), -1)
      
      # forward call
      out_net = self.network(data, self.gumbel_temp, self.hard_gumbel) 
      unlab_loss_dic = self.unlabeled_loss(data, out_net) 
      total = unlab_loss_dic['total']

      # accumulate values
      total_loss += total.item()
      recon_loss += unlab_loss_dic['reconstruction'].item()
      gauss_loss += unlab_loss_dic['gaussian'].item()
      cat_loss += unlab_loss_dic['categorical'].item()

      # perform backpropagation
      total.backward()
      optimizer.step()  

      # save predicted and true labels
      predicted = unlab_loss_dic['predicted_labels']
      true_labels_list.append(labels)
      predicted_labels_list.append(predicted)   
   
      num_batches += 1. 

    # average per batch
    total_loss /= num_batches
    recon_loss /= num_batches
    gauss_loss /= num_batches
    cat_loss /= num_batches
    
    # concat all true and predicted labels
    true_labels = torch.cat(true_labels_list, dim=0).cpu().numpy()
    predicted_labels = torch.cat(predicted_labels_list, dim=0).cpu().numpy()

    # compute metrics
    accuracy = 100.0 * self.metrics.cluster_acc(predicted_labels, true_labels)
    nmi = 100.0 * self.metrics.nmi(predicted_labels, true_labels)

    return total_loss, recon_loss, gauss_loss, cat_loss, accuracy, nmi


  def test(self, data_loader, return_loss=False):
    """Test the model with new data

    Args:
        data_loader: (DataLoader) corresponding loader containing the test/validation data
        return_loss: (boolean) whether to return the average loss values
          
    Return:
        accuracy and nmi for the given test data

    """
    self.network.eval()
    total_loss = 0.
    recon_loss = 0.
    cat_loss = 0.
    gauss_loss = 0.

    accuracy = 0.
    nmi = 0.
    num_batches = 0.
    
    true_labels_list = []
    predicted_labels_list = []

    with torch.no_grad():
      for data, labels in data_loader:
        if self.cuda == 1:
          data = data.cuda()
      
        # flatten data
        data = data.view(data.size(0), -1)

        # forward call
        out_net = self.network(data, self.gumbel_temp, self.hard_gumbel) 
        unlab_loss_dic = self.unlabeled_loss(data, out_net)  

        # accumulate values
        total_loss += unlab_loss_dic['total'].item()
        recon_loss += unlab_loss_dic['reconstruction'].item()
        gauss_loss += unlab_loss_dic['gaussian'].item()
        cat_loss += unlab_loss_dic['categorical'].item()

        # save predicted and true labels
        predicted = unlab_loss_dic['predicted_labels']
        true_labels_list.append(labels)
        predicted_labels_list.append(predicted)   
   
        num_batches += 1. 

    # average per batch
    if return_loss:
      total_loss /= num_batches
      recon_loss /= num_batches
      gauss_loss /= num_batches
      cat_loss /= num_batches
    
    # concat all true and predicted labels
    true_labels = torch.cat(true_labels_list, dim=0).cpu().numpy()
    predicted_labels = torch.cat(predicted_labels_list, dim=0).cpu().numpy()

    # compute metrics
    accuracy = 100.0 * self.metrics.cluster_acc(predicted_labels, true_labels)
    nmi = 100.0 * self.metrics.nmi(predicted_labels, true_labels)

    if return_loss:
      return total_loss, recon_loss, gauss_loss, cat_loss, accuracy, nmi
    else:
      return accuracy, nmi


  def train(self, train_loader, val_loader):
    """Train the model

    Args:
        train_loader: (DataLoader) corresponding loader containing the training data
        val_loader: (DataLoader) corresponding loader containing the validation data

    Returns:
        output: (dict) contains the history of train/val loss
    """
    best_acc = 0
    optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate) # , weight_decay=1e-4
    train_history_acc, val_history_acc = [], []
    train_history_nmi, val_history_nmi = [], []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.4, patience=90,cooldown=2,verbose=True)
    for epoch in range(1, self.num_epochs + 1):
      train_loss, train_rec, train_gauss, train_cat, train_acc, train_nmi = self.train_epoch(optimizer, train_loader)
      val_loss, val_rec, val_gauss, val_cat, val_acc, val_nmi = self.test(val_loader, True)

      # if verbose then print specific information about training
      if self.verbose == 1:
        print("(Epoch %d / %d)" % (epoch, self.num_epochs) )
        print("Train - REC: %.5lf;  Gauss: %.5lf;  Cat: %.5lf;" % \
              (train_rec, train_gauss, train_cat))
        print("Valid - REC: %.5lf;  Gauss: %.5lf;  Cat: %.5lf;" % \
              (val_rec, val_gauss, val_cat))
        print("Accuracy=Train: %.5lf; Val: %.5lf   NMI=Train: %.5lf; Val: %.5lf   Total Loss=Train: %.5lf; Val: %.5lf" % \
              (train_acc, val_acc, train_nmi, val_nmi, train_loss, val_loss))
      else:
        print('(Epoch %d / %d) Train_Loss: %.3lf; Val_Loss: %.3lf   Train_ACC: %.3lf; Val_ACC: %.3lf   Train_NMI: %.3lf; Val_NMI: %.3lf' % \
              (epoch, self.num_epochs, train_loss, val_loss, train_acc, val_acc, train_nmi, val_nmi))
      
      # visualize latent space
      if self.visualize == 1 and epoch in (1,3,5,10,20,50):
        test_features, test_labels = self.latent_features(train_loader, "mean", True)
        tsne_features = TSNE(n_components=2).fit_transform(test_features[:10000])
        fig = plt.figure(figsize=(10, 6))

        plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=test_labels[:10000], marker='o',
                    edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)
        plt.grid(False)
        plt.axis('off')
        plt.colorbar()
        plt.show()  

      # decay gumbel temperature
      if self.decay_temp == 1:
        self.gumbel_temp = np.maximum(self.init_temp * np.exp(-self.decay_temp_rate * epoch), self.min_temp)
        if self.verbose == 1:
          print("Gumbel Temperature: %.3lf" % self.gumbel_temp)

      train_history_acc.append(train_acc)
      val_history_acc.append(val_acc)
      train_history_nmi.append(train_nmi)
      val_history_nmi.append(val_nmi)
      scheduler.step(val_loss)
      
      if (val_acc > best_acc):
          torch.save(self.network,"/content/drive/MyDrive/ColabNotebooks/DeepLearning/project/fashionMnist-1205.pth")
          best_acc = val_acc
          print("best acc is %.5lf" % best_acc)
          print("model saved")
      
      if epoch > 30 and epoch % 3 == 2
    #   # use dpgmm prior to relace model prior
        print("Calculating similarity by DPGMM")
        test_features, test_labels = self.latent_features(train_loader, "mean", True)
        dpgmm = BayesianGaussianMixture(n_components = self.num_classes, weight_concentration_prior_type='dirichlet_process', 
                    covariance_type='diag', max_iter = 1000, random_state=0)  # , weight_concentration_prior = 100
        dpgmm.fit(test_features)
        labels_dpgmm = dpgmm.predict(test_features)
        label_dict_dpgmm = dict()
        for l in np.unique(labels_dpgmm):
            counts = np.bincount(test_labels[labels_dpgmm == l])
            label_dict_dpgmm.update({l: np.argmax(counts)})
        labels_transform = [label_dict_dpgmm[l] for l in labels_dpgmm]
        # #   temp_mu = self.network.generative.y_mu.weight
        # #   temp_var = self.network.generative.y_var.weight
        cos_simi = cosine_similarity(self.network.generative.y_mu.weight.detach().cpu().numpy().T, dpgmm.means_)
        
        temp = cos_simi.argmax(axis=1)
        replaced_mu = []
        replaced_var = []
        countChanged = 0
        maxSim = []
        print("number of different mean: " + str(len(set(temp))))
        if (len(set(temp)) == 10):
            print("Using prior of dpgmm")
            for i in range(10):
                maxSim.append(cos_simi[i, temp[i]])
                if (cos_simi[i, temp[i]] > 0.90):
                    countChanged += 1
                replaced_mu += [dpgmm.means_[temp[i]] if cos_simi[i, temp[i]] > 0.90 else self.network.generative.y_mu.weight.detach().cpu().numpy().T[i]]
                replaced_var += [dpgmm.covariances_[temp[i]] if cos_simi[i, temp[i]] > 0.90 else self.network.generative.y_var.weight.detach().cpu().numpy().T[i]]
            #   # 防止model的Mu一对多
            #   # 使用model的mu和var代替dpgmm的if cos_sim < 0.92
            #   replaced_mu = dpgmm.means_[cos_simi.argmax(axis=1)].T
            #   replaced_var = dpgmm.covariances_[cos_simi.argmax(axis=1)].T
            self.network.generative.y_mu.weight = torch.nn.parameter.Parameter(torch.Tensor(np.array(replaced_mu)).transpose(0,1).cuda()) # 
            self.network.generative.y_var.weight = torch.nn.parameter.Parameter(torch.Tensor(np.array(replaced_var)).transpose(0,1).cuda()) #
            
            if(countChanged!=0):
                print(maxSim)
                print(str(countChanged) + " have been changed")
            
    #   countGreater_0_9 = 0
    #   for i in range(10):
    #       if(cos_simi[i][i]) < 0.95:
    #         self.network.generative.y_mu.weight[i] = temp_mu[i]
    #         self.network.generative.y_var.weight[i] = temp_var[i]
    #   print(cosine_similarity(self.network.generative.y_mu.weight.detach().cpu().numpy().T, temp_mu))  
    return {'train_history_nmi' : train_history_nmi, 'val_history_nmi': val_history_nmi,
            'train_history_acc': train_history_acc, 'val_history_acc': val_history_acc}
  

  def latent_features(self, data_loader, which_fea, return_labels=False):
    """Obtain latent features learnt by the model

    Args:
        data_loader: (DataLoader) loader containing the data
        return_labels: (boolean) whether to return true labels or not

    Returns:
       features: (array) array containing the features from the data
    """
    self.network.eval()
    N = len(data_loader.dataset)
    features = np.zeros((N, self.gaussian_size))
    if return_labels:
      true_labels = np.zeros(N, dtype=np.int64)
    start_ind = 0
    with torch.no_grad():
      for (data, labels) in data_loader:
        if self.cuda == 1:
          data = data.cuda()
        # flatten data
        data = data.view(data.size(0), -1)  
        out = self.network.inference(data, self.gumbel_temp, self.hard_gumbel)
        # latent_feat = out['mean']
        # latent_feat = out['gaussian']
        latent_feat = out[which_fea]
        end_ind = min(start_ind + data.size(0), N+1)
        # print("mean shape: ")
        # print(out['mean'].shape)
        # print("gaussian shape: ")
        # print(out['gaussian'].shape)
        # return true labels
        if return_labels:
          true_labels[start_ind:end_ind] = labels.cpu().numpy()
        features[start_ind:end_ind] = latent_feat.cpu().detach().numpy()  
        start_ind += data.size(0)
    if return_labels:
      return features, true_labels
    return features


  def reconstruct_data(self, data_loader, sample_size=-1):
    """Reconstruct Data

    Args:
        data_loader: (DataLoader) loader containing the data
        sample_size: (int) size of random data to consider from data_loader
      
    Returns:
        reconstructed: (array) array containing the reconstructed data
    """
    self.network.eval()

    # sample random data from loader
    indices = np.random.randint(0, len(data_loader.dataset), size=sample_size)
    test_random_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=sample_size, sampler=SubsetRandomSampler(indices))
  
    # obtain values
    it = iter(test_random_loader)
    test_batch_data, _ = it.next()
    original = test_batch_data.data.numpy()
    if self.cuda:
      test_batch_data = test_batch_data.cuda()  

    # obtain reconstructed data  
    out = self.network(test_batch_data, self.gumbel_temp, self.hard_gumbel) 
    reconstructed = out['x_rec']
    return original, reconstructed.data.cpu().numpy()


  def plot_latent_space(self, data_loader, save=False):
    """Plot the latent space learnt by the model

    Args:
        data: (array) corresponding array containing the data
        labels: (array) corresponding array containing the labels
        save: (bool) whether to save the latent space plot

    Returns:
        fig: (figure) plot of the latent space
    """
    # obtain the latent features
    features = self.latent_features(data_loader)
    
    # plot only the first 2 dimensions
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(features[:, 0], features[:, 1], c=labels, marker='o',
            edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)
    plt.colorbar()
    if(save):
        fig.savefig('latent_space.png')
    return fig
  
  
  def random_generation(self, num_elements=1):
    """Random generation for each category

    Args:
        num_elements: (int) number of elements to generate

    Returns:
        generated data according to num_elements
    """ 
    # categories for each element
    arr = np.array([])
    for i in range(self.num_classes):
      arr = np.hstack([arr,np.ones(num_elements) * i] )
    indices = arr.astype(int).tolist()

    categorical = F.one_hot(torch.tensor(indices), self.num_classes).float()
    
    if self.cuda:
      categorical = categorical.cuda()
  
    # infer the gaussian distribution according to the category
    mean, var = self.network.generative.pzy(categorical)

    # gaussian random sample by using the mean and variance
    noise = torch.randn_like(var)
    std = torch.sqrt(var)
    gaussian = mean + noise * std

    # generate new samples with the given gaussian
    generated = self.network.generative.pxz(gaussian)

    return generated.cpu().detach().numpy() 

