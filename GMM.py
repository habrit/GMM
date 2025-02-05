from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
mu, sigma = 0.5, 1
data = np.random.normal(mu, sigma, 10000)
plt.hist(data, 30, density=True)


class CAVI():
    def __init__(self, data):
        self.data = data
        self.ndata = len(data)
        self.xbar = np.sum(data)/len(data)
        self.ELBOs = [1, 2]

        # hyperpriors (these don't change)
        self.mu_0 = 0
        self.alpha_0 = 0
        self.beta_0 = 0
        self.sigma_0 = 2
        
        # parameters
        self.mu_n = self.mu_0
        self.alpha_n = self.alpha_0
        self.beta_n = self.beta_0
        self.sigma = self.sigma_0
        self.mu = float(np.random.normal(self.mu_0, self.sigma))
    
    def calc_elbo(self):
        x_mu_sq = np.sum((self.data-self.mu)**2)
        cov_term = -0.5 * (1/self.sigma)
        logp_x = cov_term * x_mu_sq
        logp_mu = cov_term * np.sum((self.data-self.mu_0)**2)
        logp_sigma = (-self.alpha_0 -1) * np.log(sigma) - (self.beta_0/self.sigma)
        
        logq_mu = cov_term*(1/(self.ndata+1)) * ((self.mu-self.mu_n)**2)
        logq_sigma = (self.alpha_n -1)*np.log(self.sigma) - 1/(self.sigma) *(self.beta_n)
        
        ELBO = logp_x + logp_mu + logp_sigma - logq_mu - logq_sigma
        return ELBO
    
    def has_converged(self):
        diff = abs(self.ELBOs[-1] - self.ELBOs[-2])
        return diff<0.01
        
    def coordinate_ascent(self, iterate):
        itr=0
        while itr<iterate and not self.has_converged():
            itr+=1
            self.mu_n = (self.ndata*self.xbar + self.mu_0)/(self.ndata+1) # this actually converges aft 1 iteration
            self.mu = self.mu_n
            
            self.alpha_n = self.alpha_0 + (self.ndata+1)/2
            self.beta_n = self.beta_0 + 0.5*(np.sum(self.data**2)) - self.xbar*np.sum(self.data) + ((self.ndata+1)/2)*( (self.sigma/self.ndata) + self.xbar**2) - self.mu_0*self.xbar + 0.5*self.mu_0**2           
            self.sigma = (self.beta_n-1)/self.alpha_n
            ELBO = self.calc_elbo()
            self.ELBOs.append(ELBO)
            print("iteration:", itr, "ELBO:", ELBO)



cavi = CAVI(data)
cavi.coordinate_ascent(1000)
x = np.linspace(min(data), max(data), 100)
y = stats.norm.pdf(x, loc=cavi.mu, scale=cavi.sigma)
plt.plot(x, y, 'r-', label='Estimated Gaussian')
plt.legend()
plt.show()


# Generate three separate clusters in 2D
np.random.seed(42)
cluster1 = np.random.normal(loc=[0, 0], scale=1.0, size=(100, 2))
cluster2 = np.random.normal(loc=[5, 5], scale=1.0, size=(100, 2))
cluster3 = np.random.normal(loc=[10, 0], scale=1.0, size=(100, 2))

# Combine all clusters into one dataset
data = np.vstack((cluster1, cluster2, cluster3))

# Cluster the data into three clusters using KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(data)

# Plot the clusters with different colors and plot the centroids
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, label='Centroids')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Three Clusters with KMeans")
plt.legend()
plt.show()
