from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Part 1: CAVI and Gaussian estimation

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
            self.mu_n = (self.ndata*self.xbar + self.mu_0)/(self.ndata+1)
            self.mu = self.mu_n
            
            self.alpha_n = self.alpha_0 + (self.ndata+1)/2
            self.beta_n = self.beta_0 + 0.5*(np.sum(self.data**2)) - self.xbar*np.sum(self.data) + ((self.ndata+1)/2)*((self.sigma/self.ndata) + self.xbar**2) - self.mu_0*self.xbar + 0.5*self.mu_0**2           
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

# Part 2: Custom K-Means clustering

# Generate three clusters of two-dimensional data points
np.random.seed()
cluster1 = np.random.normal(loc=[0, 0], scale=0.5, size=(100, 2))
cluster2 = np.random.normal(loc=[3, 3], scale=0.5, size=(100, 2))
cluster3 = np.random.normal(loc=[0, 3], scale=0.5, size=(100, 2))
data_clusters = np.vstack((cluster1, cluster2, cluster3))

def custom_kmeans(data, k, max_iter=100, tol=1e-4):
    n_samples, n_features = data.shape
    # Initialize centroids randomly from the data points
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = data[indices]
    
    for i in range(max_iter):
        # Compute distances and assign clusters
        distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            if np.any(labels == j):
                new_centroids[j] = data[labels == j].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]  # Avoid empty cluster
        
        # Check for convergence
        if np.allclose(new_centroids, centroids, atol=tol):
            break
        centroids = new_centroids
        
    return labels, centroids

labels, centroids = custom_kmeans(data_clusters, k=3)

# Plot the clustered data with centroids highlighted
plt.figure()
plt.scatter(data_clusters[:,0], data_clusters[:,1], c=labels, cmap='viridis', s=30)
plt.scatter(centroids[:,0], centroids[:,1], color='red', marker='x', s=100, label='Centroids')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
# Use the custom K-means labels to form clusters and then run CAVI on each dimension separately.
# (Note: The original CAVI class estimates parameters for a one-dimensional Gaussian.
#  Here we apply it independently to each feature of each cluster.)

cluster_params = {}
unique_labels = np.unique(labels)

for lbl in unique_labels:
    cluster_data = data_clusters[labels == lbl]
    mu_est = []
    sigma_est = []
    # Run CAVI independently on each dimension
    for d in range(cluster_data.shape[1]):
        cavi_instance = CAVI(cluster_data[:, d])
        cavi_instance.coordinate_ascent(100)
        mu_est.append(cavi_instance.mu)
        sigma_est.append(cavi_instance.sigma)
    cluster_params[lbl] = (mu_est, sigma_est)
    print(f"Cluster {lbl}: Estimated mean = {mu_est}, sigma = {sigma_est}")

# Plot clusters with the estimated means from CAVI for a visual check
plt.figure()
colors = ['tab:blue', 'tab:orange', 'tab:green']
for lbl in unique_labels:
    pts = data_clusters[labels == lbl]
    plt.scatter(pts[:, 0], pts[:, 1], c=colors[lbl], label=f"Cluster {lbl}", alpha=0.6)
    mu_vec, _ = cluster_params[lbl]
    plt.scatter(mu_vec[0], mu_vec[1], c='red', marker='x', s=150, linewidths=3)

plt.title("2D Clustering with CAVI Estimated Means")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
