import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, RationalQuadratic, ExpSineSquared, DotProduct
import matplotlib.pyplot as plt
import pdb

class RetrievalGPUCB:
    """
    GP-UCB implementation specifically for retrieval tasks.
    """
    
    def __init__(self, beta=2.0, kernel='rbf'):
        """
        Initialize the GP-UCB algorithm for retrieval
        
        Args:
            beta: Exploration parameter that balances exploitation vs exploration
            is_embeddings_based: Whether to use embeddings (True) or indices (False) as features
        """
        self.beta = beta
        self.is_fitted = False
        # Observations
        self.X = []  # Features of observed points (indices or embeddings)
        self.y = []  # Rewards (relevance scores) observed so far
        
        # Setup GP regressor with appropriate kernel: 
        # 1. RBF
        if kernel == "rbf":
            kernel = C(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        # 2. Matern
        elif kernel == 'matern':
            kernel = C(1.0) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        # 3. Rational Quadratic
        elif kernel == 'rational_quadratic':
            kernel = C(1.0) * RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        # 4. Exponential
        elif kernel == 'exp_sine_squared':
            kernel = C(1.0) * ExpSineSquared(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        # 5. Dot Product
        elif kernel == 'dot_product':
            kernel = C(1.0) * DotProduct()
        
            
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small noise to avoid numerical issues
            normalize_y=True,
            n_restarts_optimizer=5
        )
    
    def update(self, x, reward):
        """
        Update the GP model with a new observation
        
        Args:
            x: The feature of the selected candidate (index or embedding)
            reward: The observed reward (relevance score)
        """
        x_processed = self._process_x(x)
        self.X.append(x_processed)  # Add the flattened array
        self.y.append(reward)
        # Mark that model needs refitting after new data
        self.is_fitted = False
    
    def _process_x(self, candidates):
        """
        Process candidates into appropriate format for GP
        
        Args:
            candidates: List of candidates (indices or embeddings), already a numpy array
            
        Returns:
            Numpy array of processed candidates
        """
        # convert to numpy array if not already):
        if not isinstance(candidates, np.ndarray):
            candidates = np.array(candidates)
        
        # Ensure proper dimensionality (2D: samples × features)
        if candidates.ndim == 1:
            # If 1D, add a dimension to make it 2D
            candidates = candidates.reshape(1, -1)
            
        return candidates
    
    def _process_candidates(self, candidates):
        if isinstance(candidates, list):
            candidates = np.array(candidates)

        if candidates.ndim == 1:
            candidates = candidates.reshape(-1, 1)
        return candidates
            

    def get_mean_std(self, candidates):
        """
        Get mean and standard deviation predictions for candidates
        
        Args:
            candidates: List of candidates to get predictions for
            
        Returns:
            mu: Mean predictions
            sigma: Standard deviation predictions
        """
        if not self.X:
            # If no observations, return default values
            return np.zeros(len(candidates)), np.ones(len(candidates))
        
        # Convert candidates to appropriate format
        X_candidates = self._process_candidates(candidates)
        
        # Fit GP model only if it hasn't been fitted since last update
        if not self.is_fitted:
            # Ensure proper X format for fitting
            X_train = np.stack(self.X)
            if X_train.ndim != 1:
                X_train = X_train.reshape(-1, X_candidates.shape[1])
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
            
            self.gp.fit(X_train, np.array(self.y))
            self.is_fitted = True
        
        # Predict mean and standard deviation
        mu, sigma = self.gp.predict(X_candidates, return_std=True)
        
        return mu, sigma

    def select(self, candidates, n=1):
        """
        Select the best n candidates to sample next based on GP-UCB
        
        Args:
            candidates: List of candidates (indices or embeddings)
            n: Number of candidates to select
            
        Returns:
            List of selected candidates
        """
        if not self.X:
            # Cold start: if no observations yet, just return the first n candidates
            return np.arange(min(n, len(candidates)))
        
        # Convert candidates to appropriate format
        X_candidates = self._process_candidates(candidates)
        
        # Fit GP model only if it hasn't been fitted since last update
        if not self.is_fitted:
            # Ensure proper X format for fitting
            X_train = np.stack(self.X)
            if X_train.ndim != 1:
                X_train = X_train.reshape(-1, X_candidates.shape[1])
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
                
            # Fit the model
            self.gp.fit(X_train, np.array(self.y))
            self.is_fitted = True
        
        # Predict mean and standard deviation for all candidates
        mu, sigma = self.gp.predict(X_candidates, return_std=True)
        
        # Calculate upper confidence bound
        ucb = mu + np.sqrt(self.beta) * sigma
        
        # Get indices of candidates with highest UCB values
        top_indices = np.argsort(-ucb)[:n]
        
        # Return selected candidates
        return top_indices[:n]
    
    def get_top_k(self, candidates, k):
        """
        Get the top k candidates with highest mean values
        """
        mu, sigma = self.get_mean_std(candidates)
        return np.argsort(-mu)[:k]
    
if __name__ == "__main__":
    # test the retrieval_gpucb class, and plot the results
    # create a simple dataset
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create a synthetic dataset with 2 dimensions
    N = 1000
    np.random.seed(42)  # For reproducibility
    
    # Generate 2-dimensional features
    X_data = np.random.rand(N, 2)
    
    # Define a true function (for demonstration purposes)
    def true_function(x):
        return np.sin(5 * x[:, 0]) * np.cos(3 * x[:, 1]) + x[:, 0] + x[:, 1]
    
    # Generate true values with some noise
    y_true = true_function(X_data) + 0.1 * np.random.randn(N)
    
    # Initialize GP-UCB
    gpucb = RetrievalGPUCB(beta=2.0, kernel='rbf')
    
    # Simulation parameters
    n_iterations = 20
    
    # Keep track of selected points and their rewards
    selected_indices = []
    rewards = []
    
    # Create a grid for visualization
    resolution = 30
    x_grid = np.linspace(0, 1, resolution)
    y_grid = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Compute true function values on grid for comparison
    z_true = true_function(grid_points).reshape(resolution, resolution)
    
    # Visualization setup: create one large figure for all iterations
    plt.figure(figsize=(20, 15))
    
    # Run GP-UCB for n_iterations
    for i in range(n_iterations):
        # Select the next point to sample
        if i == 0:
            # Cold start: select a random point
            next_idx = np.random.randint(0, N)
        else:
            # Use GP-UCB to select the next point
            remaining_indices = [j for j in range(N) if j not in selected_indices]
            remaining_X = X_data[remaining_indices]
            next_idx_in_remaining = gpucb.select(remaining_X)
            # next_idx_in_remaining is the indices of the remaining points
            next_idx = remaining_indices[next_idx_in_remaining[0]]
            
        # Get the reward for the selected point
        reward = y_true[next_idx]
        
        # Update the model
        gpucb.update(X_data[next_idx], reward)
        
        # Store the selected point and reward
        selected_indices.append(next_idx)
        rewards.append(reward)
        
        # Create visualization of the GP-UCB model after each update
        ax = plt.subplot(4, 5, i+1, projection='3d')
        
        # Get model predictions for the grid
        if len(selected_indices) > 0:
            pdb.set_trace()
            mu, sigma = gpucb.get_mean_std(grid_points)
            ucb = mu + np.sqrt(gpucb.beta) * sigma
            
            # Reshape for plotting
            mu = mu.reshape(resolution, resolution)
            sigma = sigma.reshape(resolution, resolution)
            ucb = ucb.reshape(resolution, resolution)
            
            # Plot the mean prediction surface
            surf = ax.plot_surface(xx, yy, mu, cmap=cm.viridis, alpha=0.7, 
                                  linewidth=0, antialiased=True)
            
            # Add contour lines on the bottom
            ax.contourf(xx, yy, mu, zdir='z', offset=mu.min(), cmap=cm.viridis, alpha=0.3)
        
        # Plot the selected points
        if selected_indices:
            ax.scatter(X_data[selected_indices, 0], X_data[selected_indices, 1], 
                      rewards, c='red', marker='o', s=50, label='Sampled points')
            
        # Highlight the most recent point
        if i > 0:
            ax.scatter(X_data[next_idx, 0], X_data[next_idx, 1], 
                      reward, c='black', marker='x', s=100, label='Latest point')
        
        # Set labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Value')
        ax.set_title(f'Iteration {i+1}')
        
        # Set consistent view angle
        ax.view_init(elev=30, azim=45)
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Only add legend on the first plot
        if i == 0:
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('gpucb_iterations_3d.png', dpi=300)
    
    # Create a final visualization with multiple aspects of the GP-UCB model
    plt.figure(figsize=(20, 15))
    
    # 1. True function
    ax1 = plt.subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(xx, yy, z_true, cmap=cm.viridis, alpha=0.7,
                           linewidth=0, antialiased=True)
    ax1.set_title('True Function')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Value')
    
    # 2. Final mean prediction
    ax2 = plt.subplot(2, 2, 2, projection='3d')
    mu, sigma = gpucb.get_mean_std(grid_points)
    mu = mu.reshape(resolution, resolution)
    surf2 = ax2.plot_surface(xx, yy, mu, cmap=cm.plasma, alpha=0.7,
                           linewidth=0, antialiased=True)
    ax2.scatter(X_data[selected_indices, 0], X_data[selected_indices, 1], 
               rewards, c='red', marker='o', s=50, label='Sampled points')
    ax2.set_title('GP Mean Prediction')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('Value')
    ax2.legend()
    
    # 3. Uncertainty (sigma)
    ax3 = plt.subplot(2, 2, 3, projection='3d')
    sigma = sigma.reshape(resolution, resolution)
    surf3 = ax3.plot_surface(xx, yy, sigma, cmap=cm.cool, alpha=0.7,
                           linewidth=0, antialiased=True)
    ax3.scatter(X_data[selected_indices, 0], X_data[selected_indices, 1], 
               np.zeros_like(rewards), c='red', marker='o', s=50, label='Sampled points')
    ax3.set_title('GP Uncertainty (Sigma)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('Uncertainty')
    ax3.legend()
    
    # 4. Upper Confidence Bound
    ax4 = plt.subplot(2, 2, 4, projection='3d')
    ucb = (mu + np.sqrt(gpucb.beta) * sigma)
    surf4 = ax4.plot_surface(xx, yy, ucb, cmap=cm.magma, alpha=0.7,
                           linewidth=0, antialiased=True)
    ax4.scatter(X_data[selected_indices, 0], X_data[selected_indices, 1], 
               rewards, c='red', marker='o', s=50, label='Sampled points')
    ax4.set_title('Upper Confidence Bound (UCB)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('UCB Value')
    ax4.legend()
    
    # Add a colorbar for each surface
    plt.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    plt.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    plt.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig('gpucb_final_analysis.png', dpi=300)
    plt.show()
