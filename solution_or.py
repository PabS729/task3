"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, NonlinearConstraint
from sklearn import gaussian_process as gp
# import additional ...
from scipy.stats import norm


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.beta = 3
        self.beta_v = 4
        self.mean = 4
        self.var = 0.0001
        self.gpf = gp.GaussianProcessRegressor(kernel = 0.5 * gp.kernels.Matern(length_scale=1, nu=2.5, length_scale_bounds="fixed"))
        self.gpv = gp.GaussianProcessRegressor(kernel = np.sqrt(2) * gp.kernels.Matern(length_scale=1,nu=2.5, length_scale_bounds="fixed"), normalize_y=True)
        self.x = []
        self.f = []
        self.v = []

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        x_next = self.optimize_acquisition_function()
        return

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        def upperb(x):
            mean_v,std_v = self.gpv.predict(np.atleast_2d(x), return_std=True)
            fc = (mean_v + self.beta_v * std_v)
            return np.mean(fc)
        
        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            const = NonlinearConstraint(fun=upperb, lb=SAFETY_THRESHOLD, ub=np.inf)
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        mean, std = self.gpf.predict(x, return_std=True)
        mean_v, std_v = self.gpv.predict(x, return_std=True)
        nd = norm
        fmax = 1.0
        xi = 0

        #Z = (mean - fmax - xi) / std 
        #af_value = (mean - fmax - xi) * nd.cdf(Z) + std * norm.pdf(Z)
        #print(af_value.shape)
        af_value = mean +self.beta * std
        if mean_v + std_v*1e-3 <= SAFETY_THRESHOLD:
            af_value -= (mean_v+SAFETY_THRESHOLD * 2)

        return np.mean(af_value)

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        self.x.append(x)
        self.f.append(f)
        self.v.append(v)

        x_n = np.reshape(self.x, [-1,1])
        f_n = np.reshape(self.f, [-1,1])
        v_n = np.reshape(self.v, [-1,1])

        #print(x_n, f_n, v_n)

        self.gpf.fit(x_n, f_n)
        self.gpv.fit(x_n, v_n)


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        mask = np.array(self.v) > SAFETY_THRESHOLD
        f_msk = np.array(self.f)[mask]
        x_msk = np.array(self.x)[mask]
        ind = np.argmax(f_msk)
        return np.atleast_2d(x_msk[ind])

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
