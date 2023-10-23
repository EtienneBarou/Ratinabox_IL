from ratinabox import Neurons
import numpy as np
import matplotlib as plt 

class PyramidalNeurons(Neurons):
    """
    The PyramidalNeuorn class defines a layer of Neurons() whos firing rates are derived from the 
    firing rates in two DendriticCompartments. 
    They are theta modulated, during early theta phase the apical DendriticCompartment (self.apical_compartment) 
    drives the soma, during late theta phases the basal DendriticCompartment (self.basal_compartment) drives the 
    soma.

    Must be initialised with an Agent and a 'params' dictionary.

    Check that the input layers are all named differently.
    List of functions:
        • get_state()
        • update()
        • update_dendritic_compartments()
        • update_weights()
        • plot_loss()
        • plot_rate_map()
    """

    def __init__(self, Agent, params={}):
        """Initialises a layer of pyramidal neurons

        Args:
            Agent (_type_): _description_
            params (dict, optional): _description_. Defaults to {}.
        """
        default_params = {
            "n": 10,
            "name": "PyramidalNeurons",
            # theta params
            "theta_freq": 8,
            "theta_frac": 0.5,  # -->0 all basal input, -->1 all apical input
        }
        default_params.update(params)
        self.params = default_params
        super().__init__(Agent, self.params)

        self.history["loss"] = []
        self.error = None

        self.basal_compartment = DendriticCompartment(
            self.Agent,
            params={
                "soma": self,
                "name": f"{self.name}_basal",
                "n": self.n,
                "color": self.color,
            },
        )
        self.apical_compartment = DendriticCompartment(
            self.Agent,
            params={
                "soma": self,
                "name": f"{self.name}_apical",
                "n": self.n,
                "color": self.color,
            },
        )

    def update(self):
        """
        Updates the firing rate of the layer. 
        Saves a loss (lpf difference between basal and apical). Also adds noise.
        """
        super().update()  # this sets and saves self.firingrate

        dt = self.Agent.dt
        tau_smooth = 10
        # update a smoothed history of the loss
        fr_b, fr_a = (
            self.basal_compartment.firingrate,
            self.apical_compartment.firingrate,
        )
        error = np.mean(np.abs(fr_b - fr_a))
        if self.Agent.t < 2 / self.theta_freq:
            self.error = None
        else:
            # loss_smoothing_timescale = dt
            self.error = (dt / tau_smooth) * error + (1 - dt / tau_smooth) * (
                self.error or error
            )
        self.history["loss"].append(self.error)
        return

    def update_dendritic_compartments(self):
        """Individually updates the basal and apical firing rates."""
        self.basal_compartment.update()
        self.apical_compartment.update()
        return

    def get_state(self, evaluate_at="last", **kwargs):
        """
        Returns the firing rate of the soma. 
        This depends on the firing rates of the basal and apical compartments and the current theta phase. 
        By default the theta is obtained from self.Agent.t but it can be passed manually as an kwarg to override this.

        theta (or theta_gating) is a number between [0,1] controlling flow of information into soma from the two compartment.s 
        0 = entirely basal. 1 = entirely apical. 
        Between equals weighted combination. The function theta_gating() takes a time and returns theta.
        
        Args:
            evaluate_at (str, optional): 'last','agent','all' or None (in which case pos can be passed directly as a kwarg). Defaults to "last".
        Returns:
            firingrate
        """
        # theta can be passed in manually as a kwarg. 
        # If it isn't ithe time from the agent will be used to get theta. 
        # Theta determines how much basal and how much apical this neurons uses.
        if "theta" in kwargs:
            theta = kwargs["theta"]
        else:
            theta = theta_gating(
                t=self.Agent.t, freq=self.theta_freq, frac=self.theta_frac
            )
        fr_basal, fr_apical = 0, 0
        # these are special cases, no need to even get their fr's if they aren't used
        if theta != 0:
            fr_apical = self.apical_compartment.get_state(evaluate_at, **kwargs)
        if theta != 1:
            fr_basal = self.basal_compartment.get_state(evaluate_at, **kwargs)
        firingrate = (1 - theta) * fr_basal + (theta) * fr_apical
        return firingrate

    def update_weights(self):
        """Trains the weights, this function actually defined in the dendrite class."""
        if self.Agent.t > 2 / self.theta_freq:
            self.basal_compartment.update_weights()
            self.apical_compartment.update_weights()
        return

    def plot_loss(self, fig=None, ax=None):
        """Plots the loss against time to see if learning working"""
        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=(1.5, 1.5))
            ylim = 0
        else:
            ylim = ax.get_ylim()[1]
        t = np.array(self.history["t"]) / 60
        loss = self.history["loss"]
        ax.plot(t, loss, color=self.color, label=self.name)
        ax.set_ylim(
            bottom=0, top=max(ylim, np.nanmax(np.array(loss, dtype=np.float64)))
        )
        ax.set_xlim(left=0)
        ax.legend(frameon=False)
        ax.set_xlabel("Training time / min")
        ax.set_ylabel("Loss")
        return fig, ax

    def plot_rate_map(self, route="basal", **kwargs):
        """
        This is a wrapper function for the general Neuron class function plot_rate_map. 
        It takes the same arguments as Neurons.plot_rate_map() but, in addition, 
        route can be set to basal or apical in which case theta is set correspondingly and teh soma with take its 
        input from downstream or upstream sources entirely.

        The arguments for the standard plottiong function plot_rate_map() can be passed as usual as kwargs.

        Args:
            route (str, optional): _description_. Defaults to 'basal'.
        """
        if route == "basal":
            theta = 0
        elif route == "apical":
            theta = 1
        fig, ax = super().plot_rate_map(**kwargs, theta=theta)
        return fig, ax


class DendriticCompartment(Neurons):
    """
    The DendriticCompartment class defines a layer of Neurons() whos firing rates are an activated 
    linear combination of input layers. This class is a subclass of Neurons() and inherits it 
    properties/plotting functions.

    Must be initialised with an Agent and a 'params' dictionary.
    Input params dictionary must  contain a list of input_layers which feed into these Neurons. 
    This list looks like [Neurons1, Neurons2,...] where each is a Neurons() class.

    Currently supported activations include 'sigmoid' (paramterised by max_fr, min_fr, mid_x, width), 
    'relu' (gain, threshold) and 'linear' specified with the "activation_params" dictionary in the inout params dictionary. 
    See also activate() for full details.

    Check that the input layers are all named differently.
    List of functions:
        • get_state()
        • add_input()
    """

    def __init__(self, Agent, params={}):
        default_params = {
            "soma": None,
            "activation_params": {
                "activation": "sigmoid",
                "max_fr": 1,
                "min_fr": 0,
                "mid_x": 1,
                "width_x": 2,
            },
        }
        self.Agent = Agent
        default_params.update(params)
        self.params = default_params
        super().__init__(Agent, self.params)

        self.firingrate_temp = None
        self.firingrate_prime_temp = None
        self.inputs = {}

    def add_input(
        self, input_layer, eta=0.001, w_init=0.1, L1=0.0001, L2=0.001, tau_PI=100e-3
    ):
        """
        Adds an input layer to the class. Each input layer is stored in a dictionary of self.inputs. 
        Each has an associated matrix of weights which are initialised randomly.

        Args:
            input_layer (_type_): the layer which feeds into this compartment
            eta: learning rate of the weights
            w_init: initialisation scale of the weights
            L1: how much L1 regularisation
            L2: how much L2 regularisation
            tau_PI: smoothing timescale of plasticity induction variable
        """
        name = input_layer.name
        n_in = input_layer.n
        w = np.random.normal(loc=0, scale=w_init / np.sqrt(n_in), size=(self.n, n_in))
        I = np.zeros(n_in)
        PI = np.zeros(n_in)
        if name in self.inputs.keys():
            print(
                f"There already exists a layer called {input_layer_name}. Overwriting it now."
            )
        self.inputs[name] = {}
        self.inputs[name]["layer"] = input_layer
        self.inputs[name]["w"] = w
        self.inputs[name]["w_init"] = w.copy()
        self.inputs[name]["I"] = I  # input current
        self.inputs[name]["I_temp"] = None  # input current
        self.inputs[name]["PI"] = PI  # plasticity induction variable
        self.inputs[name]["eta"] = eta
        self.inputs[name]["L2"] = L2
        self.inputs[name]["L1"] = L1
        self.inputs[name]["tau_PI"] = tau_PI

    def get_state(self, evaluate_at="last", **kwargs):
        """
        Returns the "firing rate" of the dendritic compartment. 
        By default this layer uses the last saved firingrate from its input layers. 
        Alternatively evaluate_at and kwargs can be set to be anything else which will just be passed 
        to the input layer for evaluation.
        Once the firing rate of the inout layers is established these are multiplied by the weight matrices and 
        then activated to obtain the firing rate of this FeedForwardLayer.

        Args:
            evaluate_at (str, optional). Defaults to 'last'.
        Returns:
            firingrate: array of firing rates
        """
        if evaluate_at == "last":
            V = np.zeros(self.n)
        elif evaluate_at == "all":
            V = np.zeros(
                (self.n, self.Agent.Environment.flattened_discrete_coords.shape[0])
            )
        else:
            V = np.zeros((self.n, kwargs["pos"].shape[0]))

        for inputlayer in self.inputs.values():
            w = inputlayer["w"]
            if evaluate_at == "last":
                I = inputlayer["layer"].firingrate
            else:  # kick can down the road let input layer decide how to evaluate the firingrate
                I = inputlayer["layer"].get_state(evaluate_at, **kwargs)
            inputlayer["I_temp"] = I
            V += np.matmul(w, I)
        firingrate = utils.activate(V, other_args=self.activation_params)
        firingrate_prime = utils.activate(
            V, other_args=self.activation_params, deriv=True
        )

        self.firingrate_temp = firingrate
        self.firingrate_prime_temp = firingrate_prime

        return firingrate

    def update(self):
        """Updates firingrate of this compartment and saves it to file"""
        self.get_state()
        self.firingrate = self.firingrate_temp.reshape(-1)
        self.firingrate_deriv = self.firingrate_prime_temp.reshape(-1)
        for inputlayer in self.inputs.values():
            inputlayer["I"] = inputlayer["I_temp"].reshape(-1)
        self.save_to_history()
        return

    def update_weights(self):
        """Implements the weight update: dendritic prediction of somatic activity."""
        target = self.soma.firingrate
        delta = (target - self.firingrate) * (self.firingrate_deriv)
        dt = self.Agent.dt
        for inputlayer in self.inputs.values():
            eta = inputlayer["eta"]
            if eta != 0:
                tau_PI = inputlayer["tau_PI"]
                assert (dt / tau_PI) < 0.2
                I = inputlayer["I"]
                w = inputlayer["w"]
                # first updates plasticity induction variable (smoothed delta error outer product with the input current for this input layer)
                PI_old = inputlayer["PI"]
                PI_update = np.outer(delta, I)
                PI_new = (dt / tau_PI) * PI_update + (1 - dt / tau_PI) * PI_old
                inputlayer["PI"] = PI_new
                # updates weights
                dw = eta * (
                    PI_new - inputlayer["L2"] * w - inputlayer["L1"] * np.sign(w)
                )
                inputlayer["w"] = w + dw
        return


def theta_gating(t, freq=8, frac=0.5):
    T = 1 / freq
    phase = ((t / T) % 1) % 1
    if phase < frac:
        return 1
    elif phase >= frac:
        return 0