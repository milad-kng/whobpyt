"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Davide Momi, Sorenza Bastiaens, Parsa Oveisi, Kevin Kadak, Taha Morshedzadeh, Shreyas Harita 
Neural Mass Model fitting module for JR with forward, backward, and lateral connection for EEG
"""

# @title new function PyTepFit

# Pytorch stuff


"""
Importage
"""
import torch
from torch.nn.parameter import Parameter
from whobpyt.datatypes import AbstractNMM, par
from whobpyt.models.JansenRit.ParamsJR import ParamsJR
from whobpyt.functions.arg_type_check import method_arg_type_check


# PyTorch stuff
from torch.nn.parameter import Parameter as ptParameter
from torch.nn import ReLU as ptReLU
from torch.linalg import norm as ptnorm
from torch import (tensor as pttensor, float32 as ptfloat32, sum as ptsum, exp as ptexp, diag as ptdiag, 
                   transpose as pttranspose, zeros_like as ptzeros_like, int64 as ptint64, randn as ptrandn, 
                   matmul as ptmatmul, tanh as pttanh, matmul as ptmatmul, reshape as ptreshape, sqrt as ptsqrt,
                   ones as ptones, cat as ptcat)

import numpy as np
from numpy import ones,zeros

class RNNJANSEN(AbstractNMM):
    """
    A module for forward model (JansenRit) to simulate EEG signals

    Attibutes
    ---------
    state_size : int
        Number of states in the JansenRit model

    output_size : int
        Number of EEG channels.

    node_size: int
        Number of ROIs

    hidden_size: int
        Number of step_size for each sampling step

    step_size: float
        Integration step for forward model

    tr : float # TODO: CHANGE THE NAME TO sampling_rate
        Sampling rate of the simulated EEG signals

    TRs_per_window: int # TODO: CHANGE THE NAME
        Number of EEG signals to simulate

    sc: ndarray (node_size x node_size) of floats
        Structural connectivity

    lm: ndarray of floats
        Leadfield matrix from source space to EEG space

    dist: ndarray of floats
        Distance matrix

    use_fit_gains: bool
        Flag for fitting gains. 1: fit, 0: not fit

    use_fit_lfm: bool
        Flag for fitting the leadfield matrix. 1: fit, 0: not fit

    # FIGURE OUT: g, c1, c2, c3, c4: tensor with gradient on
    #     model parameters to be fit

    std_in: tensor with gradient on
        Standard deviation for input noise

    params: ParamsJR
        Model parameters object.


    Methods
    -------
    createIC(self, ver):
        Creates the initial conditions for the model.

    createDelayIC(self, ver):
        Creates the initial conditions for the delays.

    setModelParameters(self):
        Sets the parameters of the model.

    forward(input, noise_out, hx)
        Forward pass for generating a number of EEG signals with current model parameters

    """

    def __init__(self, params: ParamsJR, node_size=200, mode_size = 20, eigvecs=np.ones((200,200)),eigvals=np.ones((200,1)), \
                    sc_mode = 1/20*np.ones((20,20)), use_fit_gains = True,
                 TRs_per_window=20, step_size=0.0001, output_size= 62, tr=0.001, lm=np.ones((62,200)), dist=np.ones((200,200))):
        """
        Parameters
        ----------
        node_size: int
            Number of ROIs
        TRs_per_window: int # TODO: CHANGE THE NAME
            Number of EEG signals to simulate
        step_size: float
            Integration step for forward model
        output_size : int
            Number of EEG channels.
        tr : float # TODO: CHANGE THE NAME TO sampling_rate
            Sampling rate of the simulated EEG signals
        sc: ndarray node_size x node_size float array
            Structural connectivity
        lm: ndarray float array
            Leadfield matrix from source space to EEG space
        dist: ndarray float array
            Distance matrix
        use_fit_gains: bool
            Flag for fitting gains. 1: fit, 0: not fit
        params: ParamsJR
            Model parameters object.
        """
        method_arg_type_check(self.__init__) # Check that the passed arguments (excluding self) abide by their expected data types

        super(RNNJANSEN, self).__init__(params)
        
        self.pop_names = np.array(['P', 'E', 'I'])
        self.state_names = np.array(['current', 'voltage'])
        self.output_names = ["eeg"]
        self.track_params = [] #Is populated during setModelParameters()

        self.model_name = "JR_mode"
        self.pop_size = 3  # 3 populations JR
        self.state_size = 2  # 2 states in each population
        self.tr = tr  # tr ms (integration step 0.1 ms)
        self.step_size = torch.tensor(step_size, dtype=torch.float32)  # integration step 0.1 ms
        self.steps_per_TR = int(tr / step_size)
        self.TRs_per_window = TRs_per_window  # size of the batch used at each step
        self.node_size = node_size  # num of ROI
        self.mode_size = mode_size  #
        self.output_size = output_size  # num of EEG channels
        self.eigvecs = eigvecs  # matrix node_size x node_size eigvectors based on geometry 
        self.eigvals = eigvals  # matrix node_size x 1
        self.sc_mode = sc_mode
        self.dist = torch.tensor(dist, dtype=torch.float32)
        self.lm = lm
        self.use_fit_gains = use_fit_gains
        self.params = params
        self.output_size = lm.shape[0]  # number of EEG channels

        self.setModelParameters()
        self.setModelSCParameters()

    def createIC(self, ver):
        """
        Creates the initial conditions for the model.

        Parameters
        ----------
        ver : int # TODO: ADD MORE DESCRIPTION
            Initial condition version. (in the JR model, the version is not used. It is just for consistency with other models)

        Returns
        -------
        torch.Tensor
            Tensor of shape (node_size, state_size) with random values between `state_lb` and `state_ub`.
        """

        state_lb = -0.1
        state_ub = 0.1
        state_mod_lb = 0
        state_mod_ub = 1
        state_ini = torch.tensor(np.random.uniform(state_lb, state_ub, (self.node_size, self.pop_size, self.state_size)),
                             dtype=torch.float32)
        y = self.eigvals[:self.mode_size]
        x = -1*np.log((self.eigvals.max()-y)/y)
        state_mod_ini = torch.tensor(np.array([x,x,x]).T, dtype=torch.float32)
        return state_ini, state_mod_ini

    def createDelayIC(self, ver):
        """
        Creates the initial conditions for the delays.

        Parameters
        ----------
        ver : int
            Initial condition version. (in the JR model, the version is not used. It is just for consistency with other models)

        Returns
        -------
        torch.Tensor
            Tensor of shape (node_size, delays_max) with random values between `state_lb` and `state_ub`.
        """

        delays_max = 500
        state_ub = 0.1
        state_lb = 0

        return torch.tensor(np.random.uniform(state_lb, state_ub, (self.node_size,  delays_max)), dtype=torch.float32)

    def setModelSCParameters(self):
        """
        Sets the parameters of the model.
        """
         # Create the arrays in numpy
        small_constant = 0.05
        n_nodes = self.node_size
        zsmat = zeros((self.mode_size, self.mode_size)) + small_constant 
        w_p2e = zsmat.copy() # the pyramidal to excitatory interneuron cross-layer gains
        w_p2i = zsmat.copy() # the pyramidal to inhibitory interneuron cross-layer gains
        w_p2p = zsmat.copy() # the pyramidal to pyramidal cells same-layer gains
        
        # Set w_bb, w_ff, and w_ll as attributes as type Parameter if use_fit_gains is True
        if self.use_fit_gains:
            self.w_bb = ptParameter(pttensor(w_p2i, dtype=ptfloat32))
            self.w_ff = ptParameter(pttensor(w_p2e, dtype=ptfloat32))
            self.w_ll = ptParameter(pttensor(w_p2p, dtype=ptfloat32))
            self.params_fitted['modelparameter'].append(self.w_ll)
            self.params_fitted['modelparameter'].append(self.w_ff)
            self.params_fitted['modelparameter'].append(self.w_bb)
        else:
            self.w_bb = torch.tensor(np.zeros((self.mode_size, self.mode_size)), dtype=torch.float32)
            self.w_ff = torch.tensor(np.zeros((self.mode_size, self.mode_size)), dtype=torch.float32)
            self.w_ll = torch.tensor(np.zeros((self.mode_size, self.mode_size)), dtype=torch.float32)

        
    def forward(self, external, hx, hx_mode, hE):
        """
        This function carries out the forward Euler integration method for the JR neural mass model,
        with time delays, connection gains, and external inputs considered. Each population (pyramidal,
        excitatory, inhibitory) in the network is modeled as a non-linear second order system. The function
        updates the state of each neural population and computes the EEG signals at each time step.

        Parameters
        ----------
        external : torch.Tensor
            Input tensor of shape (batch_size, num_ROIs) representing the input to the model.
        hx : Optional[torch.Tensor]
            Optional tensor of shape (state_size, pop_size, num_ROIs) representing the initial hidden state.
        hx_mode : Optional[torch.Tensor]
            Optional tensor of shape (mode_size,  3) representing the initial hidden state.
        hE : Optional[torch.Tensor]
            Optional tensor of shape (batch_size, num_ROIs, delays_max) representing the initial delays.

        Returns
        -------
        next_state : dict
            Dictionary containing the updated current state, EEG signals, and the history of
            each population's current and voltage at each time step.

        hE : torch.Tensor
            Tensor representing the updated history of the pyramidal population's current.
        """

        # Generate the ReLU module
        m = torch.nn.ReLU()

        # Define some constants
        con_1 = torch.tensor(1.0, dtype=torch.float32) # Define constant 1 tensor
        conduct_lb = 0  # lower bound for conduct velocity
        u_2ndsys_ub = 500  # the bound of the input for second order system
        noise_std_lb = 0  # lower bound of std of noise
        lb = 0.  # lower bound of local gains
        k_lb = 0.5  # lower bound of coefficient of external inputs


        # Defining NMM Parameters to simplify later equations
        #TODO: Change code so that params returns actual value used without extras below
        A = 0 * con_1 + m(self.params.A.value())
        a = 0 * con_1 + m(self.params.a.value())
        B = 0 * con_1 + m(self.params.B.value())
        b = 0 * con_1 + m(self.params.b.value())
        g = (lb * con_1 + m(self.params.g.value()))
        g_mode = (lb * con_1 + m(self.params.g_mode.value()))
        c1 = (lb * con_1 + m(self.params.c1.value()))
        c2 = (lb * con_1 + m(self.params.c2.value()))
        c3 = (lb * con_1 + m(self.params.c3.value()))
        c4 = (lb * con_1 + m(self.params.c4.value()))
        std_in = (noise_std_lb * con_1 + m(self.params.std_in.value())) #around 20
        vmax = self.params.vmax.value()
        v0 = self.params.v0.value()
        r = self.params.r.value()
        y0 = self.params.y0.value()
        mu = (0.1 * con_1 + m(self.params.mu.value()))
        k = (5.0 * con_1 + m(self.params.k.value()))
        cy0 = self.params.cy0.value()
        ki = self.params.ki.value()

        g_f = (lb * con_1 + m(self.params.g_f.value()))
        g_b = (lb * con_1 + m(self.params.g_b.value()))
        lm = self.params.lm.value()

        next_state = {}

        P = hx[:, 0:1, 0]  # current of pyramidal population
        E = hx[:, 1:2, 0]  # current of excitory population
        I = hx[:, 2:3, 0]  # current of inhibitory population

        Pv = hx[:, 0:1, 1]  # voltage of pyramidal population
        Ev = hx[:, 1:2, 1]  # voltage of exictory population
        Iv = hx[:, 2:3, 1]  # voltage of inhibitory population
        
        eigval_p2p = hx_mode[:,0:1]
        eigval_p2e = hx_mode[:,1:2]
        eigval_p2i = hx_mode[:,2:3]
        #print(M.shape)
        dt = self.step_size
        n_nodes = self.node_size
        n_chans = self.output_size
        n_modes = self.mode_size
        
        con_con = pttensor(np.ones((n_nodes,1)).dot(np.ones((1,n_nodes))), dtype=ptfloat32)
        pteigvecs = pttensor(self.eigvecs[:,:n_modes], dtype=ptfloat32)
        ptsc = pttensor(self.sc_mode, dtype=ptfloat32)
        
        if self.dist.shape[0] > 1:

            # Update the Laplacian based on the updated connection gains w_bb.
            w_b = ptexp(self.w_bb) * ptsc
            w_n_b = w_b / ptnorm(w_b)#ptreshape(ptsum(w_b, dim=1), [n_modes, 1])#
            self.sc_m_b = w_n_b
            lap_adj_p2i = -1*ptdiag(ptsum(w_n_b, dim=1)) +w_n_b

            # Update the Laplacian based on the updated connection gains w_ff.
            w_f = ptexp(self.w_ff) * ptsc     
            w_n_f = w_f / ptnorm(w_f)#ptreshape(ptsum(w_f, dim=1), [n_modes, 1])#
            self.sc_m_f = w_n_f
            lap_adj_p2e = -1*ptdiag(ptsum(w_n_f, dim=1)) + w_n_f

            # Update the Laplacian based on the updated connection gains w_ll.
            w_l = ptexp(self.w_ll) * ptsc         
            w_n_l = w_l/ ptnorm(w_l)#ptreshape(ptsum(w_l, dim=1), [n_modes, 1])#
            self.sc_fitted = w_n_l
            lap_adj_p2p = -1*ptdiag(ptsum(w_n_l, dim=1)) +w_n_l
        else:
            l_s = torch.tensor(np.zeros((1, 1)), dtype=torch.float32) #TODO: This is not being called anywhere
            dg_l = 0
            dg_b = 0
            dg_f = 0
            w_n_l = 0
            w_n_b = 0
            w_n_f = 0

        self.delays = (self.dist / mu).type(torch.int64)

        # Placeholder for the updated current state
        current_state = ptzeros_like(hx)
        current_state_mode = ptzeros_like(hx_mode)

        # Initializing lists for the history of the EEG signals, as well as each population's current and voltage.
        eeg_window = []
        E_window = []
        I_window = []
        P_window = []
        eigval_p2p_window = []
        eigval_p2e_window = []
        eigval_p2i_window = []
        I_window = []
        P_window = []
        Ev_window = []
        Iv_window = []
        Pv_window = []
        states_window = []
        states_mode_window = []
        
        # Use the forward model to get EEG signal at the i-th element in the window.
        for i_window in range(self.TRs_per_window):
            for step_i in range(self.steps_per_TR):
                #print(eigval_p2p.shape)
                Ed = pttranspose(hE.clone().gather(1,self.delays), 0, 1)
                lap_p2p = ptmatmul(ptmatmul(pteigvecs, ptdiag(self.eigvals.max()/(1+ptexp(-1*eigval_p2p[:,0])))),pteigvecs.T) +0*con_con
                lap_p2e = ptmatmul(ptmatmul(pteigvecs, ptdiag(self.eigvals.max()/(1+ptexp(-1*eigval_p2e[:,0])))),pteigvecs.T) +0*con_con
                lap_p2i = ptmatmul(ptmatmul(pteigvecs, ptdiag(self.eigvals.max()/(1+ptexp(-1*eigval_p2i[:,0])))),pteigvecs.T) +0*con_con
                dg_p2p = -ptdiag(lap_p2p)
                dg_p2e = -ptdiag(lap_p2e)
                dg_p2i = -ptdiag(lap_p2i)
                sc_p2p = ptdiag(lap_p2p) - lap_p2p
                sc_p2e = ptdiag(lap_p2e) - lap_p2e
                sc_p2i = ptdiag(lap_p2i) - lap_p2i
                
                LEd_p2i = (ptreshape(ptsum(sc_p2i * Ed, 1), (n_nodes, 1)) - ptmatmul(dg_p2i, E - I))/ptnorm(sc_p2i)
                LEd_p2e = (ptreshape(ptsum(sc_p2e * Ed, 1), (n_nodes, 1)) + ptmatmul(dg_p2e, E - I))/ptnorm(sc_p2e)
                LEd_p2p = (ptreshape(ptsum(sc_p2p * Ed, 1), (n_nodes, 1)) + ptmatmul(dg_p2p, P))/ptnorm(sc_p2p)

                # external input
                u_stim = external[:, step_i:step_i + 1, i_window, 0]
                
                # Stochastic / noise term
                P_noise = std_in * ptrandn(n_nodes, 1) 
                E_noise = std_in * ptrandn(n_nodes, 1)
                I_noise = std_in * ptrandn(n_nodes, 1)
                
                # Compute the firing rate for each neural populatin 
                # at every node using the wave-to-pulse (sigmoid) functino
                # (vmax = max value of sigmoid, v0 = midpoint of sigmoid)
                P_sigm = vmax / ( 1 + ptexp ( r*(v0 -  (E-I) ) ) )
                E_sigm = vmax / ( 1 + ptexp ( r*(v0 - (c1*P) ) ) )
                I_sigm = vmax / ( 1 + ptexp ( r*(v0 - (c3*P) ) ) )
                # Sum the four different input types into a single input value for each neural 
                # populatin state variable
                # The four input types are:
                # - Local      (L)      - from other neural populations within a node (E->P,P->I, etc.)
                # - Long-range (L-R)    - from other nodes in the network, weighted by the long-range 
                #                         connectivity matrices, and time-delayed
                # - Noise      (N)      - stochastic noise input
                # - External   (E)      - external stimulation, eg from TMS or sensory stimulus
                #
                #        Local    Long-range   Noise   External
                rP =     P_sigm  + g*LEd_p2p   + P_noise + k*ki*u_stim 
                rE =  c2*E_sigm  + g_f*LEd_p2e + E_noise          
                rI =  c4*I_sigm  + g_b*LEd_p2i + I_noise 
                
                # Apply some additional scaling
                rP_bd = u_2ndsys_ub * pttanh(rP / u_2ndsys_ub)
                rE_bd = u_2ndsys_ub * pttanh(rE / u_2ndsys_ub)
                rI_bd = u_2ndsys_ub * pttanh(rI / u_2ndsys_ub)

                # Compute d/dt   ('_tp1' = state variable at time t+1) 
                P_tp1 =  P + dt * Pv
                E_tp1 =  E + dt * Ev
                I_tp1 =  I + dt * Iv
                Pv_tp1 = Pv + dt * ( A*a*rP_bd  -  2*a*Pv  -  a**2 * P )
                Ev_tp1 = Ev + dt * ( A*a*rE_bd  -  2*a*Ev  -  a**2 * E )
                Iv_tp1 = Iv + dt * ( B*b*rI_bd  -  2*b*Iv  -  b**2 * I )
                
                Ieigvec_p2p = g_mode * ptmatmul(lap_adj_p2p, eigval_p2p)-.1*ptmatmul(pteigvecs.T,P) # input currents for E
                Ieigvec_p2i = g_mode * ptmatmul(lap_adj_p2i, eigval_p2i) - .1*ptmatmul(pteigvecs.T,P)
                Ieigvec_p2e = g_mode * ptmatmul(lap_adj_p2e, eigval_p2e) - .1*ptmatmul(pteigvecs.T,P)
                
                eigval_p2p1 = eigval_p2p + dt * (- 10*eigval_p2p + pttanh(Ieigvec_p2p)) +ptsqrt(dt)*0.01*ptrandn(n_modes, 1)
                eigval_p2e1 = eigval_p2e + dt * (- 10*eigval_p2e + pttanh(Ieigvec_p2e)) +ptsqrt(dt)*0.01*ptrandn(n_modes, 1)
                eigval_p2i1 = eigval_p2i + dt * (- 10*eigval_p2i + pttanh(Ieigvec_p2i)) +ptsqrt(dt)*0.01*ptrandn(n_modes, 1)
                
                # Calculate the saturation for model states (for stability and gradient calculation).
                
                # Add some additional saturation on the model states
                # (for stability and gradient calculation).
                P = 1000*pttanh(P_tp1/1000)
                E = 1000*pttanh(E_tp1/1000)
                I = 1000*pttanh(I_tp1/1000)
                Pv = 1000*pttanh(Pv_tp1/1000)
                Ev = 1000*pttanh(Ev_tp1/1000)
                Iv = 1000*pttanh(Iv_tp1/1000)
                eigval_p2p = eigval_p2p1
                eigval_p2e = eigval_p2e1
                eigval_p2i = eigval_p2i1
                #print('after M', M.shape)
                # Update placeholders for pyramidal buffer
                hE[:, 0] = P[:,0]

            # Capture the states at every tr in the placeholders for checking them visually.

            # Capture the states at every tr in the placeholders for checking them visually.
            hE = ptcat([P, hE[:, :-1]], dim=1)  # update placeholders for pyramidal buffer

            # Capture the states at every tr in the placeholders which is then used in the cost calculation.
            lm_t = (lm.T / torch.sqrt((lm ** 2).sum(1))).T
            lm_t_dm = (lm_t - 1 / n_chans * torch.matmul(torch.ones((1, n_chans)), lm_t))
            temp = cy0 * torch.matmul(lm_t_dm, E-I) - 1 * y0
            eeg_window.append(temp)
            states_window.append(torch.cat([torch.cat([P, E, I], dim=1)[:,:,np.newaxis], \
                                   torch.cat([Pv, Ev, Iv], dim=1)[:,:,np.newaxis]], dim=2)[:,:,:,np.newaxis])
            states_mode_window.append(torch.cat([eigval_p2p, eigval_p2e, eigval_p2i], dim=1)[:,:,np.newaxis])
        # Update the current state.
        self.lm_t = lm_t_dm
        
        current_state = torch.cat([torch.cat([P, E, I], dim=1)[:,:,np.newaxis], \
                                   torch.cat([Pv, Ev, Iv], dim=1)[:,:,np.newaxis]], dim=2)
        current_state_mode = torch.cat([eigval_p2p, eigval_p2e, eigval_p2i], dim=1)
        next_state['current_state'] = current_state
        next_state['current_state_mode'] = current_state_mode
        next_state['eeg'] = torch.cat(eeg_window, dim=1)
        next_state['states'] = torch.cat(states_window, dim=3)
        next_state['states_mode'] = torch.cat(states_mode_window, dim=2)


        return next_state, hE
