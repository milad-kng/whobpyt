import torch
import numpy as np

class Recording():
    '''
    This is the class to load in and store one neuroimaing recording or simulated data. 
        - Part of the input and output of model_fitting and model_simulation[future]. 
        - It is the format expected by the visualization function of whobpyt. 
    
    However, recordings() is not used within and across the NMM and Modals, as a simpler dictionary of tensors is used in those contexts. 
    
    Input(NumPy or PyTorch): num_regions x ts_length

    This format is more efficient for plotting and comparison in an objective function.
    Numerical simulation don't use this data structure and are more efficient as the transpose: ts_length x num_regions.
    
    '''
    
    def __init__(self, data, step_size, modality = ""):
        if not(torch.is_tensor(data)):
            data = torch.tensor(data) # Store as Tensor
        
        self.data = data
        self.step_size = step_size
        self.modality = modality
    
    def pyTS(self):
        return self.data
    
    def npTS(self):
        return self.data.numpy()
        
    def npResample(self):
        #This outputs resampled data used for Figures
        pass
    
    def windowedTensor(self, TPperWindow):
        # This adds another dimension for windows used in Model_Fitting
        
        # Output: num_windows x num_regions x window_length
        
        node_size = self.data.shape[0]
        length_ts = self.data.shape[1]
        num_windows = int(length_ts / TPperWindow)
        data_out = np.zeros((num_windows, node_size, TPperWindow))
    
        for i_win in range(num_windows):
            data_out[i_win, :, :] = self.data[:, i_win * TPperWindow:(i_win + 1) * TPperWindow]
    
        return data_out
