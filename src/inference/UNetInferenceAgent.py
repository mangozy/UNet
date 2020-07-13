"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        
        raise NotImplementedError

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()
        
        # ((self.model(data)).cpu()).detach().numpy()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        # slices = []
      
        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        # <YOUR CODE HERE>
        
        # data = volume

        for i, s in enumerate(volume):
            pass
        
        # new_shape = (volume.shape[0], 1, volume.shape[1], volume.shape[2])
        new_shape = (volume.shape[0], volume.shape[1], volume.shape[2])

        new_volume = med_reshape(volume, new_shape=new_shape)
        
        data = torch.from_numpy(new_volume).to(device='cuda',dtype=torch.float)
        prediction = self.model(data)                                                     
        print(prediction.shape)
        pred = prediction[prediction.shape[0],1,prediction.shape[2],prediction.shape[3]]
        print(pred.shape)            
        pred_newshape = np.reshape(pred,newshape=(prediction.shape[0],prediction.shape[2],prediction.shape[3]))
        print(pred_newshape.shape)
        print("haha...")
        
        return pred_newshape
