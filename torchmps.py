import numpy as np
import torch
import torch.nn as nn

class MPS(nn.Module):
    
    def __init__(
        self,
        f_name,
        input_dim,
        bond_dim,
        feature_dim=2,
        init_std=1e-9,
    ):
        super().__init__()

        # Initialize the core tensor defining our model near the identity
        # This tensor holds all of the trainable parameters of our model
        
        tensor_list = []

        for imx in range(int(input_dim/2)):
            if imx == 0:
                shape = [2,2]
                tensor = init_tensor(bond_str="ir", shape=shape, init_method=("random_zero"), init_std=init_std,)
                tensor_list.append(tensor)
            else:
                l_label = 2**imx
                r_label = 2**(imx+1)

                if l_label <= bond_dim and r_label >= bond_dim:
                    r_label = bond_dim
                    shape = [l_label,2,r_label]
                    tensor = init_tensor(bond_str="lir", shape=shape, init_method=("random_zero"), init_std=init_std,)
                    tensor_list.append(tensor)

                elif l_label >= bond_dim and r_label >= bond_dim:
                    l_label = bond_dim
                    r_label = bond_dim
                    shape = [l_label,2,r_label]
                    tensor = init_tensor(bond_str="lir", shape=shape, init_method=("random_zero"), init_std=init_std,)
                    tensor_list.append(tensor)

                else:
                    shape = [l_label,2,r_label]
                    tensor = init_tensor(bond_str="lir", shape=shape, init_method=("random_zero"), init_std=init_std,)
                    tensor_list.append(tensor)

        
        for imx in range(int(input_dim/2), input_dim):
            if imx == input_dim-1:
                shape = [2,2]
                tensor = init_tensor(bond_str="li", shape=shape, init_method=("random_zero"), init_std=init_std,)
                tensor_list.append(tensor)
            else:
                r_label = 2**(input_dim-1-imx)
                l_label = 2**(input_dim-1-imx+1)

                if l_label >= bond_dim and r_label >= bond_dim:
                    l_label = bond_dim
                    r_label = bond_dim
                    shape = [l_label,2,r_label]
                    tensor = init_tensor(bond_str="lir", shape=shape, init_method=("random_zero"), init_std=init_std,)
                    tensor_list.append(tensor)

                elif l_label >= bond_dim and r_label <= bond_dim:
                    l_label = bond_dim
                    shape = [l_label,2,r_label]
                    tensor = init_tensor(bond_str="lir", shape=shape, init_method=("random_zero"), init_std=init_std,)
                    tensor_list.append(tensor)

                else:
                    shape = [l_label,2,r_label]
                    tensor = init_tensor(bond_str="lir", shape=shape, init_method=("random_zero"), init_std=init_std,)
                    tensor_list.append(tensor)
        
        
        
        module_list = []

        file_name_1 = f_name
        fin = open(file_name_1)
        lines = len(fin.readlines())
        fin.close()

        for idx in range(input_dim):
            if idx == 0:
                for i in range(lines):
                    data_1 = np.genfromtxt(file_name_1, dtype = str, skip_header=i,  max_rows=1)
                    if int(data_1[1]) == idx+1:
                        tensor_list[idx][int(data_1[4])-1, int(data_1[3])-1] = float(data_1[5])

            elif idx == input_dim-1:
                for i in range(lines):
                    data_1 = np.genfromtxt(file_name_1, dtype = str, skip_header=i,  max_rows=1)
                    if int(data_1[1]) == idx+1:
                        tensor_list[idx][int(data_1[2])-1, int(data_1[4])-1] = float(data_1[5])

            else:
                for i in range(lines):
                    data_1 = np.genfromtxt(file_name_1, dtype = str, skip_header=i,  max_rows=1)
                    if int(data_1[1]) == idx+1:
                        tensor_list[idx][int(data_1[2])-1, int(data_1[4])-1, int(data_1[3])-1] = float(data_1[5])


            tensor_list[idx].requires_grad = True
            module_list.append(InputRegion(tensor_list[idx]))


        self.linear_region = LinearRegion(module_list=module_list,)
        assert len(self.linear_region) == input_dim
        

        # Set the rest of our MPS attributes
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        self.bond_dim = bond_dim
        self.module_list = module_list
        self.f_name = f_name
        self.init_std = init_std 

    def forward(self, input_data):
       
        # Embed our input data before feeding it into our linear region
        input_data = self.embed_input(input_data)
        output = self.linear_region(input_data)

        return output


    def embed_input(self, input_data):
        """
        Embed pixels of input_data into separate local feature spaces

        Args:
            input_data (Tensor):    Input with shape [batch_size, input_dim], or
                                [batch_size, input_dim, feature_dim]. In the
                                latter case, the data is assumed to already
                                be embedded, and is returned unchanged.

        Returns:
            embedded_data (Tensor): Input embedded into a tensor with shape
                                [batch_size, input_dim, feature_dim]
        """
        assert len(input_data.shape) in [2, 3]
        assert input_data.size(1) == self.input_dim

        # If input already has a feature dimension, return it as is
        if len(input_data.shape) == 3:
            if input_data.size(2) != self.feature_dim:
                raise ValueError(
                    f"input_data has wrong shape to be unembedded "
                    "or pre-embedded data (input_data.shape = "
                    f"{list(input_data.shape)}, feature_dim = {self.feature_dim})"
                )
            return input_data

        # Otherwise, use a simple linear embedding map with feature_dim = 2
        else:
            if self.feature_dim != 2:
                raise RuntimeError(
                    f"self.feature_dim = {self.feature_dim}, "
                    "but default feature_map requires self.feature_dim = 2"
                )
            
            embedded_data = np.zeros((input_data.shape[0], input_data.shape[1], self.feature_dim))
            embedded_data = torch.tensor(embedded_data)
            emb_shape = embedded_data.shape
            for i in range(emb_shape[0]):
                for j in range(emb_shape[1]):
                    if input_data[i][j] == -1.0:
                        embedded_data[i][j][0] = 0.01
                        embedded_data[i][j][1] = 0.99
                    else:
                        embedded_data[i][j][0] = 0.99
                        embedded_data[i][j][1] = 0.01
            

        return embedded_data
    
    def __len__(self):
        """
        Returns the number of input sites, which equals the input size
        """
        return self.input_dim
    

class LinearRegion(nn.Module):

    def __init__(
        self, module_list):
        # Check that module_list is a list whose entries are Pytorch modules
        if not isinstance(module_list, list) or module_list is []:
            raise ValueError("Input to LinearRegion must be nonempty list")
        for i, item in enumerate(module_list):
            if not isinstance(item, nn.Module):
                raise ValueError(
                    "Input items to LinearRegion must be PyTorch "
                    f"Module instances, but item {i} is not"
                )
        super().__init__()

        # Wrap as a ModuleList for proper parameter registration
        self.module_list = nn.ModuleList(module_list)

    def forward(self, input_data):

        input_shape = list(input_data.shape)
        mod_input = np.random.rand(input_shape[1],input_shape[0],input_shape[2])

        for mk in range(input_shape[0]):
            for ml in range(input_shape[1]):
                for mp in range(input_shape[2]):
                    mod_input[ml,mk,mp] = input_data[mk,ml,mp]

        mod_input = torch.from_numpy(mod_input).float()

        # Check that input_data has the correct shape
        assert len(input_data.shape) == 3
        assert input_data.size(1) == len(self)

        # Whether to move intermediate vectors to a GPU (fixes Issue #8)
        to_cuda = input_data.is_cuda
        device = f"cuda:{input_data.get_device()}" if to_cuda else "cpu"

        contractable_list = []

        for idx in range(len(self.module_list)):
            if idx == 0:
                # Contract the input with left tensor
                mats = torch.einsum("ir,bi->br", [self.module_list[idx].tensor, mod_input[idx]])
                contractable_list.append(mats)

            elif idx == len(self.module_list)-1:
                # Contract the input with right tensor
                mats = torch.einsum("li,bi->bl", [self.module_list[idx].tensor, mod_input[idx]])
                contractable_list.append(mats)

            else:
                # Contract the input with middle tensor
                mats = torch.einsum("lir,bi->blr", [self.module_list[idx].tensor, mod_input[idx]])
                contractable_list.append(mats)


        for nl in range(len(self.module_list)-1):
            if nl == 0:
                shape = list(contractable_list[nl].shape)
                contractable_list[nl] = torch.reshape(contractable_list[nl], (shape[0], 1, shape[1]))
                tem = torch.bmm(contractable_list[nl], contractable_list[nl+1])

            elif nl == len(self.module_list)-2:
                shape = list(contractable_list[nl+1].shape)
                contractable_list[nl+1] = torch.reshape(contractable_list[nl+1], (shape[0], shape[1], 1))
                tem = torch.bmm(tem, contractable_list[nl+1])

            else:
                tem = torch.bmm(tem, contractable_list[nl+1])

        output = torch.squeeze(tem, (2))

        return output

    def __len__(self):
        """
        Returns the number of input sites, which is the required input size
        """
        return len(self.module_list) 



class InputRegion(nn.Module):
    
    def __init__(
        self, tensor):
        super().__init__()

        self.register_parameter(name="tensor", param=nn.parameter.Parameter(data=tensor, requires_grad=True))

    
    def __len__(self):
        return self.tensor.size(0)

def init_tensor(shape, bond_str, init_method, init_std):

    # Check that bond_str is properly sized and doesn't have repeat indices
    assert len(shape) == len(bond_str)
    assert len(set(bond_str)) == len(bond_str)

    if init_method not in ["random_zero", "zeros"]:
        raise ValueError(f"Unknown initialization method: {init_method}")

    if init_method == "random_zero":
        tensor = init_std * torch.randn(shape)

    elif init_method == "zeros":
        tensor = torch.zeros(shape)


    return tensor

