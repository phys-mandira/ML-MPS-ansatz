import numpy as np
import torch
import torch.nn as nn

class MPS(nn.Module):
    
    def __init__(self, f_name, input_dim, bond_dim, feature_dim=2, init_std=1e-9,):
        super().__init__()

        # Set the MPS attributes
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        self.bond_dim = bond_dim
        self.f_name = f_name
        self.init_std = init_std


        # Initialize the core tensor defining our model near the identity
        # This tensor holds all of the trainable parameters of our model
        
        tensor_list = []

        half_dim = self.input_dim // 2

        for imx in range(self.input_dim):
        # Determine shape and bond_str for each tensor
            if imx == 0:
                shape = [2, 2]
                bond_str = "ir"
            elif imx == self.input_dim - 1:
                shape = [2, 2]
                bond_str = "li"
            else:
                if imx < half_dim:
                    l_label, r_label = 2 ** imx, 2 ** (imx + 1)
                else:
                    r_label, l_label = 2 ** (self.input_dim - 1 - imx), 2 ** (self.input_dim - imx)

                # Clip labels to bond_dim
                l_label = min(l_label, self.bond_dim)
                r_label = min(r_label, self.bond_dim)
                shape = [l_label, 2, r_label]
                bond_str = "lir"

            tensor = init_tensor(
                bond_str=bond_str,
                shape=shape,
                init_method="random_zero",
                init_std=init_std,
            )
            tensor_list.append(tensor)
            #print("tensor", imx+1, shape)

        module_list = []

        # Load file 
        data = np.genfromtxt(f_name, dtype=str)
        lines = data.shape[0]

        # Convert to numeric where possible
        data = np.array(data)

        for idx in range(self.input_dim):
            # Select only rows corresponding to current index
            rows = data[data[:, 1].astype(int) == idx + 1]

            for row in rows:
                if idx == 0:
                    i, j, val = int(row[4]) - 1, int(row[3]) - 1, float(row[5])
                    tensor_list[idx][i, j] = val
                elif idx == self.input_dim - 1:
                    i, j, val = int(row[2]) - 1, int(row[4]) - 1, float(row[5])
                    tensor_list[idx][i, j] = val
                else:
                    i, j, k, val = int(row[2]) - 1, int(row[4]) - 1, int(row[3]) - 1, float(row[5])
                    tensor_list[idx][i, j, k] = val

            tensor_list[idx].requires_grad = True
            module_list.append(InputRegion(tensor_list[idx]))
        
        #print("tensor_list[0]", tensor_list[0])

        self.linear_region = LinearRegion(module_list=module_list,)
        assert len(self.linear_region) == self.input_dim
        
        # Set the rest of our MPS attributes
        self.module_list = module_list

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

            # Initialize embedded_data directly as a torch tensor
            embedded_data = torch.zeros((*input_data.shape, self.feature_dim), dtype=torch.float32)

            # Boolean mask for input == -1.0
            mask = (input_data == -1.0)

            # Set values based on mask
            embedded_data[..., 0] = torch.where(mask, 0.01, 0.99)
            embedded_data[..., 1] = torch.where(mask, 0.99, 0.01)
            
            #print("input_data", input_data)
            #print("embedded_data", embedded_data)

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
        # Validate input
        assert len(input_data.shape) == 3, "input_data must be 3D (batch, length, feature_dim)"
        assert input_data.size(1) == len(self), "Mismatch between input length and module list length"

        # Move device info
        device = input_data.device

        # Transpose input 
        # Old shape: (batch, length, feature_dim)
        # Needed shape: (length, batch, feature_dim)
        mod_input = input_data.permute(1, 0, 2).contiguous()

        contractable_list = []

        # Compute local contractions (einsum over each module tensor)
        for idx, module in enumerate(self.module_list):
            tens = module.tensor

            if idx == 0:  # left tensor: bond_str = "ir"
                mats = torch.einsum("ir,bi->br", tens, mod_input[idx])
            elif idx == len(self.module_list) - 1:  # right tensor: bond_str = "li"
                mats = torch.einsum("li,bi->bl", tens, mod_input[idx])
            else:  # middle tensor: bond_str = "lir"
                mats = torch.einsum("lir,bi->blr", tens, mod_input[idx])

            contractable_list.append(mats)

        # Sequential bond contractions
        tem = contractable_list[0].unsqueeze(1)  # (batch, 1, bond)
        for nl in range(1, len(contractable_list)):
            nxt = contractable_list[nl]
            if nl == len(contractable_list) - 1:
                nxt = nxt.unsqueeze(-1)  # final reshape for right-end contraction
            tem = torch.bmm(tem, nxt)

        output = tem.squeeze(-1)  # remove trailing dimension
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

