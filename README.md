# Introduction

Matrix product state (MPS) is a popular wave function ansatz in electronic structure theory. MPS can be obtained from multi-dimensional exact coefficient tensor via singular value decomposition (SVD). It  efficiently compress the multi-dimensional exact coefficient tensor to avoid the curse of dimensionality and computational cost of higher rank tensor. 

Here, supervised machine learning is employed to optimize the MPS. The data for learning can be generated from any ab initio method. Input is the spin configurations on each site (+1 for up spin and -1 for down spin) and ouput is the magnitude of CI coefficient corresponding to the input configuration. As the MPS gets optimize, the energy and other quantum state properties also get converge to the actual   values.

# Prerequisites :
1. Python 3.0+
2. PyTorch
3. TorchMPS

# Contributors :
1. Mandira Dey
2. Debashree Ghosh

# Compilation :
        a) First, install the TorchMPS software in workstation: git clone https://github.com/jemisjoky/TorchMPS.git
        b) Then install this code.

#How to run this code ?
Modify inputs according to the system of interest in "input.in" file.
python3 mps_train.py input.in &

# Input arguments
1. M             :      INT
                        Bond dimention, represents the degree of correlation between two consecutive sites of MPS.
2. batch_size    :      INT
                        Batch size gives the amount of train data in a given batch while training of MPS.
3. num_epochs    :      INT
                        The number of epochs/iterations for model training.
4. learn_rate    :      FLOAT
                        Learning rate of the model.
5. l2_reg        :      FLOAT
                        This is L2 regularization element in typical machine learning.
6. inp_dim       :      INT
                        Dimension of input descriptor.
7. csv_column   :       (INT a, INT b, INT c, INT d)
                        The range of input, output descriptor and determinant serial number should be given. Where, input ranges from 'a' to 'b' column of a csv file.
                        c and d representing the determinant number and output descriptor respectively.
8. path         :       STR
                        Path of the directory where the csv data file is placed and ouput files will be generated.
9. input_file   :       STR
                        Name of input data file in csv format.
10. reference_file   :  STR
                        Name of reference file where unconverged TEBD wavefunction is saved.

# Generated output files
After successful training, there will be three type of output file - training and testing loss with iterations, energy and Sz value with iterations, actual vs predicted output both for training and testing data. The converged model will be stored in *converged_model* file. 

