import sys

inputfile = sys.argv[1]

def readInput():
    fin = open(inputfile,"r")
    lines = fin.readlines()
    length = len(lines)
    for i in range(length):
        toks = lines[i].split(",")
        if len(toks) >= 2:
            if toks[0] == 'M':
                M = int(toks[1])
            
            if toks[0] == 'batch_size':
                batch_size = int(toks[1])

            if toks[0] == 'num_epochs':
                num_epochs = int(toks[1])

            if toks[0] == 'learn_rate':
                learn_rate = float(toks[1])

            if toks[0] == 'l2_reg':
                l2_reg = float(toks[1])

            if toks[0] == 'inp_dim':
                inp_dim = int(toks[1])

            if toks[0] == 'path':
                path = str(toks[1]).strip()
            if toks[0] == 'input_file':
                input_file = str(toks[1]).strip()
            if toks[0] == 'csv_column':
                start = int(toks[1])
                end = int(toks[2])
                det = int(toks[3])
                ci = int(toks[4])
            if toks[0] == 'reference_file':
                reference_file = str(toks[1]).strip()

    inp_list = []
    for i in range(start,end+1):
        inp_list.append(i)
    inp_list.append(ci)
    return M, batch_size, num_epochs, learn_rate, l2_reg, inp_dim, path, input_file, reference_file, inp_list, det



