# Getting Data Array from text file

def get_datafromfile(filename):
    filepath = './/dataset/' + filename
    data_inputs = []
    output_labels = []
    lines = []
    lengthData = 0
    i = 0
    o = 0
    
    data_pos = {
        'i_start': 0,
        'i_end': 0,
        'o_start': 0,
        'o_end': 0,
        'inputs': 0,
        'outputs': 0
    }
        
    with open(filepath) as f:
        lines = f.readlines()
        
    if filename == 'Flood_dataset.txt':
        data_pos['inputs'] = 8
        data_pos['outputs'] = 1
        data_pos['i_end'] = 8
    
    elif filename == 'xor.txt':
        data_pos['inputs'] = 2
        data_pos['outputs'] = 1
        data_pos['i_end'] = 2
    
    elif filename == 'cross.pat':
        pass
    
    count = 0
    for line in lines:
        count += 1
        if count != 1 and count != 2:
            # print(line)
            data = line.split('\t')
            data[len(data) - 1] = data[len(data) - 1].strip('\n')
            
            t_input = None
            t_label = None
            
            if data_pos.get('inputs') == 1:
                t_input = [data[data_pos.get('i_start')]]
            else:
                t_input = data[data_pos.get('i_start'):data_pos.get('i_end')]
                
            if data_pos.get('outputs') == 1:
                t_label = [data[len(data) - 1]]
            else:
                t_label = data[data_pos.get('o_start'):data_pos.get('o_end')]
                
            t_input = [int(i) for i in t_input]
            t_label = [int(i) for i in t_label]
            
            data_inputs.append(t_input)
            output_labels.append(t_label)
            
            # print(count, data)
            # print(count, t_input)
            # print(count, t_label)
    
    # print('Inputs:',len(data_inputs),data_inputs)
    # print('Outputs:',len(output_labels),output_labels)
    
    lengthData = len(data_inputs)
    i = data_pos.get('inputs')
    o = data_pos.get('outputs')
    
    return data_inputs, output_labels, lengthData, i, o

# data1,output1,lengthd = get_datafromfile('xor.txt')

# print(data1)
# print(output1)
# print(lengthd)

        