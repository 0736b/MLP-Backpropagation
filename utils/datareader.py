import math

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
        data_pos['inputs'] = 2
        data_pos['outputs'] = 2
        
    
    if filename != 'cross.pat':
        count = 0
        for line in lines:
            count += 1
            if count != 1 and count != 2:
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
    else:
        count = 0
        for line in lines:
            count += 1
            data = line.split(' ')
            data[len(data) - 1] = data[len(data) - 1].strip('\n')
            if len(data) == 2:
                data = [int(i) for i in data]
            if len(data) == 3:
                data.remove('')
                data = [float(i) for i in data]
            # print('len',len(data),data,'data type:',type(data[0]))
            if type(data[0]) == type(0.0):
                data_inputs.append(data)
            elif type(data[0]) == type(0):
                output_labels.append(data)
    
    lengthData = len(data_inputs)
    i = data_pos.get('inputs')
    o = data_pos.get('outputs')
    
    return data_inputs, output_labels, lengthData, i, o

# Getting Data Array, Mean, Standard deviation from text file
def get_datafornorm(filename):
    filepath = './/dataset/' + filename
    lines = []
    
    datas = []
    m = 0.0
    s = 0.0
        
    with open(filepath) as f:
        lines = f.readlines()
        
    count = 0
    for line in lines:
        count += 1
        if count != 1 and count != 2:
            data = line.split('\t')
            data[len(data) - 1] = data[len(data) - 1].strip('\n')
            for d in data:
                datas.append(float(d))

    m = mean(datas)
    s = std_d(datas)
    
    return datas, m, s
    
def mean(data):
  n = len(data)
  mean = sum(data) / n
  return mean

def variance(data):
  n = len(data)
  mean = sum(data) / n
  deviations = [(x - mean) ** 2 for x in data]
  variance = sum(deviations) / n
  return variance

def std_d(data):
    var = variance(data)
    std_dev = math.sqrt(var)
    return std_dev
    
# for Flood dataset    
def standardization(data, mean, standard_deviation):
    # y = (x - mean) / standard_deviation
    # x = (y * standard_deviation) + mean
    norm_data_list = []
    data_inputs = []
    output_labels = []
    norm_data = [None] * len(data)
    for i in range(0,len(data),1):
        t_data = (data[i] - mean) / standard_deviation
        norm_data[i] = t_data
        
    leng = len(norm_data) / 9
    leng = int(leng)    
    for i in range(0,leng, 1):
        start = i*9
        end = (i+1) * 9
        t_input_row = []
        for j in range(start,end,1):
            t_input_row.append(norm_data[j])
        norm_data_list.append(t_input_row)
        data_inputs.append(t_input_row[0:8])
        output_labels.append([t_input_row[8]])
    
    return norm_data_list, data_inputs, output_labels
        
def de_standardization(value,mean, standard_deviation):
    result = (value * standard_deviation) + mean
    return result    

def get_datarow(filename):
    filepath = './/dataset/' + filename
    data_row = []
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
        
    elif filename == 'cross.pat':
        data_pos['inputs'] = 2
        data_pos['outputs'] = 2
        
    if filename != 'cross.pat':
        count = 0
        for line in lines:
            count += 1
            if count != 1 and count != 2:
                data = line.split('\t')
                data[len(data) - 1] = data[len(data) - 1].strip('\n')
                data_row.append(data)
    else:
        t_data = []
        count = 0
        for line in lines:
            count += 1
            data = line.split(' ')
            data[len(data) - 1] = data[len(data) - 1].strip('\n')
            if len(data) == 2:
                data = [int(i) for i in data]
            if len(data) == 3:
                data.remove('')
                data = [float(i) for i in data]
            # print('len',len(data),data,'data type:',type(data[0]))
            if type(data[0]) == type(0.0):
                t_data = t_data + data
            elif type(data[0]) == type(0):
                t_data = t_data + data
            if(len(t_data) == 4):
                data_row.append(t_data)
                t_data = [] 
                
    return data_row
    
    
def get_inputoutputfromrow(row,filename):
    row_input = []
    row_output = []
    
    data_pos = {
        'i_start': 0,
        'i_end': 0,
        'o_start': 0,
        'o_end': 0,
        'inputs': 0,
        'outputs': 0
    }

    if filename == 'Flood_dataset.txt':
        data_pos['inputs'] = 8
        data_pos['outputs'] = 1
        data_pos['i_end'] = 8
        if data_pos.get('inputs') == 1:
            row_input = [row[data_pos.get('i_start')]]
        else:
            row_input = row[data_pos.get('i_start'):data_pos.get('i_end')]
                
        if data_pos.get('outputs') == 1:
            row_output = [row[len(row) - 1]]
        else:
            row_output = row[data_pos.get('o_start'):data_pos.get('o_end')]
                
        row_input = [float(i) for i in row_input]
        row_output = [float(i) for i in row_output]
        
        return row_input, row_output
        
    elif filename == 'cross.pat':
        data_pos['inputs'] = 2
        data_pos['outputs'] = 2
        length = len(row)
        mid_idx = length // 2
        row_input = row[:mid_idx]
        row_output = row[mid_idx:]
        
        row_input = [float(i) for i in row_input]
        row_output = [int(i) for i in row_output]
        
        return row_input, row_output