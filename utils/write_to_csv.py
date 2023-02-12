import csv  
def write_scv(policy, module_name, weight_s, input_s, output_s, 
              f_read, f_write, f_flops, f_peakmem,
              b_read, b_write, b_flops, b_peakmem):
    header = ['Layer_index', 'Shape', 
            'Weight Size', 'Input Size','Output Size',
            'f_read','f_write', 'f_flops','f_peakmem',
            'b_read','b_write', 'b_flops','b_peakmem',
            ]
    data = []
    for row in zip(range(len(module_name)),module_name, 
                   weight_s, input_s, output_s, 
                    f_read, f_write, f_flops, f_peakmem,
                    b_read, b_write, b_flops, b_peakmem):
        row_convert = []
        for x in row:
            if isinstance(x, str):
                row_convert.append(x)
            else:
                row_convert.append(x)
                # row_convert.append('{:.2E}'.format(x))
        
        data.append([i for i in row_convert])

    with open('./table/'+policy+'.csv', 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        writer.writerows(data)

# import csv

# header = ['name', 'area', 'country_code2', 'country_code3']
# data = [
#     ['Albania', 28748, 'AL', 'ALB'],
#     ['Algeria', 2381741, 'DZ', 'DZA'],
#     ['American Samoa', 199, 'AS', 'ASM'],
#     ['Andorra', 468, 'AD', 'AND'],
#     ['Angola', 1246700, 'AO', 'AGO']
# ]

# with open('countries.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)

#     # write the header
#     writer.writerow(header)

#     # write multiple rows
#     writer.writerows(data)
