outfile = 'result' # outfile name (change if needed!!!!)
out = open(outfile, 'w')
cfg_seen = set()

for line in open('training_data_1', 'r'): # put in your data file name. (change if needed!!!)
    split = line.split(',')
    if split[0] not in cfg_seen:
        out.write(line)
        cfg_seen.add(split[0])

out.close()
