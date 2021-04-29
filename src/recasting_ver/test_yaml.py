import os

normal_op = [
           '3x3_Conv', '5x5_Conv',
           '3x3_DilatedConv', '5x5_DilatedConv',
           '3x3_DepthConv', '5x5_DepthConv',
           '3x3_DilatedDepthConv', '5x5_DilatedDepthConv',
           '3x3_MAXPool', '3x3_AVGPool',
         ]

reduction_op = [
              '3x3_Conv', '5x5_Conv',
              '3x3_DilatedConv', '5x5_DilatedConv',
              '3x3_DepthConv', '5x5_DepthConv',
              '3x3_DilatedDepthConv', '5x5_DilatedDepthConv',
              '2x2_MAXPoolExpand', '2x2_AVGPoolExpand',
            ]

normal_feature = [(32, 16, 16, 1), (16, 32, 32, 1), (8, 64, 64, 1)]
reduction_feature = [(32, 16, 32, 2), (16, 32, 64, 2)]

f = open('test.yaml', mode='wt')

for F, In, Out, S in normal_feature:
    for op in normal_op :
        fsize = str(F) + 'x' + str(F)
        infos = [op, 'feature:%s' % fsize, 'input:%d' % In, 'output:%d' % Out, 'stride:%d' % S]
        key = '-'.join(infos) + ':\n'
        f.write(key)
        f.write('    count: \n')
        f.write('    mean: \n')
        f.write('    var: \n')
for F, In, Out, S in reduction_feature:
    for op in reduction_op :
        fsize = str(F) + 'x' + str(F)
        infos = [op, 'feature:%s' % fsize, 'input:%d' % In, 'output:%d' % Out, 'stride:%d' % S]
        key = '-'.join(infos) + ':\n'
        f.write(key)
        f.write('    count: \n')
        f.write('    mean: \n')
        f.write('    var: \n')

f.close()
