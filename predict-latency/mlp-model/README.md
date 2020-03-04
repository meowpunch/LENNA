# DATA INFO
# for original ver
X_data, X_train : torch.Size([# of data, 28]) + 정규화 및 표준화 (input node별로) <br />
Y_data, Y_train : torch.Size([# of data, 1])

# for 2D ver.
X_data, X_train : torch.Size([# of data, 7, 4]) + 정규화 및 표준화 (각 cfg 튜플에서 (expansion, out_planes, num_blocks, stride)별로) <br />
Y_data, Y_train : torch.Size([# of data, 1])
