import torch
import torch.nn as nn
import torch.nn.functional as F


#number of local pointnet input points, gathered via K nearest neighbors
KNN_num = 8


class pc_conv_first(nn.Module):
    def __init__(self, ef_dim):
        super(pc_conv_first, self).__init__()
        self.ef_dim = ef_dim
        self.linear_1 = nn.Linear(3, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)

    def forward(self, KNN_xyz):
        output = KNN_xyz
        #[newpointnum*KNN_num,3]
        
        output = self.linear_1(output)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.linear_2(output)
        #[newpointnum*KNN_num,ef_dim]
        
        output = output.view(-1,KNN_num,self.ef_dim) # Returns a new tensor with the same data as the self tensor but of a different shape.
        #[newpointnum,KNN_num,ef_dim]
        output = torch.max(output,1)[0] # 计算第1维的最大值，并取其中的values属性，[1] 是其indices属性， [0] 是其values属性;
        #[newpointnum,ef_dim]
        return output


class pc_conv(nn.Module):
    def __init__(self, ef_dim):
        super(pc_conv, self).__init__()
        self.ef_dim = ef_dim
        self.linear_1 = nn.Linear(self.ef_dim + 3, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)

    def forward(self, input, KNN_idx, KNN_xyz):
        output = input
        #[pointnum,ef_dim]
        output = output[KNN_idx]
        #[newpointnum*KNN_num,ef_dim]
        output = torch.cat([output, KNN_xyz], 1)
        #[newpointnum*KNN_num,ef_dim+3]
        
        output = self.linear_1(output)
        #[newpointnum*KNN_num,ef_dim]
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        #[newpointnum*KNN_num,ef_dim]
        output = self.linear_2(output)
        #[newpointnum*KNN_num,ef_dim]
        
        output = output.view(-1, KNN_num, self.ef_dim)
        #[newpointnum, KNN_num, ef_dim]
        output = torch.max(output, 1)[0] # 相当于pointnet++里面的pooling操作
        #[newpointnum, ef_dim]
        return output


class pc_resnet_block(nn.Module):
    def __init__(self, ef_dim):
        super(pc_resnet_block, self).__init__()
        self.ef_dim = ef_dim
        self.linear_1 = nn.Linear(self.ef_dim, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)

    def forward(self, input):
        output = self.linear_1(input) # 输出 [newpointnum, ef_dim]
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True) # negative_slope 控制“负斜率”的角度值，默认0.01
        output = self.linear_2(output) # 输出 [newpointnum, ef_dim]
        output = output+input # 输出 [newpointnum, ef_dim]
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        return output


class resnet_block(nn.Module):
    def __init__(self, ef_dim):
        super(resnet_block, self).__init__()
        self.ef_dim = ef_dim
        self.pc_conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.pc_conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = self.pc_conv_1(input)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.pc_conv_2(output)
        output = output+input
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        return output


class resnet_block_rec3(nn.Module):
    def __init__(self, ef_dim):
        super(resnet_block_rec3, self).__init__()
        self.ef_dim = ef_dim
        # in_channels, out_channels, kernel_size
        self.pc_conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.pc_conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)

    def forward(self, input): # [1, 128, Sx, Sy, Sz] 整个接口的输入和输出维度相同
        output = self.pc_conv_1(input)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        # 输出 [1, 128, Sx, Sy, Sz]
        output = self.pc_conv_2(output)
        output = output+input # 将 input 和 output 直接加在一块
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        return output


class local_pointnet(nn.Module):

    def __init__(self, out_bool, out_float):
        super(local_pointnet, self).__init__()
        self.ef_dim = 128
        self.out_bool = out_bool
        self.out_float = out_float
        
        self.pc_conv_0 = pc_conv_first(self.ef_dim)

        self.pc_res_1 = pc_resnet_block(self.ef_dim)
        self.pc_conv_1 = pc_conv(self.ef_dim)

        self.pc_res_2 = pc_resnet_block(self.ef_dim)
        self.pc_conv_2 = pc_conv(self.ef_dim)

        self.pc_res_3 = pc_resnet_block(self.ef_dim)
        self.pc_conv_3 = pc_conv(self.ef_dim)

        self.pc_res_4 = pc_resnet_block(self.ef_dim)
        self.pc_conv_4 = pc_conv(self.ef_dim)

        self.pc_res_5 = pc_resnet_block(self.ef_dim)
        self.pc_conv_5 = pc_conv(self.ef_dim)

        self.pc_res_6 = pc_resnet_block(self.ef_dim)
        self.pc_conv_6 = pc_conv(self.ef_dim)

        self.pc_res_7 = pc_resnet_block(self.ef_dim)

        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)

        self.conv_4 = nn.Linear(self.ef_dim, self.ef_dim)
        self.conv_5 = nn.Linear(self.ef_dim, self.ef_dim)

        if self.out_bool:
            self.pc_conv_out_bool = nn.Linear(self.ef_dim, 3)
        if self.out_float:
            self.pc_conv_out_float = nn.Linear(self.ef_dim, 3)

    def forward(self, pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz):
        out = pc_KNN_xyz
        
        out = self.pc_conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_1(out)
        out = self.pc_conv_1(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_2(out)
        out = self.pc_conv_2(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_3(out)
        out = self.pc_conv_3(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_4(out)
        out = self.pc_conv_4(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_5(out)
        out = self.pc_conv_5(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_6(out)
        out = self.pc_conv_6(out, voxel_KNN_idx, voxel_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_7(out)

        voxel_xyz_int_max = torch.max(voxel_xyz_int,0)[0]
        voxel_xyz_int_min = torch.min(voxel_xyz_int,0)[0]
        voxel_xyz_int_size = voxel_xyz_int_max-voxel_xyz_int_min+1
        voxel_xyz_int = voxel_xyz_int-voxel_xyz_int_min.view(1,-1)
        tmp_grid = torch.zeros(voxel_xyz_int_size[0],voxel_xyz_int_size[1],voxel_xyz_int_size[2],self.ef_dim, device=out.device)
        tmp_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = out
        tmp_grid = tmp_grid.permute(3,0,1,2).unsqueeze(0)
        out = tmp_grid


        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_3(out)
        
        out = out.squeeze(0).permute(1,2,3,0)
        out = out[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]]

        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)


        if self.out_bool and self.out_float:
            out_bool = self.pc_conv_out_bool(out)
            out_float = self.pc_conv_out_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.pc_conv_out_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.pc_conv_out_float(out)
            return out_float


class local_pointnet_larger(nn.Module):

    def __init__(self, out_bool, out_float):
        super(local_pointnet_larger, self).__init__()
        self.ef_dim = 128
        self.out_bool = out_bool
        self.out_float = out_float
        
        self.pc_conv_0 = pc_conv_first(self.ef_dim)

        self.pc_res_1 = pc_resnet_block(self.ef_dim)
        self.pc_conv_1 = pc_conv(self.ef_dim)

        self.pc_res_2 = pc_resnet_block(self.ef_dim)
        self.pc_conv_2 = pc_conv(self.ef_dim)

        self.pc_res_3 = pc_resnet_block(self.ef_dim)
        self.pc_conv_3 = pc_conv(self.ef_dim)

        self.pc_res_4 = pc_resnet_block(self.ef_dim)
        self.pc_conv_4 = pc_conv(self.ef_dim)

        self.pc_res_5 = pc_resnet_block(self.ef_dim)
        self.pc_conv_5 = pc_conv(self.ef_dim)

        self.pc_res_6 = pc_resnet_block(self.ef_dim)
        self.pc_conv_6 = pc_conv(self.ef_dim)

        self.pc_res_7 = pc_resnet_block(self.ef_dim)

        self.res_1 = resnet_block_rec3(self.ef_dim)
        self.res_2 = resnet_block_rec3(self.ef_dim)
        self.res_3 = resnet_block_rec3(self.ef_dim)
        self.res_4 = resnet_block_rec3(self.ef_dim)
        self.res_5 = resnet_block_rec3(self.ef_dim)
        self.res_6 = resnet_block_rec3(self.ef_dim)
        self.res_7 = resnet_block_rec3(self.ef_dim)
        self.res_8 = resnet_block_rec3(self.ef_dim)

        self.linear_1 = nn.Linear(self.ef_dim, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)

        if self.out_bool:
            self.linear_bool = nn.Linear(self.ef_dim, 3)
        if self.out_float:
            self.linear_float = nn.Linear(self.ef_dim, 3)

    def forward(self, pc_KNN_idx, pc_KNN_xyz, voxel_xyz_int, voxel_KNN_idx, voxel_KNN_xyz):
        out = pc_KNN_xyz
        
        out = self.pc_conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_1(out)
        out = self.pc_conv_1(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_2(out)
        out = self.pc_conv_2(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_3(out)
        out = self.pc_conv_3(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_4(out)
        out = self.pc_conv_4(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_5(out)
        out = self.pc_conv_5(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_6(out)
        out = self.pc_conv_6(out, voxel_KNN_idx, voxel_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_7(out)

        voxel_xyz_int_max = torch.max(voxel_xyz_int, 0)[0] # 找 voxel points 中的三轴坐标各自的最大值
        voxel_xyz_int_min = torch.min(voxel_xyz_int,0)[0] # 找 voxel points 中的三轴坐标各自的最小值
        voxel_xyz_int_size = voxel_xyz_int_max - voxel_xyz_int_min+1 # voxel points 中的三轴坐标各自的size
        voxel_xyz_int = voxel_xyz_int - voxel_xyz_int_min.view(1,-1) # 将整体包围框的最小坐标挪到[0,0,0]位置
        tmp_grid = torch.zeros(voxel_xyz_int_size[0], voxel_xyz_int_size[1], voxel_xyz_int_size[2], self.ef_dim, device=out.device)
        tmp_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = out # 将前面网络提取到的特征放置到栅格中去
        tmp_grid = tmp_grid.permute(3,0,1,2).unsqueeze(0) # 先把128特征维度放在最前面，然后再加一个维度，再放在最前面
        out = tmp_grid

        out = self.res_1(out)
        out = self.res_2(out)
        out = self.res_3(out)
        out = self.res_4(out)
        out = self.res_5(out)
        out = self.res_6(out)
        out = self.res_7(out)
        out = self.res_8(out)
        
        out = out.squeeze(0).permute(1,2,3,0) # 把最前面的第0维度取消掉，然后把128的特征维度放在最后面，即恢复原来的布局
        out = out[voxel_xyz_int[:,0], voxel_xyz_int[:,1], voxel_xyz_int[:,2]]

        out = self.linear_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.linear_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)


        if self.out_bool and self.out_float:
            out_bool = self.linear_bool(out)
            out_float = self.linear_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.linear_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.linear_float(out)
            return out_float





class local_pointnet_larger_ifnet(nn.Module):
    
    def __init__(self, out_bool, out_float):
        super(local_pointnet_larger_ifnet, self).__init__()
        
        
        # 128**3 res input
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='border')
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='border')
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='border')
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='border')
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='border')
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='border')
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 ) * 7
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()
        # self.displacments = torch.Tensor(displacments).cuda(torch.device("cuda:0"))
        
        
        
        
        if self.out_bool:
            self.linear_bool = nn.Linear(self.ef_dim, 3)
        if self.out_float:
            self.linear_float = nn.Linear(self.ef_dim, 3)
        
        
        #====================== raw ===========================
        self.ef_dim = 128
        self.out_bool = out_bool
        self.out_float = out_float
        
        self.pc_conv_0 = pc_conv_first(self.ef_dim)

        self.pc_res_1 = pc_resnet_block(self.ef_dim)
        self.pc_conv_1 = pc_conv(self.ef_dim)

        self.pc_res_2 = pc_resnet_block(self.ef_dim)
        self.pc_conv_2 = pc_conv(self.ef_dim)

        self.pc_res_3 = pc_resnet_block(self.ef_dim)
        self.pc_conv_3 = pc_conv(self.ef_dim)

        self.pc_res_4 = pc_resnet_block(self.ef_dim)
        self.pc_conv_4 = pc_conv(self.ef_dim)

        self.pc_res_5 = pc_resnet_block(self.ef_dim)
        self.pc_conv_5 = pc_conv(self.ef_dim)

        self.pc_res_6 = pc_resnet_block(self.ef_dim)
        self.pc_conv_6 = pc_conv(self.ef_dim)

        self.pc_res_7 = pc_resnet_block(self.ef_dim)

        self.res_1 = resnet_block_rec3(self.ef_dim)
        self.res_2 = resnet_block_rec3(self.ef_dim)
        self.res_3 = resnet_block_rec3(self.ef_dim)
        self.res_4 = resnet_block_rec3(self.ef_dim)
        self.res_5 = resnet_block_rec3(self.ef_dim)
        self.res_6 = resnet_block_rec3(self.ef_dim)
        self.res_7 = resnet_block_rec3(self.ef_dim)
        self.res_8 = resnet_block_rec3(self.ef_dim)

        self.linear_1 = nn.Linear(self.ef_dim, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)

        if self.out_bool:
            self.linear_bool = nn.Linear(self.ef_dim, 3)
        if self.out_float:
            self.linear_float = nn.Linear(self.ef_dim, 3)

    def forward(self, pc_KNN_idx, pc_KNN_xyz, voxel_xyz_int, voxel_KNN_idx, voxel_KNN_xyz):
        
        x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
        feature_0 = F.grid_sample(x, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, padding_mode='border')

        # here every channel corresponds to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)
        
        
        
        if self.out_bool:
            out_bool = self.linear_bool(out) # out: [Nvp, 128], out_bool: [Nvp, 3]
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.linear_float(out) # out: [Nvp, 128], out_float: [Nvp, 3]
            return out_float
        
        #======================= raw ==================================
        out = pc_KNN_xyz
        
        out = self.pc_conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_1(out)
        out = self.pc_conv_1(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_2(out)
        out = self.pc_conv_2(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_3(out)
        out = self.pc_conv_3(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_4(out)
        out = self.pc_conv_4(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_5(out)
        out = self.pc_conv_5(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_6(out)
        out = self.pc_conv_6(out, voxel_KNN_idx, voxel_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_7(out)

        voxel_xyz_int_max = torch.max(voxel_xyz_int, 0)[0] # 找 voxel points 中的三轴坐标各自的最大值
        voxel_xyz_int_min = torch.min(voxel_xyz_int,0)[0] # 找 voxel points 中的三轴坐标各自的最小值
        voxel_xyz_int_size = voxel_xyz_int_max - voxel_xyz_int_min+1 # voxel points 中的三轴坐标各自的size
        voxel_xyz_int = voxel_xyz_int - voxel_xyz_int_min.view(1,-1) # 将整体包围框的最小坐标挪到[0,0,0]位置
        tmp_grid = torch.zeros(voxel_xyz_int_size[0], voxel_xyz_int_size[1], voxel_xyz_int_size[2], self.ef_dim, device=out.device)
        tmp_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = out # 将前面网络提取到的特征放置到栅格中去
        tmp_grid = tmp_grid.permute(3,0,1,2).unsqueeze(0) # 先把128特征维度放在最前面，然后再加一个维度，再放在最前面
        out = tmp_grid

        out = self.res_1(out)
        out = self.res_2(out)
        out = self.res_3(out)
        out = self.res_4(out)
        out = self.res_5(out)
        out = self.res_6(out)
        out = self.res_7(out)
        out = self.res_8(out)
        
        out = out.squeeze(0).permute(1,2,3,0) # 把最前面的第0维度取消掉，然后把128的特征维度放在最后面，即恢复原来的布局
        out = out[voxel_xyz_int[:,0], voxel_xyz_int[:,1], voxel_xyz_int[:,2]]

        out = self.linear_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.linear_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)


        if self.out_bool and self.out_float:
            out_bool = self.linear_bool(out)
            out_float = self.linear_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.linear_bool(out)
            return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.linear_float(out)
            return out_float


# 后处理，用于弥补小洞
def postprocessing(pred_output_bool):
    for t in range(2):

        #open edges
        gridedge_x_outedge_y_0 = pred_output_bool[:-1, :,   :-1, 0]
        gridedge_x_outedge_y_1 = pred_output_bool[:-1, :,   1: , 0]
        gridedge_x_outedge_z_0 = pred_output_bool[:-1, :-1, :  , 0]
        gridedge_x_outedge_z_1 = pred_output_bool[:-1, 1:,  :  , 0]
        gridedge_y_outedge_x_0 = pred_output_bool[:,   :-1, :-1, 1]
        gridedge_y_outedge_x_1 = pred_output_bool[:,   :-1, 1: , 1]
        gridedge_y_outedge_z_0 = pred_output_bool[:-1, :-1, :  , 1]
        gridedge_y_outedge_z_1 = pred_output_bool[1:,  :-1, :  , 1]
        gridedge_z_outedge_x_0 = pred_output_bool[:,   :-1, :-1, 2]
        gridedge_z_outedge_x_1 = pred_output_bool[:,   1:,  :-1, 2]
        gridedge_z_outedge_y_0 = pred_output_bool[:-1, :,   :-1, 2]
        gridedge_z_outedge_y_1 = pred_output_bool[1:,  :,   :-1, 2]
        outedge_x = gridedge_y_outedge_x_0+gridedge_y_outedge_x_1+gridedge_z_outedge_x_0+gridedge_z_outedge_x_1
        outedge_y = gridedge_x_outedge_y_0+gridedge_x_outedge_y_1+gridedge_z_outedge_y_0+gridedge_z_outedge_y_1
        outedge_z = gridedge_x_outedge_z_0+gridedge_x_outedge_z_1+gridedge_y_outedge_z_0+gridedge_y_outedge_z_1
        boundary_x_flag = (outedge_x==1).int()
        boundary_y_flag = (outedge_y==1).int()
        boundary_z_flag = (outedge_z==1).int()

        tmp_int = torch.zeros(pred_output_bool.size(), dtype=torch.int32, device=pred_output_bool.device)
        tmp_int[:,    :-1, :-1, 1] += boundary_x_flag
        tmp_int[:,    :-1, 1: , 1] += boundary_x_flag
        tmp_int[:,    :-1, :-1, 2] += boundary_x_flag
        tmp_int[:,    1:,  :-1, 2] += boundary_x_flag
        tmp_int[:-1, :,    :-1, 0] += boundary_y_flag
        tmp_int[:-1, :,    1: , 0] += boundary_y_flag
        tmp_int[:-1, :,    :-1, 2] += boundary_y_flag
        tmp_int[1:,  :,    :-1, 2] += boundary_y_flag
        tmp_int[:-1, :-1, :   , 0] += boundary_z_flag
        tmp_int[:-1, 1:,  :   , 0] += boundary_z_flag
        tmp_int[:-1, :-1, :   , 1] += boundary_z_flag
        tmp_int[1:,  :-1, :   , 1] += boundary_z_flag

        #create a quad if meet 3 open edges
        pred_output_bool = torch.max( pred_output_bool, (tmp_int>=3).int() )

        #delete a quad if meet 3 open edges
        pred_output_bool = torch.min( pred_output_bool, (tmp_int<3).int() )


    for t in range(1): #radical

        #open edges
        gridedge_x_outedge_y_0 = pred_output_bool[:-1, :,   :-1, 0]
        gridedge_x_outedge_y_1 = pred_output_bool[:-1, :,   1: , 0]
        gridedge_x_outedge_z_0 = pred_output_bool[:-1, :-1, :  , 0]
        gridedge_x_outedge_z_1 = pred_output_bool[:-1, 1:,  :  , 0]
        gridedge_y_outedge_x_0 = pred_output_bool[:,   :-1, :-1, 1]
        gridedge_y_outedge_x_1 = pred_output_bool[:,   :-1, 1: , 1]
        gridedge_y_outedge_z_0 = pred_output_bool[:-1, :-1, :  , 1]
        gridedge_y_outedge_z_1 = pred_output_bool[1:,  :-1, :  , 1]
        gridedge_z_outedge_x_0 = pred_output_bool[:,   :-1, :-1, 2]
        gridedge_z_outedge_x_1 = pred_output_bool[:,   1:,  :-1, 2]
        gridedge_z_outedge_y_0 = pred_output_bool[:-1, :,   :-1, 2]
        gridedge_z_outedge_y_1 = pred_output_bool[1:,  :,   :-1, 2]
        outedge_x = gridedge_y_outedge_x_0+gridedge_y_outedge_x_1+gridedge_z_outedge_x_0+gridedge_z_outedge_x_1
        outedge_y = gridedge_x_outedge_y_0+gridedge_x_outedge_y_1+gridedge_z_outedge_y_0+gridedge_z_outedge_y_1
        outedge_z = gridedge_x_outedge_z_0+gridedge_x_outedge_z_1+gridedge_y_outedge_z_0+gridedge_y_outedge_z_1
        boundary_x_flag = (outedge_x==1).int()
        boundary_y_flag = (outedge_y==1).int()
        boundary_z_flag = (outedge_z==1).int()

        tmp_int = torch.zeros(pred_output_bool.size(), dtype=torch.int32, device=pred_output_bool.device)
        tmp_int[:,    :-1, :-1, 1] += boundary_x_flag
        tmp_int[:,    :-1, 1: , 1] += boundary_x_flag
        tmp_int[:,    :-1, :-1, 2] += boundary_x_flag
        tmp_int[:,    1:,  :-1, 2] += boundary_x_flag
        tmp_int[:-1, :,    :-1, 0] += boundary_y_flag
        tmp_int[:-1, :,    1: , 0] += boundary_y_flag
        tmp_int[:-1, :,    :-1, 2] += boundary_y_flag
        tmp_int[1:,  :,    :-1, 2] += boundary_y_flag
        tmp_int[:-1, :-1, :   , 0] += boundary_z_flag
        tmp_int[:-1, 1:,  :   , 0] += boundary_z_flag
        tmp_int[:-1, :-1, :   , 1] += boundary_z_flag
        tmp_int[1:,  :-1, :   , 1] += boundary_z_flag

        #create a quad if meet 2 open edges, only if it helps close a hole, see below code
        pred_output_bool_backup = pred_output_bool
        pred_output_bool = torch.max( pred_output_bool, (tmp_int>=2).int() )

        #open edges
        gridedge_x_outedge_y_0 = pred_output_bool[:-1, :,   :-1, 0]
        gridedge_x_outedge_y_1 = pred_output_bool[:-1, :,   1: , 0]
        gridedge_x_outedge_z_0 = pred_output_bool[:-1, :-1, :  , 0]
        gridedge_x_outedge_z_1 = pred_output_bool[:-1, 1:,  :  , 0]
        gridedge_y_outedge_x_0 = pred_output_bool[:,   :-1, :-1, 1]
        gridedge_y_outedge_x_1 = pred_output_bool[:,   :-1, 1: , 1]
        gridedge_y_outedge_z_0 = pred_output_bool[:-1, :-1, :  , 1]
        gridedge_y_outedge_z_1 = pred_output_bool[1:,  :-1, :  , 1]
        gridedge_z_outedge_x_0 = pred_output_bool[:,   :-1, :-1, 2]
        gridedge_z_outedge_x_1 = pred_output_bool[:,   1:,  :-1, 2]
        gridedge_z_outedge_y_0 = pred_output_bool[:-1, :,   :-1, 2]
        gridedge_z_outedge_y_1 = pred_output_bool[1:,  :,   :-1, 2]
        outedge_x = gridedge_y_outedge_x_0+gridedge_y_outedge_x_1+gridedge_z_outedge_x_0+gridedge_z_outedge_x_1
        outedge_y = gridedge_x_outedge_y_0+gridedge_x_outedge_y_1+gridedge_z_outedge_y_0+gridedge_z_outedge_y_1
        outedge_z = gridedge_x_outedge_z_0+gridedge_x_outedge_z_1+gridedge_y_outedge_z_0+gridedge_y_outedge_z_1
        boundary_x_flag = (outedge_x==1).int()
        boundary_y_flag = (outedge_y==1).int()
        boundary_z_flag = (outedge_z==1).int()

        tmp_int = torch.zeros(pred_output_bool.size(), dtype=torch.int32, device=pred_output_bool.device)
        tmp_int[:,    :-1, :-1, 1] += boundary_x_flag
        tmp_int[:,    :-1, 1: , 1] += boundary_x_flag
        tmp_int[:,    :-1, :-1, 2] += boundary_x_flag
        tmp_int[:,    1:,  :-1, 2] += boundary_x_flag
        tmp_int[:-1, :,    :-1, 0] += boundary_y_flag
        tmp_int[:-1, :,    1: , 0] += boundary_y_flag
        tmp_int[:-1, :,    :-1, 2] += boundary_y_flag
        tmp_int[1:,  :,    :-1, 2] += boundary_y_flag
        tmp_int[:-1, :-1, :   , 0] += boundary_z_flag
        tmp_int[:-1, 1:,  :   , 0] += boundary_z_flag
        tmp_int[:-1, :-1, :   , 1] += boundary_z_flag
        tmp_int[1:,  :-1, :   , 1] += boundary_z_flag

        pred_output_bool = torch.min( pred_output_bool, (tmp_int<2).int() )
        pred_output_bool = torch.max( pred_output_bool, pred_output_bool_backup )

    return pred_output_bool



