import torch
import numpy as np
import torch_geometric.data.collate

from .trajdata import TrajData


class ETHNew(object):
    def __init__(self, dataset, past_frames, future_frames, traj_scale, phase, return_index=False):
        # file_dir = 'eth_ucy/processed_data_diverse/'
        file_dir = f'eth_ucy/processed_data_diverse/'
        if phase == 'training':
            data_file_path = file_dir + dataset +'_data_train.npy'
            num_file_path = file_dir + dataset +'_num_train.npy'
        elif phase == 'testing':
            data_file_path = file_dir + dataset +'_data_test.npy'
            num_file_path = file_dir + dataset +'_num_test.npy'
        all_data = np.load(data_file_path)
        all_num = np.load(num_file_path)
        self.all_data = torch.Tensor(all_data)
        self.all_num = torch.Tensor(all_num)
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.traj_scale = traj_scale
        self.return_index = return_index

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, item):
        all_seq = self.all_data[item] / self.traj_scale  # [N, T, D]
        num = self.all_num[item].long()

        # 按有效节点数量 num 截取，结果形状为 [num, 20, 2]
        all_seq = all_seq[:num]
        x = all_seq.permute(0, 2, 1)  # [N, D, T]

        # 前 8 个时间片（历史部分）
        x_past = x[..., :self.past_frames]  # [N, D, 8]
        v_in = torch.zeros_like(x_past)  # [N, D, 8]
        v_in[..., 1:] = x_past[..., 1:] - x_past[..., :-1]
        v_in[..., 0] = v_in[..., 1]  # 填充初始速度

        # 后 12 个时间片（未来部分）
        x_future = x[..., self.past_frames:self.past_frames + self.future_frames]  # [N, D, 12]

        # h 构造
        # Velocity as h
        h = torch.norm(v_in, p=2, dim=1, keepdim=True)  # [N, 1, T]
        # All one as h
        # h = torch.ones(x.size(0), 1, x.size(-1))  # [N, 1, T]
        h_past = h[..., :self.past_frames]  # [N, 8]
        h_future = h[..., self.past_frames:self.past_frames + self.future_frames]  # [N, 12]

        # h = torch.zeros(x.size(0), 2, x.size(-1))  # [N, 2, T]
        # h[:, 0, :self.past_frames] = 1  # 第 0 通道标记历史部分
        # v_in_norm = torch.norm(v_in, p=2, dim=1, keepdim=True)  # [N, 1, 8]
        # h[:, 1:, :self.past_frames] = v_in_norm
        # h[:, 1:, self.past_frames:] = v_in_norm[..., -1:].repeat(1, 1, self.future_frames)  # 将最后一个速度模长延展到未来部分

        # 构造 edge_index
        num_nodes = x.size(0)
        row = torch.arange(num_nodes, dtype=torch.long)
        col = torch.arange(num_nodes, dtype=torch.long)
        row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
        col = col.repeat(num_nodes)
        edge_index = torch.stack([row, col], dim=0)  # [2, n_edges]

        # 构造 edge_attr
        edge_attr = torch.ones(edge_index.size(1), 1)  # [n_edges, 1]

        # 调整维度
        h = h.permute(0, 2, 1)  # [N, T, 1]
        x_past = x_past.permute(0, 2, 1)  # [N, 8, D]
        x_future = x_future.permute(0, 2, 1)  # [N, 12, D]

        # 构造 TrajData
        data = TrajData(h=h, x_past=x_past, x_future=x_future, edge_index=edge_index, edge_attr=edge_attr, v=v_in, num_nodes=num_nodes)

        # 附加其他信息
        select_index = torch.zeros(1)
        data['select_index'] = select_index.long()
        data['num'] = num

        if self.return_index:
            data['system_id'] = torch.ones(1) * item

        return data




if __name__ == '__main__':
    # 假设你希望查看前 5 个数据项
    num_data_items_to_print = 5

    # 初始化数据集
    dataset = ETHNew(dataset='univ', past_frames=8, future_frames=12, traj_scale=1., phase='testing', return_index=True)

    print(f"Total number of data items: {len(dataset)}")

    # 打印前 5 个数据项
    for i in range(min(num_data_items_to_print, len(dataset))):
        data = dataset[i]

        print(f"Data {i}:")
        print("Edge Index:")
        print(data['edge_index'])  # 打印当前数据的 edge_index
        print("h shape:", data['h'].shape)
        print("x_past shape:", data['x_past'].shape)
        print("x_future shape:", data['x_future'].shape)
        print("v shape:", data['v'].shape)
        print("num:", data['num'])
        print("select_index:", data['select_index'])
        print("edge attr:", data['edge_attr'])
        print("=" * 50)

    # 使用 collate 将多个数据项合并为一个批次
    data_list = [dataset[i] for i in range(0, num_data_items_to_print)]
    temp = torch_geometric.data.collate.collate(cls=dataset[0].__class__, data_list=data_list)[0]

    # 打印合并后的数据
    print("Collated Data:")
    print(f"Edge Index (batch version): {temp.edge_index.shape}")
    print(f"Batch Size: {temp.batch.shape}")
    print(f"Number of Nodes: {temp.num_nodes}")
    print(f"Edge Attribute (batch version): {temp.edge_attr.shape}")
    print(f"h shape (batch version): {temp.h.shape}")
    print(f"x_past shape (batch version): {temp.x_past.shape}")
    print(f"x_future shape (batch version): {temp.x_future.shape}")



# import torch
# import numpy as np
# import torch_geometric.data.collate
#
# from trajdata import TrajData
#
# class ETHNew(object):
#     def __init__(self, dataset, past_frames, future_frames, traj_scale, phase, return_index=False):
#         file_dir = f'data/{dataset}/'
#         if phase == 'training':
#             data_file_path = file_dir + dataset + '_data_train.npy'
#             num_file_path = file_dir + dataset + '_num_train.npy'
#         elif phase == 'testing':
#             data_file_path = file_dir + dataset + '_data_test.npy'
#             num_file_path = file_dir + dataset + '_num_test.npy'
#         all_data = np.load(data_file_path)
#         all_num = np.load(num_file_path)
#         self.all_data = torch.Tensor(all_data)
#         self.all_num = torch.Tensor(all_num)
#         self.past_frames = past_frames
#         self.future_frames = future_frames
#         self.traj_scale = traj_scale
#         self.return_index = return_index
#
#     def __len__(self):
#         return self.all_data.shape[0]
#
#     def __getitem__(self, item):
#         all_seq = self.all_data[item] / self.traj_scale  # [N, T, D]
#         num = self.all_num[item].long()
#
#         # Select valid nodes based on num
#         all_seq = all_seq[:num]
#         x = all_seq.permute(0, 2, 1)  # [N, D, T]
#
#         # Split into past and future parts
#         x_past = x[..., :self.past_frames]  # [N, D, T_past]
#         x_future = x[..., self.past_frames:self.past_frames + self.future_frames]  # [N, D, T_future]
#
#         # Compute velocity for the past frames
#         v_in = torch.zeros_like(x_past)
#         v_in[..., 1:] = x_past[..., 1:] - x_past[..., :-1]
#         v_in[..., 0] = v_in[..., 1]  # [N, D, T_past]
#
#         # Construct h
#         h = torch.zeros(x.size(0), 2, self.past_frames + self.future_frames)
#         h[:, 0, :self.past_frames] = 1
#         v_in_norm = torch.norm(v_in, p=2, dim=1, keepdim=True)
#         h[:, 1:, :self.past_frames] = v_in_norm
#         h[:, 1:, self.past_frames:] = v_in_norm[..., -1:].repeat(1, 1, self.future_frames)
#
#         # Construct edge index for a fully connected graph
#         num_nodes = x.size(0)
#         row = torch.arange(num_nodes, dtype=torch.long)
#         col = torch.arange(num_nodes, dtype=torch.long)
#         row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
#         col = col.repeat(num_nodes)
#         edge_index = torch.stack([row, col], dim=0)  # [2, num_nodes^2]
#
#         # Edge attributes (all set to 1)
#         edge_attr = torch.ones(edge_index.size(1), 1)
#
#         # Create data object
#         data = TrajData(h=h, x_past=x_past, x_future=x_future, edge_index=edge_index, edge_attr=edge_attr, v=v_in)
#
#         # Additional metadata
#         data['num'] = num
#         if self.return_index:
#             data['system_id'] = torch.ones(1) * item
#
#         return data
#
# if __name__ == '__main__':
#     # Number of data items to inspect
#     num_data_items_to_print = 5
#
#     # Initialize dataset
#     dataset = ETHNew(dataset='zara1_main.sh', past_frames=8, future_frames=12, traj_scale=1., phase='training', return_index=True)
#
#     print(f"Total number of data items: {len(dataset)}")
#
#     # Print the first few data items
#     for i in range(min(num_data_items_to_print, len(dataset))):
#         data = dataset[i]
#
#         print(f"Data {i}:")
#         print("Edge Index:")
#         print(data['edge_index'])  # Print edge index
#         print("Edge Attr:", data['edge_attr'].shape)  # Print edge attributes
#         print("h:", data['h'].shape)
#         print("x_past:", data['x_past'].shape)
#         print("x_future:", data['x_future'].shape)
#         print("v:", data['v'].shape)
#         print("num:", data['num'])
#         print("=" * 50)
#
#     # Use collate to batch data
#     data_list = [dataset[i] for i in range(num_data_items_to_print)]
#     temp = torch_geometric.data.collate.collate(cls=dataset[0].__class__, data_list=data_list)[0]
#
#     print("Collated Data:")
#     print(f"Edge Index (batch version): {temp.edge_index.shape}")
#     print(f"Edge Attr (batch version): {temp.edge_attr.shape}")
#     print(f"Batch Size: {temp.batch.shape}")
#     print(f"Number of Nodes: {temp.num_nodes}")


