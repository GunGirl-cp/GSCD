from builtins import AttributeError, TypeError, hasattr, len, list # type: ignore
import copy
from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
from scipy.spatial.transform import Rotation as R


# def normalize(x):
#     return x / np.linalg.norm(x)

# def viewmatrix(z, up, pos):
#     vec2 = normalize(z)
#     vec1_avg = up
#     vec0 = normalize(np.cross(vec1_avg, vec2))
#     vec1 = normalize(np.cross(vec2, vec0))
#     m = np.stack([vec0, vec1, vec2, pos], 1)
#     #            up X z  2X0    z  
#     return m


# def quaternion_average(quaternions):
#     """计算四元数的平均值"""
#     # 将四元数添加到一个矩阵中
#     q_matrix = np.array(quaternions)
#     # 计算平均四元数
#     q_avg = np.mean(q_matrix, axis=0)
#     q_avg_normalized = normalize(q_avg)
#     return q_avg_normalized

# class FourDGSdataset(Dataset):
#     def __init__(
#         self,
#         dataset,
#         args,
#         dataset_type
#     ):
#         self.dataset = dataset
#         self.args = args
#         self.dataset_type=dataset_type
#         # dataset_path='/home/hello/lpf/4DGaussians/data/cvpr'
#         # self.w2c_matrices,self.focal_length = load_llff_data(dataset_path)
#         self.get_aver()
#         self.new_views()

#     def get_aver(self):
#         Rs = []  # 旋转矩阵集合
#         Ts = []  # 平移向量集合
#         for caminfo in self.dataset:
#             try:
#                Rs.append(caminfo.R)
#                Ts.append(caminfo.T)
#             except AttributeError:
#                continue
#     # 将旋转矩阵转换为四元数
#         quaternions = [R.from_matrix(r).as_quat() for r in Rs]
#     # 计算四元数的平均值
#         avg_quat = quaternion_average(quaternions)
#     # 将平均四元数转换回旋转矩阵
#         avg_R = R.from_quat(avg_quat).as_matrix()
#     # 计算所有相机位置的平均值
#         avg_T = np.mean(Ts, axis=0)
#     # 构建合成的c2w变换矩阵
#         c2w = np.eye(4)
#         c2w[:3, :3] = avg_R
#         c2w[:3, 3] = avg_T

#         up = np.mean([R[:, 1] for R in Rs], axis=0)
#         up = normalize(up)  # 归一化 'up' 向量

#     # 计算 'rads'
#         center = np.mean(Ts, axis=0)  # 计算所有相机位置的平均值
#         distances = np.linalg.norm(Ts - center, axis=1)
# # 计算这些距离的90%百分位数作为rads
#         rads = np.percentile(distances, 90)
#         return c2w,up ,rads
       
#     def new_views(self):
#         render_poses = []
#         c2w,up ,rads=self.get_aver()
#         #rads = np.array(list(rads) + [1.])
#         N_views = 120
#         rots = 2
#         zrate=.5
#         caminfo = self.dataset[0]
#         width = caminfo.width
#         FovX = caminfo.FovX
#         focal_length = width / (2 * np.tan(FovX / 2))
#         for theta in np.linspace(0., 2. * np.pi * rots, N_views+1)[:-1]:
#              c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
#              z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal_length, 1.])))
#              view_matrix = viewmatrix(z, up, c)
#              R = view_matrix[:3, :3]
#              T = c  # 直接使用c作为平移向量
        
#              render_poses.append((R, T))
#         return render_poses




#     def __getitem__(self, index):
#         # breakpoint()

#         # R, T = self.w2c_matrices[index]  # 这里假设w2c已经是(R, T)的形式
#         # caminfo = self.dataset[index]
#         # image = caminfo.image
#         # # 计算FovX和FovY
#         # FovX = caminfo.FovX
#         # FovY = caminfo.FovY
#         # # FovX = focal2fov(self.focal_length, image.shape[2])
#         # # FovY = focal2fov(self.focal_length, image.shape[1])
        
#         # # 返回相机实例
#         # return Camera(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=None, image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=None, mask=None)

        
#         if self.dataset_type != "PanopticSports":
#             try:
#                 image, w2c, time = self.dataset[index]
#                 R,T = w2c
#                 FovX = focal2fov(self.dataset.focal[0], image.shape[2])
#                 FovY = focal2fov(self.dataset.focal[0], image.shape[1])
#                 mask=None
#             except:
#                 caminfo = self.dataset[index]
#                 image = caminfo.image
#                 R = caminfo.R
#                 T = caminfo.T
#                 render_pose= self.new_views()
#                 # R,T=render_pose[index]
#                 FovX = caminfo.FovX
#                 FovY = caminfo.FovY
#                 time = caminfo.time
    
#                 #mask = caminfo['mask']
            
#             return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
#                               image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
#                               mask=None)
#         else:
#             return self.dataset[index]
#     def __len__(self):
        
#         return len(self.dataset)
    
#     def clone(self):
#         # 返回这个类实例的深拷贝
#         return copy.deepcopy(self)


#     def duplicate_and_append(self):
#         # 检查dataset是否支持切片和扩展操作
#         if hasattr(self.dataset, '__getitem__') and hasattr(self.dataset, 'extend'):
#             # 简单的浅复制，适用于不包含复杂对象的dataset
#             duplicated_dataset = self.dataset[:]  # 使用切片操作复制
#             self.dataset.extend(duplicated_dataset)
#         elif hasattr(self.dataset, '__iter__'):
#             # 对于包含复杂对象的dataset，使用deepcopy进行深复制
#             duplicated_dataset = copy.deepcopy(self.dataset)  # 使用deepcopy复制
#             # 需要根据实际情况确定如何合并原始和复制的数据集
#             # 这里假设self.dataset是一个列表
#             self.dataset.extend(duplicated_dataset)
#         else:
#             raise TypeError("Unsupported dataset type for duplication.")
# from torch.utils.data import Dataset
# from scene.cameras import Camera
# import numpy as np
# from utils.general_utils import PILtoTorch
# from utils.graphics_utils import fov2focal, focal2fov
# import torch
# from utils.camera_utils import loadCam
# from utils.graphics_utils import focal2fov
# class FourDGSdataset(Dataset):
#     def __init__(
#         self,
#         dataset,
#         args,
#         dataset_type
#     ):
#         self.dataset = dataset
#         self.args = args
#         self.dataset_type=dataset_type
#     def __getitem__(self, index):
#         # breakpoint()

#         if self.dataset_type != "PanopticSports":
#             try:
#                 image, w2c, time = self.dataset[index]
#                 R,T = w2c
#                 FovX = focal2fov(self.dataset.focal[0], image.shape[2])
#                 FovY = focal2fov(self.dataset.focal[0], image.shape[1])
#                 mask=None
#             except:
#                 caminfo = self.dataset[index]
#                 image = caminfo.image
#                 R = caminfo.R
#                 T = caminfo.T
#                 FovX = caminfo.FovX
#                 FovY = caminfo.FovY
#                 time = caminfo.time
    
#                 mask = caminfo.mask
#             return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
#                               image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
#                               mask=mask)
#         else:
#             return self.dataset[index]
#     def __len__(self):
        
#         return len(self.dataset)
#     def clone(self):
#         # 返回这个类实例的深拷贝
#         return copy.deepcopy(self)


#     def duplicate_and_append(self):
#         # 检查dataset是否支持切片和扩展操作
#         if hasattr(self.dataset, '__getitem__') and hasattr(self.dataset, 'extend'):
#             # 简单的浅复制，适用于不包含复杂对象的dataset
#             duplicated_dataset = self.dataset[:]  # 使用切片操作复制
#             self.dataset.extend(duplicated_dataset)
#         elif hasattr(self.dataset, '__iter__'):
#             # 对于包含复杂对象的dataset，使用deepcopy进行深复制
#             duplicated_dataset = copy.deepcopy(self.dataset)  # 使用deepcopy复制
#             # 需要根据实际情况确定如何合并原始和复制的数据集
#             # 这里假设self.dataset是一个列表
#             self.dataset.extend(duplicated_dataset)
#         else:
#             raise TypeError("Unsupported dataset type for duplication.")
import numpy as np
import copy
from torch.utils.data import Dataset
from scene.cameras import Camera
import torch
from utils.graphics_utils import fov2focal, focal2fov
from scipy.spatial.transform import Rotation as R

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def quaternion_average(quaternions):
    """
    计算四元数的平均值
    :param quaternions: 四元数的列表，形状为 (N, 4)
    :return: 平均四元数
    """
    q_matrix = np.array(quaternions)
    q_avg = np.mean(q_matrix, axis=0)
    q_avg_normalized = q_avg / np.linalg.norm(q_avg)
    return q_avg_normalized
def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    
    # 确保 rads 是一个长度为 3 的数组
    if isinstance(rads, (float, np.float64)):
        rads = np.array([rads, rads, rads])
    else:
        rads = np.array(rads)

    hwf = c2w[:3, 4:5]  # 调整 hwf 的形状，使其匹配 viewmatrix 的输出

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        # 创建一个长度为 3 的向量，并与 rads 相乘
        direction = np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate)]) * rads
        c = np.dot(c2w[:3, :4], np.append(direction, 1.))
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))  # 将 hwf 和 viewmatrix 连接
    
    return render_poses




class FourDGSdataset(Dataset):
    def __init__(self, dataset, args, dataset_type):
        self.dataset = dataset
        self.args = args
        self.dataset_type = dataset_type
        self.focal = args.focal if hasattr(args, 'focal') else None  # 从 args 获取焦距信息
        self.get_aver()
        self.all_render_poses = self.new_views()  # 生成120个视角

    def get_aver(self):
        Rs = []
        Ts = []
        for caminfo in self.dataset:
            try:
                Rs.append(caminfo.R)
                Ts.append(caminfo.T)
            except AttributeError:
                continue
        quaternions = [R.from_matrix(r).as_quat() for r in Rs]
        avg_quat = quaternion_average(quaternions)
        avg_R = R.from_quat(avg_quat).as_matrix()
        avg_T = np.mean(Ts, axis=0)
        self.c2w = np.eye(4)
        self.c2w[:3, :3] = avg_R
        self.c2w[:3, 3] = avg_T
        self.up = np.mean([R[:, 1] for R in Rs], axis=0)
        self.up = normalize(self.up)
        self.center = np.mean(Ts, axis=0)
        self.distances = np.linalg.norm(Ts - self.center, axis=1)
        self.rads = np.percentile(self.distances, 90)
       
    def new_views(self):
        N_views = 120  # 生成120个视角
        rots = 2  # 旋转两圈
        zrate = 0.5  # 控制Z轴上的变化率
        zdelta = 0.2  # 控制Z轴的增量
        focal = self.focal or fov2focal(self.dataset[0].FovX, self.dataset[0].image.shape[2])  # 使用给定的或计算的焦距
        render_poses = render_path_spiral(self.c2w, self.up, self.rads, focal, zdelta, zrate, rots, N_views)
        return render_poses

    def __getitem__(self, index):
        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]  # 尝试从 dataset 获取 image, w2c, time
                R, T = w2c
                FovX = focal2fov(self.focal, image.shape[2])
                FovY = focal2fov(self.focal, image.shape[1])
                mask = None
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
                mask = caminfo.mask

            # 使用渲染的视角
            render_pose = self.all_render_poses[index]
            R = render_pose[:3, :3]  # 从渲染位姿中提取R
            T = render_pose[:3, 3]   # 从渲染位姿中提取T

            # 确定时间：前60张为0.2，后60张为0.8
            if index < 60:
                time = 0.2
            else:
                time = 0.8
                
            return Camera(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=mask,
                          image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time)

    def __len__(self):
        return 120  # 总共生成120张图像
    
    def clone(self):
        return copy.deepcopy(self)

    def duplicate_and_append(self):
        if hasattr(self.dataset, '__getitem__') and hasattr(self.dataset, 'extend'):
            duplicated_dataset = self.dataset[:]
            self.dataset.extend(duplicated_dataset)
        elif hasattr(self.dataset, '__iter__'):
            duplicated_dataset = copy.deepcopy(self.dataset)
            self.dataset.extend(duplicated_dataset)
        else:
            raise TypeError("Unsupported dataset type for duplication.")


