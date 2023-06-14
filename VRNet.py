import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib as plt
from PIL import Image

## SpatialSoftmax implementation taken from https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834
class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = torch.nn.Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints

#Implementation of Network from Figure 3 (on pg 4) of paper
class VRNet(nn.Module):
    def __init__(self):
        super(VRNet, self).__init__()
        # Convolution 1 160x120x4 -> 77x57x240
        self.conv1_rgbTop = nn.Conv2d(3, 64, 7, padding='valid', stride=2)
        self.conv1_depthTop = nn.Conv2d(1, 16, 7, padding='valid', stride=2)
        self.conv1_rgbEff = nn.Conv2d(3, 64, 7, padding='valid', stride=2)
        self.conv1_depthEff = nn.Conv2d(1, 16, 7, padding='valid', stride=2)
        self.conv1_rgbSide = nn.Conv2d(3, 64, 7, padding='valid', stride=2)
        self.conv1_depthSide = nn.Conv2d(1, 16, 7, padding='valid', stride=2)
        
        # Convolution 2 77x57x240 -> 77x57x32
        self.conv2 = nn.Conv2d(240, 32, 1, padding='same')
        self.conv2_bn = nn.BatchNorm2d(32, eps=0.001, momentum=0.99)
        # Convolution 3 77x57x43 -> 75x55x32
        self.conv3 = nn.Conv2d(32, 32, 3, padding='valid')
        self.conv3_bn = nn.BatchNorm2d(32, eps=0.001, momentum=0.99)
        # Convolution 4 75x55x32 -> 73x53x32
        self.conv4 = nn.Conv2d(32, 32, 3, padding='valid')
        self.conv4_bn = nn.BatchNorm2d(32, eps=0.001, momentum=0.99)

        # spatial softmax
        self.spatialSoftmax = SpatialSoftmax(53, 73, 32, temperature=1, data_format='NCHW')

        self.flatten = nn.Flatten()

        #fully connected layers
        self.fc1 = nn.Linear(64, 50)
        self.fc1_bn = nn.BatchNorm1d(50, eps=0.001, momentum=0.99)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_bn = nn.BatchNorm1d(50, eps=0.001, momentum=0.99)
        self.fc3 = nn.Linear(50, 50)
        self.fc3_bn = nn.BatchNorm1d(50, eps=0.001, momentum=0.99)
        self.fc4 = nn.Linear(50, 7) # Vx, Vy, Vz, Wx, Wy, Wz, grabber open

        #set conv1_rgb weights to be first layer from pretrained model
        googlenet = torchvision.models.googlenet(pretrained=True)
        self.conv1_rgbTop.weight.data = googlenet.conv1.conv.weight.data

        #weights should be uniformly sampled from [-0.1, 0.1]
        self.conv1_depthTop.weight.data.uniform_(-0.1, 0.1)
        self.conv1_rgbEff.weight.data.uniform_(-0.1, 0.1)
        self.conv1_depthEff.weight.data.uniform_(-0.1, 0.1)
        self.conv1_rgbSide.weight.data.uniform_(-0.1, 0.1)
        self.conv1_depthSide.weight.data.uniform_(-0.1, 0.1)
        self.conv2.weight.data.uniform_(-0.1, 0.1)
        self.conv3.weight.data.uniform_(-0.1, 0.1)
        self.conv4.weight.data.uniform_(-0.1, 0.1)

        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc3.weight.data.uniform_(-0.1, 0.1)
        self.fc4.weight.data.uniform_(-0.1, 0.1)

    def forward(self, rgbTopImg, depthTopImg, rgbEffImg, depthEffImg, rgbSideImg, depthSideImg):
        #conv layers
        x_rgbTop = F.relu(self.conv1_rgbTop(rgbTopImg))
        x_depthTop = F.relu(self.conv1_depthTop(depthTopImg))
        x_rgbEff = F.relu(self.conv1_rgbEff(rgbEffImg))
        x_depthEff = F.relu(self.conv1_depthEff(depthEffImg))
        x_rgbSide = F.relu(self.conv1_rgbSide(rgbSideImg))
        x_depthSide = F.relu(self.conv1_depthSide(depthSideImg))

        x = torch.cat((x_rgbTop, x_depthTop, x_rgbEff, x_depthEff, x_rgbSide, x_depthSide), 1)
        
        #implement convulutional layers with batch normalization
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.spatialSoftmax(x)
        x = self.flatten(x)

        #fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class VRDataLoader(Dataset):
    def __init__(self, data_dir, batch_size=1):
        self.data_dir = data_dir
        # self.startRun = startRun
        # self.lastRun = lastRun
        self.batch_size = batch_size
        self.rgbTop_images, self.depthTop_images, self.rgbEff_images, self.depthEff_images, self.rgbSide_images, self.depthSide_images, self.states = self.load_data()
        self.arrayIndicies = list([i for i in range(len(self.rgbTop_images))])
        
        # print(len(self.rgb_images), len(self.depth_images), len(self.states))
        # assert(len(self.rgb_images) == len(self.depth_images) == len(self.states))

    def load_data(self):
        rgbsTop = []
        depthsTop = []
        rgbsEff = []
        depthsEff = []
        rgbsSide = []
        depthsSide = []
        states = []
        
        for k in os.listdir(self.data_dir):
            rgbTop_dir = os.path.join(self.data_dir, k, 'rgb_Top')
            depthTop_dir = os.path.join(self.data_dir, k, 'depth_Top')
            rgbEff_dir = os.path.join(self.data_dir, k, 'rgb_Eff')
            depthEff_dir = os.path.join(self.data_dir, k, 'depth_Eff')
            rgbSide_dir = os.path.join(self.data_dir, k, 'rgb_Side')
            depthSide_dir = os.path.join(self.data_dir, k, 'depth_Side')

            state_dir = os.path.join(self.data_dir, k, 'states')
            
            state_names = os.listdir(state_dir) #get all files in the directory
            state_names = [int(state_name[6:-4]) for state_name in state_names if state_name.endswith('.csv')] #only get the csv files
            
            num_points = sorted(state_names)
            print(num_points)
            lastState = None

            this_run_states = []
            for i in num_points:
                
                rgbTop_path = os.path.join(rgbTop_dir, f'rgb{i}.png')
                depthTop_path = os.path.join(depthTop_dir, f'depth{i}.png')
                rgbEff_path = os.path.join(rgbEff_dir, f'rgb{i}.png')
                depthEff_path = os.path.join(depthEff_dir, f'depth{i}.png')
                rgbSide_path = os.path.join(rgbSide_dir, f'rgb{i}.png')
                depthSide_path = os.path.join(depthSide_dir, f'depth{i}.png')                
                
                state_path = os.path.join(state_dir, f'states{i}.csv')
                
                with open(state_path, 'r') as f:
                    data = f.readlines()
                    isOpen = int(data[0].split(',')[6])
                    data = data[1].split(',')
                    data.append(isOpen)
                    #convert to pytorch tensor
                    state = np.array([float(x) for x in data])
                    this_state = np.zeros(7)
                    this_state[0:6] = state[0:6]
                    this_state[6] = state[6]
                    this_run_states.append(this_state)
                
                if i == num_points[0]:
                    continue
                
                rgbTop = Image.open(rgbTop_path)
                depthTop = Image.open(depthTop_path)
                rgbTop = torchvision.transforms.ToTensor()(rgbTop)
                depthTop = torchvision.transforms.ToTensor()(depthTop) 
                rgbsTop.append(rgbTop)
                depthsTop.append(depthTop)

                rgbEff = Image.open(rgbEff_path)
                depthEff = Image.open(depthEff_path)
                rgbEff = torchvision.transforms.ToTensor()(rgbEff)
                depthEff = torchvision.transforms.ToTensor()(depthEff) 
                rgbsEff.append(rgbEff)
                depthsEff.append(depthEff)

                rgbSide = Image.open(rgbSide_path)
                depthSide = Image.open(depthSide_path)
                rgbSide = torchvision.transforms.ToTensor()(rgbSide)
                depthSide = torchvision.transforms.ToTensor()(depthSide) 
                rgbsSide.append(rgbSide)
                depthsSide.append(depthSide)

            #smooth out velocities with a gaussian for this run
            this_run_states = np.array(this_run_states, dtype=np.float32)
            
            size = 15
            sigma = 4
            filter_range = np.linspace(-int(size/2),int(size/2),size)
            gaussian_filter = [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2)) for x in filter_range]
            
            #smooth out positions with a gaussian
            for i in range(6):
                this_run_states[:, i] = np.convolve(this_run_states[:, i], gaussian_filter, mode='same')
            #take numerical derivative of position to get velocity
            this_run_states[:-1, 0:6] = np.diff(this_run_states[:, 0:6], axis=0)
            
            this_run_states = this_run_states[1:, :]
            
            #smooth out velocities with a gaussian
            # for i in range(6):
            #     this_run_states[:, i] = np.convolve(this_run_states[:, i], gaussian_filter, mode='same')

            #add to total list of states
            states.extend(this_run_states)

        #smooth out velocities with a moving average
        states = np.array(states, dtype=np.float32)
        #plot x velocity after smoothing
        states = torch.tensor(states).to('cuda')
        
        #compute mean and std of images
        rgbsTop = torch.stack(rgbsTop).float() / 255
        depthsTop = torch.stack(depthsTop).float() / 255
        rgbTop_mean = torch.mean(rgbsTop, dim=(0, 2, 3))
        depthTop_mean = torch.mean(depthsTop, dim=(0, 2, 3))

        rgbsEff = torch.stack(rgbsEff).float() / 255
        depthsEff = torch.stack(depthsEff).float() / 255
        rgbEff_mean = torch.mean(rgbsEff, dim=(0, 2, 3))
        depthEff_mean = torch.mean(depthsEff, dim=(0, 2, 3))

        rgbsSide = torch.stack(rgbsSide).float() / 255
        depthsSide = torch.stack(depthsSide).float() / 255
        rgbSide_mean = torch.mean(rgbsSide, dim=(0, 2, 3))
        depthSide_mean = torch.mean(depthsSide, dim=(0, 2, 3))
        
        #compute std
        rgbTop_std = torch.std(rgbsTop, dim=(0, 2, 3))
        depthTop_std = torch.std(depthsTop, dim=(0, 2, 3))

        rgbEff_std = torch.std(rgbsEff, dim=(0, 2, 3))
        depthEff_std = torch.std(depthsEff, dim=(0, 2, 3))

        rgbSide_std = torch.std(rgbsSide, dim=(0, 2, 3))
        depthSide_std = torch.std(depthsSide, dim=(0, 2, 3))

        #normalize images
        rgbsTop[:,0,:,:] = (rgbsTop[:,0,:,:] - rgbTop_mean[0]) / rgbTop_std[0]
        rgbsTop[:,1,:,:] = (rgbsTop[:,1,:,:] - rgbTop_mean[0]) / rgbTop_std[1]
        rgbsTop[:,2,:,:] = (rgbsTop[:,2,:,:] - rgbTop_mean[0]) / rgbTop_std[2]
        depthsTop = (depthsTop - depthTop_mean) / depthTop_std

        rgbsEff[:,0,:,:] = (rgbsEff[:,0,:,:] - rgbEff_mean[0]) / rgbEff_std[0]
        rgbsEff[:,1,:,:] = (rgbsEff[:,1,:,:] - rgbEff_mean[0]) / rgbEff_std[1]
        rgbsEff[:,2,:,:] = (rgbsEff[:,2,:,:] - rgbEff_mean[0]) / rgbEff_std[2]
        depthsEff = (depthsEff - depthEff_mean) / depthEff_std

        rgbsSide[:,0,:,:] = (rgbsSide[:,0,:,:] - rgbSide_mean[0]) / rgbSide_std[0]
        rgbsSide[:,1,:,:] = (rgbsSide[:,1,:,:] - rgbSide_mean[0]) / rgbSide_std[1]
        rgbsSide[:,2,:,:] = (rgbsSide[:,2,:,:] - rgbSide_mean[0]) / rgbSide_std[2]
        depthsSide = (depthsSide - depthSide_mean) / depthSide_std



        print('rgb mean: ', rgbTop_mean)
        print('rgb std: ', rgbTop_std)
        print('depth mean: ', depthTop_mean)
        print('depth std: ', depthTop_std)
        print('states mean: ', torch.mean(states, dim=0))
        print('states std: ', torch.std(states, dim=0))
        
        #normalize states
        for i in range(6):
            states[:, i] = (states[:, i] - torch.mean(states[:, i])) / torch.std(states[:, i])
        print(rgbsTop.shape, depthsTop.shape, rgbsEff.shape, depthsEff.shape, rgbsSide.shape, depthsSide.shape, states.shape)
        return rgbsTop, depthsTop, rgbsEff, depthsEff, rgbsSide, depthsSide, states
    
    def __len__(self):
        return len(self.states) // self.batch_size

    def __getitem__(self, idx):
        #shuffle array index mapping
        # if idx == 0:
        np.random.shuffle(self.arrayIndicies)
            
        idx = idx * self.batch_size
        desiredIndexes = self.arrayIndicies[idx:idx+self.batch_size]

        rgbTop_img = []
        depthTop_img = []
        rgbEff_img = []
        depthEff_img = []
        rgbSide_img = []
        depthSide_img = []
        state = []
        
        for i in desiredIndexes:
            rgbTop_img.append(self.rgbTop_images[i])
            depthTop_img.append(self.depthTop_images[i])
            rgbEff_img.append(self.rgbEff_images[i])
            depthEff_img.append(self.depthEff_images[i])
            rgbSide_img.append(self.rgbSide_images[i])
            depthSide_img.append(self.depthSide_images[i])

            state.append(self.states[i])

        rgbTop_img = torch.stack(rgbTop_img)
        depthTop_img = torch.stack(depthTop_img)
        rgbEff_img = torch.stack(rgbEff_img)
        depthEff_img = torch.stack(depthEff_img)
        rgbSide_img = torch.stack(rgbSide_img)
        depthSide_img = torch.stack(depthSide_img)

        state = torch.stack(state)

        return rgbTop_img, depthTop_img, rgbEff_img, depthEff_img, rgbSide_img, depthSide_img, state

class DataPreprocessor():
    def __init__(self, rgb_mean, rgb_std, depth_mean, depth_std, state_mean, state_std):
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.state_mean = state_mean
        self.state_std = state_std

    def normalizeRgb(self, rgb):
        rgb[:,0,:,:] = (rgb[:,0,:,:] - self.rgb_mean[0]) / self.rgb_std[0]
        rgb[:,1,:,:] = (rgb[:,1,:,:] - self.rgb_mean[1]) / self.rgb_std[1]
        rgb[:,2,:,:] = (rgb[:,2,:,:] - self.rgb_mean[2]) / self.rgb_std[2]
        return rgb
    
    def normalizeDepth(self, depth):
        return (depth - self.depth_mean) / self.depth_std
    
    def normalizeState(self, state):
        for i in range(6):
            state[:, i] = (state[:, i] - self.state_mean[i]) / self.state_std[i]
        return state
    
    def denormalizeState(self, state):
        for i in range(6):
            state[:, i] = (state[:, i] * self.state_std[i]) + self.state_mean[i]
        return state
    

