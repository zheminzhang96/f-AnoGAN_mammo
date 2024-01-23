from torch import nn
from torch.autograd import grad
import torch
DIM = 64
OUTPUT_DIM = 64*64*3


class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True,  stride=1, bias=True):
        # print('....within MyConvo2d....')
        # print('input_dim', input_dim)
        # print('output_dim', output_dim)
        # print('kernel_size', kernel_size)
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size,
                              stride=1, padding=self.padding, bias=bias) # input_dim: input channel; output_dim: output channel

    def forward(self, input):
        output = self.conv(input)
        return output


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim,
                              kernel_size, he_init=self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] +
                  output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        return output


class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim,
                              kernel_size, he_init=self.he_init)

    def forward(self, input):
        output = input
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] +
                  output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        output = self.conv(output)
        return output


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1) #[64, 2048, 4, 4] --> [64, 4, 4, 2048]
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height,
                             input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size, input_height,
                              output_width, output_depth) for t_t in spl]
        output = torch.stack(stacks, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(
            batch_size, output_height, output_width, output_depth)
        output = output.permute(0, 3, 1, 2)
        return output #[64, 512, 8, 8]


class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim,
                              kernel_size, he_init=self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        #print('----------within UpSampleConv forward-----------')
        output = input
        output = torch.cat((output, output, output, output), 1)
        #print(output.shape) #torch.Size([64, 2048, 4, 4])
        output = self.depth_to_space(output)  # image height*2, width*2, depth/4 --> [64, 512, 8, 8]
        #print('after depth to space: ', output.shape)
        output = self.conv(output)
        #print('after conv:', output.shape) # [64, 512, 8, 8] --> [64, 512, 8, 8]
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=DIM):
        super(ResidualBlock, self).__init__()
        # print('---------within ResidualBlock---------')
        # print('input_dim', input_dim)
        # print('output_dim', output_dim)
        # print('kernel_size', kernel_size)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'up':
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample == None:
            # TODO: ????
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(
                input_dim, output_dim, kernel_size=1, he_init=False)  # [64, 64, 64, 64] --> [64, 128, 32, 32]
            self.conv_1 = MyConvo2d(
                input_dim, input_dim, kernel_size=kernel_size, bias=False) # [64, 64, 64, 64]
            self.conv_2 = ConvMeanPool(
                input_dim, output_dim, kernel_size=kernel_size) # [64, 128, 32, 32]
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(
                input_dim, output_dim, kernel_size=1, he_init=False)  # shortcut/residual layer to manipulate dim directly from input
            self.conv_1 = UpSampleConv(
                input_dim, output_dim, kernel_size=kernel_size, bias=False) # upsample from input
            self.conv_2 = MyConvo2d(
                output_dim, output_dim, kernel_size=kernel_size) 
        elif resample == None:
            self.conv_shortcut = MyConvo2d(
                input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1 = MyConvo2d(
                input_dim, input_dim, kernel_size=kernel_size, bias=False)
            self.conv_2 = MyConvo2d(
                input_dim, output_dim, kernel_size=kernel_size)
        else:
            raise Exception('invalid resample value')

    def forward(self, input):
        print('---------within ResidualBlock forward---------')
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)
            print('after conv_shortcut', shortcut.shape) # up [64, 512, 4, 4] --> [64, 512, 8, 8]

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        print('after conv_1', output.shape) # up [64, 512, 8, 8]
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)
        print('after conv_2', output.shape)

        return shortcut + output # up [64, 512, 8, 8]


class ReLULayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(ReLULayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.linear(input)
        output = self.relu(output)
        return output


class FCGenerator(nn.Module):
    def __init__(self, FC_DIM=512):
        super(FCGenerator, self).__init__()
        self.relulayer1 = ReLULayer(128, FC_DIM)
        self.relulayer2 = ReLULayer(FC_DIM, FC_DIM)
        self.relulayer3 = ReLULayer(FC_DIM, FC_DIM)
        self.relulayer4 = ReLULayer(FC_DIM, FC_DIM)
        self.linear = nn.Linear(FC_DIM, OUTPUT_DIM)
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.relulayer1(input)
        output = self.relulayer2(output)
        output = self.relulayer3(output)
        output = self.relulayer4(output)
        output = self.linear(output)
        output = self.tanh(output)
        return output


class GoodGenerator(nn.Module):
    def __init__(self, dim=DIM, output_dim=OUTPUT_DIM):
        super(GoodGenerator, self).__init__()

        self.dim = dim

        self.ln1 = nn.Linear(128, 4*4*8*self.dim)
        self.rb1 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample='up')
        self.rb2 = ResidualBlock(8*self.dim, 4*self.dim, 3, resample='up')
        self.rb3 = ResidualBlock(4*self.dim, 2*self.dim, 3, resample='up')
        self.rb4 = ResidualBlock(2*self.dim, 1*self.dim, 3, resample='up')
        self.bn = nn.BatchNorm2d(self.dim)

        self.conv1 = MyConvo2d(1*self.dim, 3, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        print('*******************Within GoodGenerator forward****************')
        output = self.ln1(input.contiguous()) #torch.Size([64, 8192])
        #print('ln1 shape', output.shape)
        output = output.view(-1, 8*self.dim, 4, 4) #torch.Size([64, 512, 4, 4])
        #print('after reshape', output.shape)
        output = self.rb1(output)
        #print('rb1 ', output.shape) # torch.Size([64, 512, 8, 8])
        output = self.rb2(output)
        #print('rb2 ', output.shape) # torch.Size([64, 256, 16, 16])
        output = self.rb3(output)
        #print('rb3 ', output.shape) # torch.Size([64, 128, 32, 32])
        output = self.rb4(output)
        #print('rb4 ', output.shape) # torch.Size([64, 64, 64, 64])

        output = self.bn(output)
        #print('after bn', output.shape)
        output = self.relu(output)
        #print('after relu', output.shape)
        output = self.conv1(output)
        #print('after conv1', output.shape)
        output = self.tanh(output)
        # output = output.view(-1, OUTPUT_DIM)
        #print('final shape', output.shape) # torch.Size([64, 3, 64, 64])
        return output




class GoodDiscriminator(nn.Module):
    def __init__(self, dim=DIM):
        super(GoodDiscriminator, self).__init__()

        self.dim = dim
        print('******************Within GoodDiscriminator******************')
        self.conv1 = MyConvo2d(3, self.dim, 3, he_init=False)
        self.rb1 = ResidualBlock(self.dim, 2*self.dim,
                                 3, resample='down', hw=DIM)
        self.rb2 = ResidualBlock(
            2*self.dim, 4*self.dim, 3, resample='down', hw=int(DIM/2))
        self.rb3 = ResidualBlock(
            4*self.dim, 8*self.dim, 3, resample='down', hw=int(DIM/4))
        self.rb4 = ResidualBlock(
            8*self.dim, 8*self.dim, 3, resample='down', hw=int(DIM/8))
        self.ln1 = nn.Linear(4*4*8*self.dim, 1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def extract_feature(self, input):
        print('******************Within GoodDiscriminator extract_feature******************')
        print('input shape', input.shape)
        output = input.contiguous()
        print(output.shape)
        output = output.view(-1, 3, DIM, DIM)
        print('after reshape', output.shape)
        output = self.conv1(output)
        print('after conv1', output.shape)
        output = self.rb1(output)
        print('after rb1', output.shape)
        output = self.rb2(output)
        print('after rb2', output.shape)
        output = self.rb3(output)
        print('after rb3', output.shape)
        output = self.rb4(output)
        print('after rb4', output.shape)
        output = output.view(-1, 4*4*8*self.dim)
        print('output extract_feature', output.shape)
        return output

    def forward(self, input):
        print('******************Within GoodDiscriminator forward******************')
        print('input shape', input.shape)
        output = self.extract_feature(input)
        print('after feature_extract', output.shape)
        output = self.ln1(output)
        print('after ln1', output.shape)
        output = output.view(-1)
        print('final ', output.shape)
        return output





class Encoder(nn.Module):
    def __init__(self, dim, output_dim, drop_rate=0.0):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.conv_in = nn.Conv2d(3, dim, 3, 1, padding=1)
        self.res1 = ResidualBlock(dim, dim*2, 3, 'down', 64)
        self.res2 = ResidualBlock(dim*2, dim*4, 3, 'down', 32)
        self.res3 = ResidualBlock(dim*4, dim*8, 3, 'down', 16)
        self.res4 = ResidualBlock(dim*8, dim*8, 3, 'down', 8)
        self.fc = nn.Linear(4*4*8*dim, output_dim)

    def forward(self, x):
        print('******************Within Encoder forward******************')
        x = self.dropout(x)
        x = self.conv_in(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.tanh(x)



