import torch.nn as nn, torch

#### The below code is from ViNet ####

class VideoSaliencyModel(nn.Module):
    def __init__(self, time_width):
        super(VideoSaliencyModel, self).__init__()
        self.backbone = BackBoneS3D()
        self.vin_decoder = VinDecoder()
        self.ssl_decoder = SSLDecoder()
        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=(11,11,11), padding=(5,5,5)),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=(0.5,1,1), mode='trilinear'), # torch.Size([1, 4, 16, 288, 512])

            nn.ConvTranspose3d(in_channels=4, out_channels=2, kernel_size=(9,9,9), padding=(4,4,4)),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=(0.5,1,1), mode='trilinear'), # torch.Size([1, 2, 8, 288, 512])

            nn.ConvTranspose3d(in_channels=2, out_channels=1, kernel_size=(7,7,7), padding=(3,3,3)),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=(1/(time_width*0.5*0.5),1,1), mode='trilinear')
        )

    def forward(self, video_data, mmv_embeddings):
        y0, y1, y2, y3 = self.backbone(video_data)
        # print(y0.shape, y1.shape, y2.shape, y3.shape) # Initial size: torch.Size([4, 1024, 4, 7, 12]) torch.Size([4, 832, 8, 14, 24]) torch.Size([4, 480, 16, 28, 48]) torch.Size([4, 192, 16, 56, 96])
        # print(y0.shape, y1.shape, y2.shape, y3.shape) # Final size: torch.Size([1, 1024, 4, 9, 16]) torch.Size([1, 832, 8, 18, 32]) torch.Size([1, 480, 16, 36, 64]) torch.Size([1, 192, 16, 72, 128])
        vin_decoder_outputs = self.vin_decoder(y0, y1, y2, y3)
        s0, s1, s2, s3, s4, s5 = mmv_embeddings
        # print(s0.shape, s1.shape, s2.shape, s3.shape, s4.shape, s5.shape) # torch.Size([2, 128, 32, 72, 128]) torch.Size([2, 512, 32, 72, 128]) torch.Size([2, 1024, 32, 36, 64]) torch.Size([2, 2048, 32, 18, 32]) torch.Size([2, 4096, 32, 9, 16]) torch.Size([2, 4096, 32, 9, 16])
        ssl_decoder_outputs = self.ssl_decoder(s0, s1, s2, s3, s4, s5)
        decoder_outputs = torch.cat((vin_decoder_outputs, ssl_decoder_outputs), dim=1) # torch.Size([1, 8, 32, 288, 512])
        result = self.final_layer(decoder_outputs) # torch.Size([1, 1, 1, 288, 512])
        return result

class VinDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=1024, out_channels=64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2,2), mode='trilinear')
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=896, out_channels=32, kernel_size=(5,5,5), padding=(2,2,2)),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2,2), mode='trilinear')
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=512, out_channels=16, kernel_size=(7,7,7), padding=(3,3,3)),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=208, out_channels=8, kernel_size=(9,9,9), padding=(4,4,4)),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2,2), mode='trilinear'), # torch.Size([1, 8, 32, 144, 256])

            nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=(7,7,7), padding=(3,3,3)),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
        )
    def forward(self, y0, y1, y2, y3):
        result = self.layer1(y0) # torch.Size([1, 64, 8, 18, 32])
        result = torch.cat((y1, result), dim=1) # torch.Size([1, 896, 8, 18, 32])
        result = self.layer2(result) # torch.Size([1, 32, 16, 36, 64])
        result = torch.cat((y2, result), dim=1) # torch.Size([1, 512, 16, 36, 64])
        result = self.layer3(result) # torch.Size([1, 16, 16, 72, 128])
        result = torch.cat((y3, result), dim=1) # torch.Size([1, 208, 16, 72, 128])
        result = self.layer4(result) # torch.Size([1, 4, 32, 288, 512])
        return result

class SSLDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=4096, out_channels=64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.LeakyReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=4160, out_channels=64, kernel_size=(5,5,5), padding=(2,2,2)),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=2112, out_channels=32, kernel_size=(7,7,7), padding=(3,3,3)),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=1056, out_channels=16, kernel_size=(9,9,9), padding=(4,4,4)),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=656, out_channels=8, kernel_size=(11,11,11), padding=(5,5,5)),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear'), # torch.Size([1, 8, 32, 144, 256])

            nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=(7,7,7), padding=(3,3,3)),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
        )
    def forward(self, s0, s1, s2, s3, s4, s5):
        result = self.layer1(s5) # torch.Size([1, 64, 32, 9, 16])
        result = torch.cat((s4, result), dim=1) # torch.Size([1, 4160, 32, 9, 16])
        result = self.layer2(result) # torch.Size([1, 64, 32, 18, 32])
        result = torch.cat((s3, result), dim=1) # torch.Size([1, 2112, 32, 18, 32])
        result = self.layer3(result) # torch.Size([1, 32, 32, 36, 64])
        result = torch.cat((s2, result), dim=1) # torch.Size([1, 1056, 32, 36, 64])
        result = self.layer4(result) # torch.Size([1, 16, 32, 72, 128])
        result = torch.cat((s0, s1, result), dim=1) # torch.Size([1, 656, 32, 72, 128])
        result = self.layer5(result) # torch.Size([1, 4, 32, 288, 512])
        return result

class DecoderConvUp(nn.Module):
    def __init__(self):
        super(DecoderConvUp, self).__init__()
        self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
        self.convtsp1 = nn.Sequential(
			nn.Conv3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
        self.convtsp2 = nn.Sequential(
			nn.Conv3d(832, 480, kernel_size=(3,3,3), stride=(3,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
        self.convtsp3 = nn.Sequential(
			nn.Conv3d(480, 192, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling
		)
        self.convtsp4 = nn.Sequential(
			nn.Conv3d(192, 64, kernel_size=(5,3,3), stride=(5,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 112 x 192. New: torch.Size([1, 64, 4, 144, 256])

			nn.Conv3d(64, 32, kernel_size=(2,3,3), stride=(2,1,1), padding=(0,1,1), bias=False),
			nn.ReLU(),
			self.upsampling, # 224 x 384. New: torch.Size([1, 32, 2, 288, 512])

			# 4 time dimension
			nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
			nn.ReLU(), # New: torch.Size([1, 32, 1, 288, 512])
            nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )
    
    def forward(self, y0, y1, y2, y3):
        z = self.convtsp1(y0)
        print('convtsp1', z.shape) # torch.Size([4, 832, 4, 14, 24]). New: torch.Size([1, 832, 4, 18, 32])
        
        z = torch.cat((z,y1), 2)
        print('cat_convtsp1', z.shape) # torch.Size([4, 832, 12, 14, 24]). New: torch.Size([1, 832, 12, 18, 32])
        
        z = self.convtsp2(z)
        print('convtsp2', z.shape) # torch.Size([4, 480, 4, 28, 48]). New: torch.Size([1, 480, 4, 36, 64])
        
        z = torch.cat((z,y2), 2)
        print('cat_convtsp2', z.shape) # torch.Size([4, 480, 20, 28, 48]). New: torch.Size([1, 480, 20, 36, 64])
        
        z = self.convtsp3(z)
        print('convtsp3', z.shape) # torch.Size([4, 192, 4, 56, 96]). New: torch.Size([1, 192, 4, 72, 128])
        
        z = torch.cat((z,y3), 2)
        print("cat_convtsp3", z.shape) # torch.Size([4, 192, 20, 56, 96]). New: torch.Size([1, 192, 20, 72, 128])
        
        z = self.convtsp4(z)
        print('convtsp4', z.shape) # torch.Size([4, 1, 1, 224, 384]). New: torch.Size([1, 1, 1, 288, 512])
        
        z = z.view(z.size(0), z.size(3), z.size(4))
        print('output', z.shape) # torch.Size([4, 224, 384]). New: torch.Size([1, 288, 512])
        
        return z

class BackBoneS3D(nn.Module):
	def __init__(self):
		super(BackBoneS3D, self).__init__()
		
		self.base1 = nn.Sequential(
			SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
			nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
			BasicConv3d(64, 64, kernel_size=1, stride=1),
			SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
		)
		self.maxp2 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
		self.base2 = nn.Sequential(
			Mixed_3b(),
			Mixed_3c(),
		)
		self.maxp3 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
		self.base3 = nn.Sequential(
			Mixed_4b(),
			Mixed_4c(),
			Mixed_4d(),
			Mixed_4e(),
			Mixed_4f(),
		)
		self.maxt4 = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1), padding=(0,0,0))
		self.maxp4 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0))
		self.base4 = nn.Sequential(
			Mixed_5b(),
			Mixed_5c(),
		)

	def forward(self, x):
		# print('input', x.shape)
		y3 = self.base1(x)
		# print('base1', y3.shape)
		
		y = self.maxp2(y3)
		# print('maxp2', y.shape)

		y2 = self.base2(y)
		# print('base2', y2.shape)

		y = self.maxp3(y2)
		# print('maxp3', y.shape)

		y1 = self.base3(y)
		# print('base3', y1.shape)

		y = self.maxt4(y1)
		y = self.maxp4(y)
		# print('maxt4p4', y.shape)

		y0 = self.base4(y)

		return [y0, y1, y2, y3]

##### The below code is taken from TASED-Net #####

class SepConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(SepConv3d, self).__init__()
        self.conv_s = nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size), stride=(1,stride,stride), padding=(0,padding,padding), bias=False)
        self.bn_s = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False)
        self.bn_t = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Mixed_3b(nn.Module):
    def __init__(self):
        super(Mixed_3b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(192, 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(192, 96, kernel_size=1, stride=1),
            SepConv3d(96, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(192, 16, kernel_size=1, stride=1),
            SepConv3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out

class Mixed_3c(nn.Module):
    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
            SepConv3d(128, 192, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 32, kernel_size=1, stride=1),
            SepConv3d(32, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(256, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Mixed_4b(nn.Module):
    def __init__(self):
        super(Mixed_4b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(480, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(480, 96, kernel_size=1, stride=1),
            SepConv3d(96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(480, 16, kernel_size=1, stride=1),
            SepConv3d(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(480, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Mixed_4c(nn.Module):
    def __init__(self):
        super(Mixed_4c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
            SepConv3d(112, 224, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            SepConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Mixed_4d(nn.Module):
    def __init__(self):
        super(Mixed_4d, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
            SepConv3d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            SepConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Mixed_4e(nn.Module):
    def __init__(self):
        super(Mixed_4e, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 144, kernel_size=1, stride=1),
            SepConv3d(144, 288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 32, kernel_size=1, stride=1),
            SepConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Mixed_4f(nn.Module):
    def __init__(self):
        super(Mixed_4f, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(528, 160, kernel_size=1, stride=1),
            SepConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(528, 32, kernel_size=1, stride=1),
            SepConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(528, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 160, kernel_size=1, stride=1),
            SepConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 32, kernel_size=1, stride=1),
            SepConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Mixed_5c(nn.Module):
    def __init__(self):
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 192, kernel_size=1, stride=1),
            SepConv3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 48, kernel_size=1, stride=1),
            SepConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out
