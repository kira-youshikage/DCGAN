import torch as pt
from config import Config

opt = Config()

class NetG(pt.nn.Module):
	'''
	生成器定义
		当kernel_size为4,stride为2,padding为1时，根据公式
		H_out = (H_in - 1) * stride - 2 * padding + kernel_size
		输出尺寸正好变为输入的两倍
	'''
	def __init__(self):
		super(NetG, self).__init__()
		ngf = opt.ngf
		self.main = pt.nn.Sequential(
			# 输入是nz维度的噪声，可以认为其是一个nz*1*1的feature map
			pt.nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias = False), #上卷积层
			pt.nn.BatchNorm2d(ngf * 8), # 批规范化层
			pt.nn.ReLU(True),
			# 上一步的输出形状：(ngf * 8) * 4 * 4
			pt.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False),
			pt.nn.BatchNorm2d(ngf * 4),
			pt.nn.ReLU(True),
			# 上一步的输出形状：(ngf * 4) * 8 * 8
			pt.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias = False),
			pt.nn.BatchNorm2d(ngf * 2),
			pt.nn.ReLU(True),
			# 上一步的输出形状：(ngf * 2) * 16 * 16
			pt.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias = False),
			pt.nn.BatchNorm2d(ngf),
			pt.nn.ReLU(True),
			# 上一步的输出形状：ngf * 32 * 32
			pt.nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias = False),
			pt.nn.Tanh()
			# 输出形状：3 * 96 * 96
			)

	def forward(self, input):
		return self.main(input)

class NetD(pt.nn.Module):
	'''
	判别器
		网络形状与生成器对称
	'''
	def __init__(self):
		super(NetD, self).__init__()
		ndf = opt.ndf
		self.main = pt.nn.Sequential(
			# 输入：3*96*96
			pt.nn.Conv2d(3, ndf, 5, 3, 1, bias = False),
			pt.nn.LeakyReLU(0.2, inplace = True),
			# 输出：ndf * 32 * 32
			pt.nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias = False),
			pt.nn.BatchNorm2d(ndf * 2),
			pt.nn.LeakyReLU(0.2, inplace = True),
			# 输出：(ndf*2)*16*16
			pt.nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = False),
			pt.nn.BatchNorm2d(ndf * 4),
			pt.nn.LeakyReLU(0.2, inplace = True),
			# 输出：(ndf*4)*8*8
			pt.nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias = False),
			pt.nn.BatchNorm2d(ndf * 8),
			pt.nn.LeakyReLU(0.2, inplace = True),
			# 输出：(ndf*8)*4*4
			pt.nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias = False),
			pt.nn.Sigmoid() # 输出一个数（概率）
			)

	def forward(self, input):
		return self.main(input).view(-1)