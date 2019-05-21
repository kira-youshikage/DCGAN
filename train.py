import torchvision as tv
import torch as pt
import torch.utils.data
import os
import tqdm
import model
from config import Config

opt = Config()
# 数据读取与加载
transforms = tv.transforms.Compose([
	tv.transforms.Scale(opt.image_size),
	tv.transforms.CenterCrop(opt.image_size),
	tv.transforms.ToTensor(),
	tv.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
dataset = tv.datasets.ImageFolder(os.path.join(opt.data_path), transform = transforms)
dataloader = torch.utils.data.DataLoader(dataset,
										batch_size = opt.batch_size,
										shuffle = True,
										num_workers = opt.num_workers,
										drop_last = True)
# 定义网络
if opt.load_model:
	netd = pt.load(opt.netd_path, map_location=lambda storage, loc: storage)
	netg = pt.load(opt.netg_path, map_location=lambda storage, loc: storage)
else:
	netd = model.NetD()
	netg = model.NetG()
# 定义优化器和损失
optimizer_g = pt.optim.Adam(netg.parameters(), opt.lr1, betas = (opt.beta1, 0.999))# 生成器优化器
optimizer_d = pt.optim.Adam(netd.parameters(), opt.lr2, betas = (opt.beta1, 0.999))# 判别器优化器
criterion = pt.nn.BCELoss()
# 真图片label为1，假图片label为0，noises为生成网络的输入噪声
true_labels = pt.autograd.Variable(pt.ones(opt.batch_size))
fake_labels = pt.autograd.Variable(pt.zeros(opt.batch_size))
fix_noises = pt.autograd.Variable(pt.randn(opt.batch_size, opt.nz, 1, 1))# 固定噪声，用于训练效果可视化
noises = pt.autograd.Variable(pt.randn(opt.batch_size, opt.nz, 1, 1))
# 如果使用GPU，则将数据移至GPU上：
if opt.use_gpu:
	netd.cuda()
	netg.cuda()
	criterion.cuda()
	true_labels, fake_labels = true_labels.cuda(), fake_labels.cuda()
	fix_noises, noises = fix_noises.cuda(), noises.cuda()

if __name__ == '__main__':
	# 训练网络过程
	for i in range(opt.max_epoch):
		for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
			real_img = pt.autograd.Variable(img)
			if opt.use_gpu:
				real_img = real_img.cuda()
			# 训练判别器
			if (ii + 1)% opt.d_every == 0:
				optimizer_d.zero_grad()
				# 尽可能地把真图片判别为1
				output = netd(real_img)
				error_d_real = criterion(output, true_labels)
				error_d_real.backward()
				# 尽可能地把假图片判别为0
				noises.data.copy_(pt.randn(opt.batch_size, opt.nz, 1, 1))
				fake_img = netg(noises).detach() # 根据噪声生成假图
				fake_output = netd(fake_img)
				error_d_fake = criterion(fake_output, fake_labels)
				error_d_fake.backward()
				optimizer_d.step()
			# 训练生成器
			if (ii + 1) % opt.g_every == 0:
				optimizer_g.zero_grad()
				noises.data.copy_(pt.randn(opt.batch_size, opt.nz, 1, 1))
				fake_img = netg(noises)
				fake_output = netd(fake_img)
				# 尽可能让判别器把假图片也判别为1
				error_g = criterion(fake_output, true_labels)
				error_g.backward()
				optimizer_g.step()
		if (i + 1) % opt.decay_every == 0:
			pt.save(netd, "save_model/netd.pth")
			pt.save(netg, "save_model/netg.pth")
