class Config(object):
	data_path = "data/"# 数据集存放路径
	num_workers = 4 # 多进程加载数据所用的进程数
	image_size = 96 # 图片尺寸
	batch_size = 50
	max_epoch = 200
	lr1 = 2e-4 # 生成器的学习率
	lr2 = 2e-4 # 判别器的学习率
	beta1 = 0.5 #Adam优化器的beta1参数
	use_gpu = True # 是否使用GPU
	nz = 100 # 噪声的维度
	ngf = 64 # 生成器feature map数
	ndf = 64 # 判别器feature map数
	save_path = 'imgs' # 生成图片保存路径
	vis = True # 是否使用visdom可视化
	env = 'GAN' # visdom的env
	plot_every = 20 # 每间隔20batch,visdom画图一次
	d_every = 1 # 每1个batch训练一次判别器
	g_every = 5 # 每5个batch训练一次生成器
	decay_every = 1 # 每10个epoch保存一次模型
	load_model = True # 是否加载模型
	netd_path = 'save_model/netd.pth' # 预训练模型
	netg_path = 'save_model/netg.pth'
	# 测试时用的参数
	gen_img = 'result.png'
	# 从512张生产图中保存最好的64张
	gen_num = 64
	gen_search_num = 512
	gen_mean = 0 # 噪声的均值
	gen_std = 1 # 噪声的方差

