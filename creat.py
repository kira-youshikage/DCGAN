import model
from config import Config
import torch as pt
import torchvision as tv

opt = Config()

# 加载预训练好的模型，并利用噪声随机生成图片
# 定义噪声和网络
#netg, netd = model.NetG().eval(), model.NetD().eval()
noises = pt.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
noises = pt.autograd.Variable(noises, volatile = True)
# 加载预训练模型
netd = pt.load(opt.netd_path, map_location=lambda storage, loc: storage)
netg = pt.load(opt.netg_path, map_location=lambda storage, loc: storage)
# 是否使用GPU
'''
if opt.use_gpu:
	netd.cuda()
	netg.cuda()
	noises = noises.cuda()
'''
# 生成图片，并计算图片在判别器的分数
fake_img = netg(noises)
scores = netd(fake_img).data
# 挑选最好的某几张
indexs = scores.topk(opt.gen_num)[1]
result = []
for ii in indexs:
	result.append(fake_img.data[ii])
# 保存图片
tv.utils.save_image(pt.stack(result), opt.gen_img, normalize = True, range = (-1, 1))

'''
# 可视化(涉及到visdom)
fix_noises = pt.autograd.Variable(pt.randn(opt.batch_size, opt.nz, 1, 1)).cuda()
fix_fake_imgs = netg(fix_noises)
vis.images(fix_fake_imgs.data.cpu().numpy()[:64])
'''