#脚本中的代码 y = model() 不能直接运行，因为它需要有效的输入数据。
#需要创建一个合适大小的输入张量，并将其传递给模型。例如：
# x = torch.randn(1, 3, 96, 96)  # 随机生成输入数据，这里的尺寸需要与模型输入相匹配
# y = model(x)  # 获取模型输出

#用于生成并显示 Wav2Lip 模型的计算图。使用 torchviz 库中的 make_dot 函数创建模型的计算图，并将其保存为一个 PDF 文件。以下是脚本的详细解释和运行说明：
#创建 Wav2Lip 模型的实例：model = Wav2Lip()。
#生成模型的计算图：g = make_dot(y)，其中 y 是模型的输出。这里有一个问题，因为 y 没有被定义，你需要传入模型的输入数据以获得输出 y。否则，脚本将无法正常运行。
#保存和/或查看计算图：g.render('Wav2Lip', view=False) 将计算图保存为名为 "Wav2Lip.pdf" 的文件。如果将 view 参数设置为 True，则在保存后自动打开该 PDF 文件。

import torch
from  .wav2lip import Wav2Lip
from torchviz import make_dot
 
# x=torch.rand(8,3,256,512)
model=Wav2Lip()
y=model()
# 这三种方式都可以
g = make_dot(y)
# g=make_dot(y, params=dict(model.named_parameters()))
#g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
# 这两种方法都可以
# g.view() # 会生成一个 Digraph.gv.pdf 的PDF文件
g.render('Wav2Lip', view=False) # 会自动保存为一个 espnet.pdf，第二个参数为True,则会自动打开该PDF文件，为False则不打开