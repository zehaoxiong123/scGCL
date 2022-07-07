import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set()
plt.rcParams['font.sans-serif']='SimHei'#设置中文显示，必须放在sns.set之后
np.random.seed(0)
max_lst_of_all = {}
max_lst_of_all["Zeisel"] = [0.4309,0.9078,0.6999,0.8581]
max_lst_of_all["Klein"] = [0.2495, 0.5480,0.4380,0.5397]
#---------------------------------------------------------------------
# max_lst_of_all["Adam"] = [0.6762, 0.5674, 0.5384, 0.7278, 0.8441, 0.9067]
# max_lst_of_all["Muraro"] = [0.7661, 0.9396, 0.8813, 0.9045, 0.8975, 0.9164]
# max_lst_of_all["Plasschaert"] = [0.4196, 0.5157, 0.4272, 0.4778, 0.7398, 0.5957]
# max_lst_of_all["Quake_10x_Bladder"] = [0.7573, 0.7532, 0.9813, 0.7568, 0.4770, 0.7529]
# max_lst_of_all["Quake_10x_Limb_Muscle"] = [0.9400, 0.7498, 0.9751, 0.9893, 0.9471, 0.9920]
# max_lst_of_all["Quake_10x_Spleen"] = [0.5268, 0.5037, 0.5322, 0.5056, 0.6916, 0.9371]
# max_lst_of_all["Quake_Smart-seq2_Diaphragm"] = [0.9586, 0.9747, 0.5380, 0.9626, 0, 0.9893]
# max_lst_of_all["Quake_Smart-seq2_Limb_Muscle"] = [0.6875, 0.6610, 0.6639, 0.9730, 0.9661, 0.9444]
# max_lst_of_all["Quake_Smart-seq2_Lung"] = [0.4922, 0.4682, 0.5534, 0.5298, 0.6147, 0.4980]
# max_lst_of_all["Quake_Smart-seq2_Trachea"] = [0.5965, 0.9044, 0.8607, 0.5886, 0.7466, 0.5862]
# max_lst_of_all["Romanov"] = [0.5966, 0.5177, 0.6639, 0.6580, 0.6634, 0.6944]
# max_lst_of_all["Wang_lung"] = [0.9231, 0.9690, 0.9296, 0.9370, 0.7127, 0.9396]
# max_lst_of_all["Young"] = [0.5778, 0.5224, 0.6420, 0.6266, 0.6488, 0.7882]
# max_lst_of_all["Alzheimer"] = [0.3510, 0.4320, 0.5762, 0.3846, 0.6648, 0.6197]
#___________________________________________________________________________
# max_lst_of_all["Adam"] = [0.7552, 0.7012, 0.6279, 0.7964, 0.8455, 0.8927]
# max_lst_of_all["Muraro"] = [0.8399, 0.9003, 0.8689, 0.8854, 0.8581, 0.8773]
# max_lst_of_all["Plasschaert"] = [0.6236, 0.6602, 0.6257, 0.6646, 0.7392, 0.7154]
# max_lst_of_all["Quake_10x_Bladder"] = [0.8081, 0.7920, 0.9460, 0.8075, 0.6169, 0.8063]
# max_lst_of_all["Quake_10x_Limb_Muscle"] = [0.9436, 0.8473, 0.9478, 0.9745, 0.9357, 0.9803]
# max_lst_of_all["Quake_10x_Spleen"] = [0.6762, 0.6749, 0.5080, 0.6860, 0.6145, 0.8661]
# max_lst_of_all["Quake_Smart-seq2_Diaphragm"] = [0.9607, 0.9541, 0.6923, 0.9457, 0, 0.9772]
# max_lst_of_all["Quake_Smart-seq2_Limb_Muscle"] = [0.8284, 0.8263, 0.8236, 0.9551, 0.9416, 0.9452]
# max_lst_of_all["Quake_Smart-seq2_Lung"] = [0.7406, 0.7293, 0.7317, 0.7493, 0.7500, 0.7531]
# max_lst_of_all["Quake_Smart-seq2_Trachea"] = [0.7043,0.8830, 0.8207, 0.7134, 0.6584, 0.7122]
# max_lst_of_all["Romanov"] = [0.6547, 0.5576, 0.6761, 0.6802, 0.6895, 0.7159]
# max_lst_of_all["Wang_lung"] = [0.8606, 0.9236, 0.8595, 0.8676, 0.6380, 0.8703]
# max_lst_of_all["Young"] = [0.7152, 0.6510,0.7499, 0.7532, 0.7627, 0.8450]
# max_lst_of_all["Alzheimer"] = [0.5911, 0.6763, 0.6628, 0.6013, 0.6542, 0.7004]
labels = ['scAFGRL','AutoClass', 'GraphSCI', 'MAGIC']
indicater = ["Zeisel","Klein"]
datas = []
for i in range(len(labels)):
    function_list = []
    for key in max_lst_of_all.keys():
        function_list.append(max_lst_of_all[key][i])
    datas.append(function_list)
 #设置二维矩阵
f, ax = plt.subplots(figsize=(4, 6))

#heatmap后第一个参数是显示值,vmin和vmax可设置右侧刻度条的范围,
#参数annot=True表示在对应模块中注释值
# 参数linewidths是控制网格间间隔
#参数cbar是否显示右侧颜色条，默认显示，设置为None时不显示
#参数cmap可调控热图颜色，具体颜色种类参考：https://blog.csdn.net/ztf312/article/details/102474190
df = pd.DataFrame(datas,columns=indicater,index=labels)
sns.heatmap(df, ax=ax,vmin=0,vmax=1,cmap='YlGn',annot=True,linewidths=2,cbar=True)

ax.set_title('Median L1 distance') #plt.title('热图'),均可设置图片标题
ax.set_ylabel('compare_function')  #设置纵轴标签
ax.set_xlabel('data_set')  #设置横轴标签

#设置坐标字体方向，通过rotation参数可以调节旋转角度
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360, horizontalalignment='right')
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=45, horizontalalignment='right')
plt.savefig("./figures/L1_distance_0.5.png",dpi=600)
plt.show()
