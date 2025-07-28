import matplotlib.pyplot as plt
import numpy as np
plt.rc('font',family='Times New Roman')

#################################气泡图
Param = [0.01, 92.61, 8.17, 30.62, 37.30, 15.50, 52.99, 5.54,61.75,38.55,1.81,36.62,3.27,134.39,72.06,44.48,22.02,65.02]
FLOPs = [0.0068, 195.60, 2.60, 261.92, 80.04, 199.67, 1497.10, 137.89,505.18,40.25,0.74,34.64,1.00,22.39,14.33,3.04,2.50,2.62]
Time =  [1.837, 3.270, 1.608, 2.898, 2.846, 2.587, 3.119, 2.046,3.202,1.275,0.077,0.810,2.241,4.115,2.469,0.779,0.520,1.203]

models =['nnUNet','UNETR', 'TransUNet', 'TransBTS', 'nnFormer', 'SwinUNETR',
          '3D-DX-Net', 'MedNeXt-S','MedNeXt-L','ConvFormer','DCNet','MLB-Seg','UPCoL',
         'ASSNet','MSA²Net','M⁴oE','VPTTA','SR-DM SegNet']

# 'nnUNet',
# Params= 0.016684M
# FLOPs= 0.00687616G
# 程序运行时间为： 2.3378210067749023 秒
# 'UNETR',
#Params= 92.617937M
# FLOPs= 195.606085632G
# 程序运行时间为： 3.2705206871032715 秒
# 'TransUNet',
#Params= 8.174545M
# FLOPs= 2.609381376G
# 程序运行时间为： 1.6084833145141602 秒
# 'TransBTS',
#Params= 30.624053M
# FLOPs= 261.927993344G
# 程序运行时间为： 2.8987619876861572 秒
# 'nnFormer',
# Params= 37.306417M
# FLOPs= 80.040895488G
# 程序运行时间为： 2.84660267829895 秒
# 'SwinUNETR',
#Params= 15.505249M
# FLOPs= 199.677949488G
# 程序运行时间为： 2.5875372886657715 秒
# '3D-DX-Net',
#Params= 52.999345M
# FLOPs= 1497.102680064G
# 程序运行时间为： 3.1190907955169678 秒
# 'MedNeXt-S',
#Params= 5.542113M
# FLOPs= 137.890464448G
# 程序运行时间为： 2.546459436416626 秒
# 'MedNeXt-L',
#Params= 61.754465M
# FLOPs= 505.18449632G
# 程序运行时间为： 3.2029120922088623 秒
# 'ConvFormer',
#Params= 38.556801M
# FLOPs= 40.257552384G
# 程序运行时间为： 0.8754274845123291 秒
# 'DCNet',
#Params= 1.813329M
# FLOPs= 0.743735296G
# 程序运行时间为： 0.07770633697509766 秒
# 'MLB-Seg',
#Params= 36.628481M
# FLOPs= 34.64626176G
# 程序运行时间为： 0.8100616931915283 秒
# 'UPCoL',
# Params= 3.270355M
# FLOPs= 1.00728832G
# 程序运行时间为： 2.241441488265991 秒
# 'ASSNet',
#Params= 134.397127M
# FLOPs= 22.39734896G
# 程序运行时间为： 4.11567234992981 秒
# 'MSA²Net',
#Params= 72.063195M
# FLOPs= 14.337668704G
# 程序运行时间为： 2.46932053565979 秒
# 'M⁴oE',
#Params= 44.482272M
# FLOPs= 3.042280704G
# 程序运行时间为： 0.7791416645050049 秒
# 'VPTTA',
#Params= 22.022401M
# FLOPs= 2.503213056G
# 程序运行时间为： 0.5208375453948975 秒
# 'UMF-SegNet'
# Params= 148.282273M
# FLOPs= 372.748940856G
# 程序运行时间为： 5.271911382675171 秒



colors = ['darkorange', 'sienna', 'darkorange', 'green', 'olivedrab', 'red', 'cyan', 'dodgerblue','blue','purple', 'magenta', 'darkviolet', 'lightcoral', 'red','gray', 'sienna', 'darkorange', 'green']
# ,
#           'purple', 'magenta', 'darkviolet', 'lightcoral', 'red']
# models =['nnUNet','UNETR', 'TransUNet', 'TransBTS', 'nnFormer', 'SwinUNETR',
#           '3D-DX-Net', 'MedNeXt-S','MedNeXt-L','ConvFormer','DCNet','MLB-Seg','UPCoL',
#          'ASSNet','MSA²Net','M⁴oE','VPTTA','UMF-SegNet']

# scatter1=plt.scatter(FLOPs,Time,s=100*np.array(Param),c=colors,alpha=0.35)
scatter=plt.scatter(FLOPs,Time,s=5*np.array(Param),c=colors,alpha=0.35)
h,l=scatter.legend_elements(prop='sizes', num=5)
l=['$\\mathdefault{20}$', '$\\mathdefault{40}$', '$\\mathdefault{60}$', '$\\mathdefault{90}$']
print(l)
plt.legend(h,l, loc=0, ncol=5)

id = 0
# models =['' '', ,
#           , '',
#           , '',,
#           '',,'MLB-Seg','',
#          ,'','M⁴oE','',]

for x, y, m in zip(FLOPs, Time, models):
    if m == 'SR-DM SegNet':
        plt.text(x+40, y-0.13, m, ha='left', va='center',fontsize=9, weight='bold')
    elif m == 'ASSNet':
        plt.text(x+80, y, m, ha='left', va='center',fontsize=9)
    elif m == 'MedNeXt-L':
        plt.text(x+60, y , m, ha='left', va='center', fontsize=9)
    elif m == 'UNETR':
        plt.text(x+60, y , m, ha='left', va='center',fontsize=9)
    elif m == '3D-DX-Net':
        plt.text(x-280, y, m, ha='left', va='center',fontsize=9)
    elif m == 'TransBTS':
        plt.text(x+50, y, m, ha='left', va='center',fontsize=9)
    elif m == 'nnFormer':
        plt.text(x, y+0.19, m, ha='center', va='center',fontsize=9)
    elif m == 'DCNet':
        plt.text(x+20, y, m, ha='left', va='center', fontsize=9)
    elif m == 'VPTTA':
        plt.text(x+40, y, m, ha='left', va='center', fontsize=9)
    elif m == 'TransUNet':
        plt.text(x+20, y, m, ha='left', va='center', fontsize=9)
    elif m == 'UPCoL':
        plt.text(x, y-0.15, m, ha='center', va='center', fontsize=9)
    elif m == 'SwinUNETR':
        plt.text(x+40, y, m, ha='left', va='center', fontsize=9)
    elif m == 'MedNeXt-S':
        plt.text(x+20, y, m, ha='left', va='center', fontsize=9)
    elif m == 'ConvFormer':
        plt.text(x+160, y, m, ha='center', va='center', fontsize=9)
    elif m == 'MSA²Net':
        plt.text(x+50, y-0.1, m, ha='left', va='center', fontsize=9)
    elif m == 'nnUNet':
        plt.text(x+20, y, m, ha='left', va='center', fontsize=9)
    elif m == 'MLB-Seg':
        plt.text(x+50, y, m, ha='left', va='center', fontsize=9)
    elif m == 'M⁴oE':
        plt.text(x-130, y, m, ha='left', va='center', fontsize=9)

    else:
        plt.text(x, y, m, ha='center', va='center',fontsize=9)

# for i,j in zip(FLOPs,Time):
#     if models[id] == 'F3Net':
#         plt.annotate(str(models[id]), xy=(i,j-0.01),fontsize=10)#,xytext = (+1,-1)
#         id = id + 1
#     elif models[id] == 'ABiU-Net':
#         plt.annotate(str(models[id]), xy=(i-15,j),fontsize=11, weight='bold',c='red')
#         id = id + 1
#     else:
#         plt.annotate(str(models[id]), xy=(i,j),fontsize=10)#,xytext = (+1,-1)
#         id = id + 1

font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 12 }

plt.yticks(fontproperties='Times New Roman', size=10)#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("FLOPs (G)", fontdict={'family': 'Times New Roman', 'weight': 'bold','size': 12})
plt.ylabel('Time (s)', fontdict={'family': 'Times New Roman','weight': 'bold','size': 12})
plt.subplots_adjust(bottom=0.18)
plt.savefig('/home/ubuntu/Luhaotian/Paper-pre/muti-model_fusion/code/myproject/tool/complex.pdf', dpi=300)
plt.show()

################################# 折线图
# main = ['TE', 'TED', 'TE+HED', 'TED+HED', 'TED+HED+DS', 'TED+DS', 'HED+DS']
#
# Fb_SOD = [0.795, 0.868, 0.870, 0.878, 0.879, 0.872, 0.799]
# Fb_HKUIS = [0.853, 0.943, 0.946, 0.948, 0.951, 0.941, 0.885]
# Fb_ECSSD = [0.881, 0.953, 0.954, 0.956, 0.959, 0.955, 0.905]
# Fb_OMRON = [0.753, 0.832, 0.836, 0.840, 0.843, 0.836, 0.755]
# Fb_THUR15K = [0.740, 0.812, 0.817, 0.817, 0.820, 0.809, 0.756]
# Fb_DUTS = [0.787, 0.894, 0.902, 0.905, 0.906, 0.893, 0.784]
#
# plt.rc('font',family='Times New Roman')
# plt.plot(main, Fb_SOD, c='dodgerblue',  linestyle='-', linewidth='3', label = "SOD")
# plt.plot(main, Fb_HKUIS, c='green',  linestyle='-', linewidth='3',  label = "HKU-IS")
# plt.plot(main, Fb_ECSSD, c='red',  linestyle='-', linewidth='3',  label = "ECSSD")
# plt.plot(main, Fb_OMRON, c='gold',  linestyle='-', linewidth='3',  label = "DUT-OMRON")
# plt.plot(main, Fb_THUR15K, c='darkorchid',  linestyle='-', linewidth='3',  label = "THUR15K")
# plt.plot(main, Fb_DUTS, c='darkorange',  linestyle='-', linewidth='3',  label = "DUTS-TE")
#
# plt.scatter(main,  Fb_SOD, c='dodgerblue', marker='o')
# plt.scatter(main,  Fb_HKUIS, c='green', marker='o')
# plt.scatter(main,  Fb_ECSSD, c='red', marker='o')
# plt.scatter(main,  Fb_OMRON, c='gold', marker='o')
# plt.scatter(main,  Fb_THUR15K, c='darkorchid', marker='o')
# plt.scatter(main,  Fb_DUTS, c='darkorange', marker='o')
#
# for i,j in zip(main,Fb_SOD):
#     plt.annotate(str(j), xy=(i,j),fontsize=12)#,xytext = (+1,-1)
# for i,j in zip(main,Fb_HKUIS):
#     plt.annotate(str(j), xy=(i,j),fontsize=12)
# for i,j in zip(main,Fb_ECSSD):
#     plt.annotate(str(j), xy=(i,j),fontsize=12)
# for i,j in zip(main,Fb_OMRON):
#     plt.annotate(str(j), xy=(i,j),fontsize=12)
# for i,j in zip(main,Fb_THUR15K):
#      if i == 'HED+DS':
#         plt.annotate(str(j), xy=(i,j),xytext=(i, j-0.0085),fontsize=12)
#      else:
#         plt.annotate(str(j), xy=(i,j),fontsize=12)
# for i,j in zip(main,Fb_DUTS):
#     plt.annotate(str(j), xy=(i,j),fontsize=12)
#
#
# # plt.rc('font',family='Times New Roman')
# # MAE_SOD = [0.147, 0.102, 0.099, 0.096, 0.089, 0.096, 0.136]
# # MAE_HKUIS = [0.090, 0.030, 0.029, 0.024, 0.021, 0.029, 0.050]
# # MAE_ECSSD = [0.090, 0.037, 0.036, 0.028, 0.026, 0.031, 0.055]
# # MAE_OMRON = [0.095, 0.051, 0.047, 0.046, 0.043, 0.048, 0.071]
# # MAE_THUR15K = [0.111, 0.066, 0.065, 0.061, 0.059, 0.066, 0.081]
# # MAE_DUTS = [0.086, 0.033, 0.034, 0.031, 0.029, 0.034, 0.069]
# #
# # plt.plot(main, MAE_SOD, c='dodgerblue',  linestyle='-', linewidth='3', label = "SOD")
# # plt.plot(main, MAE_HKUIS, c='green',  linestyle='-', linewidth='3',  label = "HKU-IS")
# # plt.plot(main, MAE_ECSSD, c='red',  linestyle='-', linewidth='3',  label = "ECSSD")
# # plt.plot(main, MAE_OMRON, c='gold',  linestyle='-', linewidth='3',  label = "DUT-OMRON")
# # plt.plot(main, MAE_THUR15K, c='darkorchid',  linestyle='-', linewidth='3',  label = "THUR15K")
# # plt.plot(main, MAE_DUTS, c='darkorange',  linestyle='-', linewidth='3',  label = "DUTS-TE")
# #
# # plt.scatter(main,  MAE_SOD, c='dodgerblue', marker='o')
# # plt.scatter(main,  MAE_HKUIS, c='green', marker='o')
# # plt.scatter(main,  MAE_ECSSD, c='red', marker='o')
# # plt.scatter(main,  MAE_OMRON, c='gold', marker='o')
# # plt.scatter(main,  MAE_THUR15K, c='darkorchid', marker='o')
# # plt.scatter(main,  MAE_DUTS, c='darkorange', marker='o')
# #
# # for i,j in zip(main,MAE_SOD):
# #     plt.annotate(str(j), xy=(i,j),xytext=(i, j+0.002), fontsize=12)
# # for i,j in zip(main,MAE_HKUIS):
# #     if i == 'TE':
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j-0.007),fontsize=12)
# #     elif i == 'TED':
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j-0.004),fontsize=12)
# #     elif i == 'TE+HED':
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j-0.005),fontsize=12)
# #     # elif i == 'TED+HED+DS':
# #     #     plt.annotate(str(j), xy=(i,j),xytext=(i, j+0.003),fontsize=12)
# #     else:
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j-0.007),fontsize=12) #xytext=(+0,-0.01)
# # for i,j in zip(main,MAE_ECSSD):
# #     if i == 'TE':
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j),fontsize=12)
# #     elif i == 'TED+HED':
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j-0.002),fontsize=12)
# #     elif i == 'HED+DS':
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j-0.001),fontsize=12)
# #     elif i == 'TED+HED+DS':
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j-0.002),fontsize=12)
# #     else:
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j+0.002),fontsize=12)
# # for i,j in zip(main,MAE_OMRON):
# #     if i == 'TE':
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j+0.002),fontsize=12)
# #     else:
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j),fontsize=12)
# # for i,j in zip(main,MAE_THUR15K):
# #     plt.annotate(str(j), xy=(i,j),xytext=(i, j+0.003),fontsize=12)
# # for i,j in zip(main,MAE_DUTS):
# #     if i == 'TED':
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j+0.000),fontsize=12)
# #     elif i == 'TE+HED':
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j-0.003),fontsize=12)
# #     elif i == 'TED+HED+DS':
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j+0.003),fontsize=12)
# #     elif i == 'TED+HED':
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j+0.003),fontsize=12)
# #     elif i == 'TED+DS':
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j+0.005),fontsize=12)
# #     else:
# #         plt.annotate(str(j), xy=(i,j),xytext=(i, j-0.009),fontsize=12)
#
# font = {'family': 'Times New Roman',
#             'weight': 'normal',
#             'size': 12 }
#
# plt.legend(loc='best')
#
#
# plt.yticks(fontproperties='Times New Roman', size=10)#设置大小及加粗
# plt.xticks(fontproperties='Times New Roman', size=10, rotation=10)
# plt.grid(True, linestyle='--', alpha=0.5)
# #plt.xlabel("Main Components of ABiU-Net", fontdict={'family': 'Times New Roman', 'weight': 'bold','size': 12})
# plt.ylabel('F'+r'$\beta$', fontdict={'family': 'Times New Roman','weight': 'bold','size': 12})
# #plt.ylabel('MAE', fontdict={'family': 'Times New Roman','weight': 'bold','size': 12})
#
# # plt.ylabel('F'+r'$\beta$', fontdict={'size': 16},weight='bold')
# # plt.savefig('fb.jpg', dpi=300)
# plt.subplots_adjust(bottom=0.18)
# plt.savefig('fb.jpg', dpi=300)