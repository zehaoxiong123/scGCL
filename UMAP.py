from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import umap
import numpy as np
import cluster
from  matplotlib.colors import  rgb2hex
import data_Preprocess
from sklearn.manifold import TSNE
import compare
import draw_heatmap
import result_class
import pickle
import pandas as pd
import utils
import h5py

def test():
    digits = load_digits()
    fig, ax_array = plt.subplots(20, 20)
    axes = ax_array.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(digits.images[i], cmap='gray_r')
    plt.setp(axes, xticks=[], yticks=[], frame_on=False)
    plt.tight_layout(h_pad=0.5, w_pad=0.01)
    plt.show()
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(digits.data)
    print(embedding.shape)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
    print(digits.target)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('UMAP projection of the Digits dataset')
    plt.show()

def tsne_for_data(data,function_name,kind_name,cell_class):
    data_class = np.shape(data[0])[1]
    # cmap = plt.get_cmap('viridis', data_class)
    embedding = TSNE(random_state=50).fit_transform(data[1])
    y_true = np.argmax(data[0], axis=1)
    fig, ax = plt.subplots()
    cmap = plt.cm.Spectral
    norm = plt.Normalize(vmin=0, vmax=data_class)
    for i in range(data_class):
        need_idx = np.where(y_true==i)[0]
        ax.scatter(embedding[need_idx, 0], embedding[need_idx, 1], c=cmap(norm(i)), s=cell_class,label = kind_name[i])

    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel('tsne_1')
    plt.ylabel('tsne_2')
    plt.title(function_name+' projection of the dataset')
    legend = ax.legend(loc='upper right',framealpha=0.5)
    plt.show()
    return embedding

def umap_for_data(data,function_name,kind_name,cell_class,data_set_name):
    data_class = np.shape(data[0])[1]
    #cmap = plt.get_cmap('viridis', data_class)
    reducer = umap.UMAP(random_state= 20150101)
    embedding = reducer.fit_transform(data[1])
    y_true = np.argmax(data[0], axis=1)
    fig, ax = plt.subplots()
    cmap = plt.cm.Spectral
    norm = plt.Normalize(vmin=0, vmax=data_class)
    for i in range(data_class):
        need_idx = np.where(y_true==i)[0]
        ax.scatter(embedding[need_idx, 0], embedding[need_idx, 1], c=cmap(norm(i)), s=0.1,label = kind_name[i])

    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel('UMAP_1')
    plt.ylabel('UMAP_2')
    plt.title(function_name+' projection of '+str(data_set_name))
    legend = ax.legend(loc='upper right',fontsize =5)
    plt.savefig("./compare_funtion/sagan/result/" + data_set_name + "/" + str(function_name) + ".png")
    plt.show()
    return embedding

def umap_for_all(data,function_list,cell_type,data_set_name):
    data_class = np.shape(data[0].data_label)[1]
    embedding_list = []
    label_list = []
    reducer = umap.UMAP(random_state=20150101)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.xticks(fontsize=10)
    fig = plt.figure(figsize=(12,15), dpi=800)
    plt.xlabel('UMAP_1')
    plt.ylabel('UMAP_2')
    plt.axis('off')
    for i in range(len(function_list)):
        embedding = reducer.fit_transform(data[i].data_mat)
        y_true = np.argmax(data[i].data_label,axis=1)
        embedding_list.append(embedding)
        label_1 = cluster.k_means(embedding, data[i].cell_type_num)
        label_list.append(label_1)
        ax = fig.add_subplot(5,2,i+1)
        cmap = plt.cm.tab20
        norm = plt.Normalize(vmin=0, vmax=data_class)
        for j in range(data_class):
            need_idx = np.where(y_true == j)[0]
            ax.scatter(embedding[need_idx, 0], embedding[need_idx, 1], c=cmap(norm(j)), s=1, label=cell_type[j])
            plt.gca().set_aspect('equal', 'datalim')
            plt.title(function_list[i],fontsize =15)

    plt.legend(bbox_to_anchor=(1.20, 0.3),fontsize =20, loc='center left');
    plt.savefig("./compare_funtion/sagan/result/" + data_set_name + "/" + str(data_set_name) + "p_compare.png")
    plt.show()
    return embedding_list,label_list

def umap_for_all_batch(data,function_list,cell_type,data_set_name):
    data_class = np.shape(data[0].data_label)[1]
    batch_class = np.shape(data[0].batch_type)[0]
    batch_type = data[0].batch_type
    embedding_list = []
    label_list = []
    reducer = umap.UMAP(random_state=20150101)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.xticks(fontsize=10)
    fig = plt.figure(figsize=(36, 12), dpi=600)
    plt.xlabel('UMAP_1')
    plt.ylabel('UMAP_2')
    plt.axis('off')
    index = 1
    for i in range(len(function_list)):
        embedding = reducer.fit_transform(data[i].data_mat)
        y_true = np.argmax(data[i].data_label, axis=1)
        batch_true = data[i].batch_label
        embedding_list.append(embedding)
        ax = fig.add_subplot(2, 4, index)
        cmap = plt.cm.winter
        norm = plt.Normalize(vmin=0, vmax=batch_class)
        for j in range(batch_class):
            need_idx = np.where(batch_true == j)[0]
            ax.scatter(embedding[need_idx, 0], embedding[need_idx, 1], c=cmap(norm(j)), s=1, label=batch_type[j])
            plt.legend(bbox_to_anchor=(1.05, 0.5),fontsize=25, loc='center left');
            plt.gca().set_aspect('equal', 'datalim')
            plt.title(function_list[i],fontsize =30)

        ax = fig.add_subplot(2, 4, index+2)
        cmap = plt.cm.tab20b
        norm = plt.Normalize(vmin=0, vmax=data_class)
        for j in range(data_class):
            need_idx = np.where(y_true == j)[0]
            ax.scatter(embedding[need_idx, 0], embedding[need_idx, 1], c=cmap(norm(j)), s=1, label=cell_type[j])
            plt.legend(bbox_to_anchor=(1.05, 0.5),fontsize=25, loc='center left');
            plt.gca().set_aspect('equal', 'datalim')
            plt.title(function_list[i],fontsize =30)
        index = index+4
    plt.savefig("./compare_funtion/sagan/result/" + data_set_name + "/" + str(data_set_name) + "_batch.png")
    plt.show()
    return embedding_list, label_list





def umap_for_interpretable(data,function_list,cell_type,data_set_name,type):

    embedding_list = []
    label_list = []
    reducer = umap.UMAP(random_state=20150101)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.xticks(fontsize=10)
    fig = plt.figure(figsize=(16,6), dpi=350)
    plt.xlabel('UMAP_1')
    plt.ylabel('UMAP_2')
    plt.axis('off')
    for i in range(len(function_list)):
        data_class = np.shape(data[i].data_label)[1]
        embedding = reducer.fit_transform(data[i].data_mat)
        y_true = np.argmax(data[i].data_label,axis=1)
        embedding_list.append(embedding)
        label_1 = cluster.k_means(embedding, data[i].cell_type_num)
        label_list.append(label_1)
        ax = fig.add_subplot(1,2,i+1)
        cmap = plt.cm.tab20c
        norm = plt.Normalize(vmin=0, vmax=data_class)
        for j in range(data_class):
            need_idx = np.where(y_true == j)[0]
            print(need_idx.shape)
            if type == "interpretable":
                ax.scatter(embedding[need_idx, 0], embedding[need_idx, 1], c=cmap(norm(j)), s=1, label=cell_type[j])
            else:
                ax.scatter(embedding[need_idx, 0], embedding[need_idx, 1], c=cmap(norm(j)), s=1, label=cell_type[i][j])
                plt.legend(bbox_to_anchor=(1.05, 0.5), fontsize=15, loc='center left');

            plt.gca().set_aspect('equal', 'datalim')
            plt.title(function_list[i])
    plt.legend(bbox_to_anchor=(1.05, 0.5), fontsize=15, loc='center left');
    if type == "interpretable":
        plt.savefig("./compare_funtion/sagan/result/" + data_set_name + "/" + str(data_set_name) + "__batch_compare.png")
    else:
        plt.savefig("./compare_funtion/sagan/result/lung_cancer/lung_cancer_compare.png")
    plt.show()
    return embedding_list,label_list




def umap_for_z_bar_lung_cancer(result_list,find_gene_1,find_gene_2, funtion_name, result_z_orign, result_z_new,result_z_orign_,result_z_new_, data_set_name):
    cm_1 = plt.cm.get_cmap('RdPu')
    cm_2 = plt.cm.get_cmap('BuGn')
    data_num = np.shape(result_list[1].data_mat)[0]
    # cmap = plt.get_cmap('viridis', data_class)
    reducer = umap.UMAP(random_state=20150101)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.xticks(fontsize=10)
    fig = plt.figure(figsize=(24,48), dpi=600)
    plt.xlabel('UMAP_1')
    plt.ylabel('UMAP_2')
    plt.axis('off')
    index = 1
    for j in range(len(find_gene_1)):
        embedding = reducer.fit_transform(result_list[0].data_mat)
        ax = fig.add_subplot(8, 2, index)
        for i in range(data_num):
            sc = ax.scatter(embedding[i, 0], embedding[i, 1], c=result_z_orign[j][i], vmin=0, vmax=1, s=1, cmap=cm_1)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(find_gene_1[j]+"_before", fontsize=15)
        plt.colorbar(sc)
        embedding_ = reducer.fit_transform(result_list[1].data_mat)
        ax = fig.add_subplot(8, 2, index+1)
        for i in range(data_num):
            sc = ax.scatter(embedding_[i, 0], embedding_[i, 1], c=result_z_new[j][i], vmin=0, vmax=1, s=1, cmap=cm_1)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(find_gene_1[j]+"_now", fontsize=15)
        plt.colorbar(sc)
        index = index+2
        print(index)

    for j in range(len(find_gene_2)):
        embedding = reducer.fit_transform(result_list[0].data_mat)
        ax = fig.add_subplot(8, 2, index)
        for i in range(data_num):
            sc = ax.scatter(embedding[i, 0], embedding[i, 1], c=result_z_orign_[j][i], vmin=0, vmax=1, s=1, cmap=cm_2)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(find_gene_2[j] + "_before", fontsize=15)
        plt.colorbar(sc)
        embedding_ = reducer.fit_transform(result_list[1].data_mat)
        ax = fig.add_subplot(8, 2, index + 1)
        for i in range(data_num):
            sc = ax.scatter(embedding_[i, 0], embedding_[i, 1], c=result_z_new_[j][i], vmin=0, vmax=1, s=1, cmap=cm_2)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(find_gene_2[j] + "_now", fontsize=15)
        plt.colorbar(sc)
        index = index + 2
        print(index)
    plt.savefig("./compare_funtion/sagan/result/" + data_set_name + "/gene_compare.png")
    plt.show()

def umap_for_z_bar(result_list, find_gene,funtion_name,result_z_orign,result_z_new, data_set_name):
    cm = plt.cm.get_cmap('PuBu')
    data_num = np.shape(result_list[1].data_mat)[0]
    # cmap = plt.get_cmap('viridis', data_class)
    reducer = umap.UMAP(random_state=20150101)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.xticks(fontsize=10)
    fig = plt.figure(figsize=(24,6*len(find_gene)), dpi=600)
    plt.xlabel('UMAP_1')
    plt.ylabel('UMAP_2')
    plt.axis('off')
    index = 1
    for j in range(len(find_gene)):
        embedding = reducer.fit_transform(result_list[0].data_mat)
        ax = fig.add_subplot(len(find_gene), 2, index)
        for i in range(data_num):
            sc = ax.scatter(embedding[i, 0], embedding[i, 1], c=result_z_orign[j][i], vmin=0, vmax=1, s=1, cmap=cm)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(find_gene[j]+"_before",fontsize=15)
        plt.colorbar(sc)
        print(index)
        embedding_ = reducer.fit_transform(result_list[1].data_mat)
        ax = fig.add_subplot(len(find_gene), 2, index+1)
        for i in range(data_num):
            sc = ax.scatter(embedding_[i, 0], embedding_[i, 1], c=result_z_new[j][i], vmin=0, vmax=1, s=1, cmap=cm)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(find_gene[j]+"_now",fontsize=15)
        plt.colorbar(sc)
        index = index+2
    plt.savefig("./compare_funtion/sagan/result/" + data_set_name + "/gene_compare.png")
    plt.show()

def show_result_h5_batch(data_set_name,compare_list,compare_function_list,label_file):
    result_list = []
    for i in range(len(compare_list)):
        mat, data_label, cell_type, cell_type_num, obs, var,batch_type,batch_label = data_Preprocess.read_cell_for_h5_imputed(
            compare_list[i], label_file)
        compare_result = result_class.result_for_impute(compare_function_list[i], mat, data_label, cell_type,
                                                        cell_type_num, obs, var,batch_type,batch_label)
        result_list.append(compare_result)
    embeddings_list, label_list = umap_for_all_batch(result_list, compare_function_list, cell_type, data_set_name)







def show_result_h5(data_set_name,compare_list,compare_function_list,label_file):
    # mat, data_label, cell_type, cell_type_num, obs, var =data_Preprocess.read_cell_for_h5("./compare_funtion/sagan/test_csv/Adam/data.h5")
    result_list = []
    # label_list = []
    # embeddings_list = []
    for i in range(len(compare_list)):
        mat, data_label, cell_type, cell_type_num, obs, var,_,_ = data_Preprocess.read_cell_for_h5_imputed(
            compare_list[i], label_file)
        compare_result = result_class.result_for_impute(compare_function_list[i], mat, data_label, cell_type,
                                                        cell_type_num, obs, var)
        result_list.append(compare_result)
    # 通过umap算法把维度降到二维,输入应该是标签在前，数组在后
    # embeddings = umap_for_data((data_label,data_array),"drop out",["Group1","Group2","Group3","Group4"],4)
    # for i in range(len(compare_list)):
    #     embeddings = umap_for_data((result_list[i].data_label, result_list[i].data_mat), compare_function_list[i],
    #                                cell_type, cell_type_num,data_set_name)
    #     embeddings_list.append(embeddings)
    #
    #     label_1 = cluster.k_means(embeddings, result_list[i].cell_type_num)
    #     label_list.append(label_1)
    embeddings_list,label_list = umap_for_all(result_list,compare_function_list,cell_type,data_set_name)
    # 在通过聚类算法对指标进行测评
    label = np.array(label_list)
    # 对function_name进行赋值
    function_name = compare_function_list
    # 通过各种指标对聚类结果评测
    # AUC = AUC.compare_for_auc(function_name, data_label, label_score)
    compare.get_indicator(function_name, data_label, label, data_set_name, embeddings_list)
    # draw_heatmap.draw_heatmap(var,obs,mat)
    #data_array,data_label,gene_name,cell_name,data_for_count=data_Preprocess.read_cell_to_image("./test_csv/splatter_exprSet_test.csv", "./test_csv/splatter_exprSet_test_label.csv", 4)

# def show_result_Klein(compare_list,label_list,compare_function_list,data_set_name):
#     # mat, data_label, cell_type, cell_type_num, obs, var =data_Preprocess.read_cell_for_h5("./compare_funtion/sagan/test_csv/Adam/data.h5")
#     result_list = []
#     class_num = [4,7]
#     cell_type = [np.array(["0","1","2","3"]),np.array(["0","0.5","1","1.5","2","2.5","3"])]
#     # label_list = []
#     # embeddings_list = []
#     for i in range(len(compare_list)):
#         data_array, data_label, gene_name, cell_name = data_Preprocess.read_cell_for_interpretable_imputed(
#             compare_list[i], label_list[i],class_num[i],data_set_name[i],4900,"interpretable")
#         compare_result = result_class.result_for_impute(compare_function_list[i], data_array, data_label, cell_type[i],
#                                                         class_num[i], cell_name, gene_name)
#         result_list.append(compare_result)
#     # 通过umap算法把维度降到二维,输入应该是标签在前，数组在后
#     # embeddings = umap_for_data((data_label,data_array),"drop out",["Group1","Group2","Group3","Group4"],4)
#     # for i in range(len(compare_list)):
#     #     embeddings = umap_for_data((result_list[i].data_label, result_list[i].data_mat), compare_function_list[i],
#     #                                cell_type, cell_type_num,data_set_name)
#     #     embeddings_list.append(embeddings)
#     #
#     #     label_1 = cluster.k_means(embeddings, result_list[i].cell_type_num)
#     #     label_list.append(label_1)
#     embeddings_list,label_list = umap_for_interpretable(result_list,compare_function_list,cell_type,data_set_name,"Klein")

def show_result_interpretable(data_set_name,compare_list,compare_function_list,label_file,gene_num,data_set,cell_type):
    # mat, data_label, cell_type, cell_type_num, obs, var =data_Preprocess.read_cell_for_h5("./compare_funtion/sagan/test_csv/Adam/data.h5")
    result_list = []
    label_list = []
    embeddings_list = []
    for i in range(len(compare_list)):
        if i == 0:
            data_array, data_label, gene_name, cell_name,_,_ = data_Preprocess.read_cell_for_interpretable_imputed(
                 compare_list[i], label_file,len(cell_type),data_set,gene_num,"interpretable")
        else:
            data_array, data_label, gene_name, cell_name, _ ,_= data_Preprocess.read_cell_for_interpretable_imputed(
                compare_list[i], label_file, len(cell_type), data_set, gene_num, "normal")
        # data_array, data_label, gene_name, cell_name, data_for_count = data_Preprocess.read_interpretable_for_train_T2D_imputed(compare_list[i],2,i)
        compare_result = result_class.result_for_impute(compare_function_list[i], data_array, data_label,  cell_type,
                                                        len(cell_type), cell_name, gene_name)
        result_list.append(compare_result)
    # 通过umap算法把维度降到二维,输入应该是标签在前，数组在后
    # embeddings = umap_for_data((data_label,data_array),"drop out",["Group1","Group2","Group3","Group4"],4)
    embeddings_list, label_list = umap_for_interpretable(result_list, compare_function_list,  cell_type, data_set_name,"interpretable")
    # 在通过聚类算法对指标进行测评
    label = np.array(label_list)
    # 对function_name进行赋值
    function_name = compare_function_list
    # 通过各种指标对聚类结果评测
    # AUC = AUC.compare_for_auc(function_name, data_label, label_score)
    compare.get_indicator(function_name, data_label, label, data_set_name, embeddings_list)
    draw_heatmap_for_interpretable(data_set_name,cell_type,result_list[0].gene_name)
    # PR.compare_for_PR(function_name, data_label, label_score)
    # draw_heatmap.draw_heatmap(var,obs,mat)
def show_result_interpretable_lung_cancer(data_set_name,compare_list,compare_function_list,label_file,gene_num,data_set):
    # mat, data_label, cell_type, cell_type_num, obs, var =data_Preprocess.read_cell_for_h5("./compare_funtion/sagan/test_csv/Adam/data.h5")
    result_list = []
    label_list = []
    embeddings_list = []
    for i in range(len(compare_list)):
        if i == 0:
            data_array, data_label, gene_name, cell_name, _, _ = data_Preprocess.read_cell_for_interpretable_imputed(
                compare_list[i], label_file, 2, data_set, gene_num, "interpretable")
        else:
            data_array, data_label, gene_name, cell_name, _, _ = data_Preprocess.read_cell_for_interpretable_imputed(
                compare_list[i], label_file, 2, data_set, gene_num, "normal")
        # data_array, data_label, gene_name, cell_name, data_for_count = data_Preprocess.read_interpretable_for_train_T2D_imputed(compare_list[i],2,i)
        compare_result = result_class.result_for_impute(compare_function_list[i], data_array, data_label,  ["AIS", "MIA","IAC"],
                                                        2, cell_name, gene_name)
        result_list.append(compare_result)
    # 通过umap算法把维度降到二维,输入应该是标签在前，数组在后
    # embeddings = umap_for_data((data_label,data_array),"drop out",["Group1","Group2","Group3","Group4"],4)
    embeddings_list, label_list = umap_for_interpretable(result_list, compare_function_list,["AIS", "MIA","IAC"], data_set_name,"interpretable")
    # 在通过聚类算法对指标进行测评
    label = np.array(label_list)
    # 对function_name进行赋值
    function_name = compare_function_list
    # 通过各种指标对聚类结果评测
    # AUC = AUC.compare_for_auc(function_name, data_label, label_score)
    compare.get_indicator(function_name, data_label, label, data_set_name, embeddings_list)
    draw_heatmap_for_interpretable_lung_cancer(data_set_name,result_list[0].gene_name)
    # PR.compare_for_PR(function_name, data_label, label_score)
    # draw_heatmap.draw_heatmap(var,obs,mat)
def draw_heatmap_for_interpretable_lung_cancer(data_set_name,gene_name=None):
    t_e = np.load("./compare_funtion/sagan/models/" + str(data_set_name) + "/_topic_embeding.npy")
    c_e = np.load("./compare_funtion/sagan/models/" + str(data_set_name) + "/_class_embeding.npy")
    topic_index = []
    # 首先找出与疾病有关联的主题是哪几个
    d_c = np.array(["AIS", "MIA","IAC"])
    topic = np.array([("topic" + str(i)) for i in range(50)])
    png_name = "comparsion_png"
    draw_heatmap.draw_heatmap(d_c, topic, c_e[:, :3], data_set_name, png_name)
    # 挑选对比度最高的10个主题,0是得病高富集的主题，1是健康高富集的主题
    topic_diff_AIS = np.array([c_e[i,2] - c_e[i,1] for i in range(50)])
    k_ais = np.argsort(-topic_diff_AIS)[0]
    # topic_diff_MIA = np.array([c_e[i,2]-c_e[i,1] for i in range(50)])
    # k_mia = np.argsort(-topic_diff_MIA)[0]
    # topic_diff_IAC = np.array([c_e[i,1]-c_e[i,2] for i in range(50)])
    # k_iac = np.argsort(-topic_diff_IAC)[0]
    # k_all = np.where(k_ais==k_mia)
    # print(k_all)
    #topic_index.append(k_mia)
    topic_index.append(k_ais)
    # k = np.argsort(topic_diff)[0:2]
    # 找出在这10个主题中表达量都相对较高的基因
    t_e_comp = t_e[:gene_name.shape[0], topic_index]
    t_e_diff = np.argsort(-np.squeeze(np.mean(t_e_comp,axis=1)), axis=-1)[0:200]
    print(topic_index)
    png_name_gene = "comparsion_mia_gene"
    df = draw_heatmap.draw_heatmap(topic, gene_name[t_e_diff], t_e[t_e_diff, :], data_set_name, png_name_gene)
    f = open('./result_txt/result_' + str(data_set_name) + '_mia_genes.txt', mode='w')
    f.write("id" + "\n")
    for i in range(200):
        f.write(str(gene_name[t_e_diff][i]) + "\n")
    f.close()




def draw_heatmap_for_interpretable(data_set_name,cell_type,gene_name=None):
    t_e = np.load("./compare_funtion/sagan/models/"+str(data_set_name)+"/_topic_embeding.npy")
    c_e = np.load("./compare_funtion/sagan/models/"+str(data_set_name)+"/_class_embeding.npy")

    # 首先找出与疾病有关联的主题是哪几个
    d_c = np.array(cell_type)
    topic = np.array([("topic"+str(i)) for i in range(50)])
    png_name = "comparsion_png"
    print(c_e.shape)
    draw_heatmap.draw_heatmap(d_c,topic,c_e[:,:len(cell_type)],data_set_name,png_name)
    #挑选对比度最高的10个主题,0是得病高富集的主题，1是健康高富集的主题
    list_t = []
    for j in range(len(cell_type)):
        topic_diff = np.array([c_e[i,j] for i in range(50)])
        k = np.argsort(-topic_diff)[0]
        print(topic_diff)
        # k = np.argsort(topic_diff)[0:2]
        # 找出在这10个主题中表达量都相对较高的基因
        t_e_comp = t_e[:gene_name.shape[0], k]
        print(k)
        t_e_diff = np.argsort(-np.squeeze(t_e_comp), axis=-1)[0:50]
        # 挑选出的200个基因对于此疾病有重大意义，进行热图绘画，和保存
        png_name_gene = "comparsion_png_gene_"+cell_type[j]
        df = draw_heatmap.draw_heatmap(topic, gene_name[t_e_diff], t_e[t_e_diff, :], data_set_name, png_name_gene)
        f = open('./result_txt/result_' + str(data_set_name) +"_"+cell_type[j]+ '_genes.txt', mode='w')
        f.write("id" + "\n")
        for i in range(50):
            f.write(str(gene_name[t_e_diff][i]) + "\n")
        f.close()

def find_gene_in_set(compare_list,label_file,compare_function_list,find_gene,data_type,data_set_name):
    result_list = []
    result_x = []
    result_z_orign = []
    result_z_new = []
    data_array_b, data_label_b, gene_name_b, cell_name_b, label_b, scaler_b = data_Preprocess.read_cell_for_interpretable_imputed(
        compare_list[0], label_file, 8, data_type, 4900, "interpretable")
    print(data_array_b.shape)
    #data_array_b = scaler_b.inverse_transform(data_array_b)
    compare_result_b = result_class.result_for_impute(compare_function_list[0], data_array_b, data_label_b,
                                                      ['OPC', 'astro', 'doublet', 'endo', 'mg', 'neuron', 'oligo', 'unID'],
                                                      8, cell_name_b, gene_name_b)
    result_list.append(compare_result_b)
    label_b = np.expand_dims(label_b, axis=1)
    data_array_a, data_label_a, gene_name_a, cell_name_a, label_a, _ = data_Preprocess.read_cell_for_interpretable_imputed(
        compare_list[1], label_file, 8, data_type, 4900, "normal")
    print(data_array_a.shape)
    #data_array_a = scaler_b.inverse_transform(data_array_a)
    compare_result_a = result_class.result_for_impute(compare_function_list[1], data_array_a, data_label_a,
                                                      ['OPC', 'astro', 'doublet', 'endo', 'mg', 'neuron', 'oligo', 'unID'],
                                                      8, cell_name_a, gene_name_a)
    result_list.append(compare_result_a)
    label_a = np.expand_dims(label_a, axis=1)
    for i in range(len(find_gene)):
        x = np.argwhere(result_list[0].gene_name == find_gene[i])
        gene_count = result_list[0].data_mat[:, x]
        gene_count_b = gene_count.reshape(data_array_b.shape[0],1)
        gene_for_box_b = np.concatenate((label_b, gene_count_b), axis=1)
        b = pd.DataFrame(gene_for_box_b)
        b.to_csv("./result_txt/gene_for_box_b_"+find_gene[i]+".csv")
        gene_count_z = utils.count_z_score(gene_count)
        result_z_orign.append(gene_count_z)
        gene_count_ = result_list[1].data_mat[:, x]
        gene_count_a = gene_count_.reshape(data_array_a.shape[0], 1)
        gene_for_box_a = np.concatenate((label_a, gene_count_a), axis=1)
        a = pd.DataFrame(gene_for_box_a)
        a.to_csv("./result_txt/gene_for_box_a_" + find_gene[i] + ".csv")
        gene_count_z_ = utils.count_z_score(gene_count_)
        result_z_new.append(gene_count_z_)
    #     result_x.append(x)
    # for i in range(len(compare_list)):
    #     gene_count = result_list[i].data_mat[:,x]
    #     print(gene_count)
    #     gene_count_z = utils.count_z_score(gene_count)
    #------------画图------------------------
    embeddings = umap_for_z_bar(result_list, find_gene,compare_function_list,result_z_orign,result_z_new, data_set_name)

def find_gene_in_set_lung(data_path,label_file,funtion_name,find_gene_1,find_gene_2,data_type,data_set_name):
    result_list = []
    result_x = []
    result_z_orign = []
    result_z_new = []
    result_z_orign_ =[]
    result_z_new_ = []

    data_array_b, data_label_b, gene_name_b, cell_name_b, label_b, scaler_b = data_Preprocess.read_cell_for_interpretable_imputed(
        compare_list[0], label_file, 3, data_type, 4900, "interpretable")
    print(data_array_b.shape)
    #data_array_b = scaler_b.inverse_transform(data_array_b)
    compare_result_b = result_class.result_for_impute(compare_function_list[0], data_array_b, data_label_b,
                                                      ["AIS", "MIA", "IAC"],
                                                      3, cell_name_b, gene_name_b)
    result_list.append(compare_result_b)
    #label_b = np.expand_dims(label_b, axis=1)
    data_array_a, data_label_a, gene_name_a, cell_name_a, label_a, _ = data_Preprocess.read_cell_for_interpretable_imputed(
        compare_list[1], label_file, 3, data_type, 4900, "normal")
    print(data_array_a.shape)
    #data_array_a = scaler_b.inverse_transform(data_array_a)
    compare_result_a = result_class.result_for_impute(compare_function_list[1], data_array_a, data_label_a,
                                                      ["AIS", "MIA", "IAC"],
                                                      3, cell_name_a, gene_name_a)
    result_list.append(compare_result_a)
    #label_a = np.expand_dims(label_a, axis=1)
    for i in range(len(find_gene_1)):
        x = np.argwhere(result_list[0].gene_name == find_gene_1[i])
        gene_count = result_list[0].data_mat[:, x]
        # gene_count_b = gene_count.reshape(data_array_b.shape[0], 1)
        # gene_for_box_b = np.concatenate((label_b, gene_count_b), axis=1)
        # b = pd.DataFrame(gene_for_box_b)
        # b.to_csv("./result_txt/gene_for_box_b_" + find_gene_1[i] + ".csv")
        gene_count_z = utils.count_z_score(gene_count)
        result_z_orign.append(gene_count_z)
        gene_count_ = result_list[1].data_mat[:, x]
        # gene_count_a = gene_count_.reshape(data_array_a.shape[0], 1)
        # gene_for_box_a = np.concatenate((label_a, gene_count_a), axis=1)
        # a = pd.DataFrame(gene_for_box_a)
        # a.to_csv("./result_txt/gene_for_box_a_" + find_gene_1[i] + ".csv")
        gene_count_z_ = utils.count_z_score(gene_count_)
        result_z_new.append(gene_count_z_)

    for i in range(len(find_gene_2)):
        x = np.argwhere(result_list[0].gene_name == find_gene_2[i])
        gene_count = result_list[0].data_mat[:, x]
        # gene_count_b = gene_count.reshape(data_array_b.shape[0], 1)
        # gene_for_box_b = np.concatenate((label_b, gene_count_b), axis=1)
        # b = pd.DataFrame(gene_for_box_b)
        # b.to_csv("./result_txt/gene_for_box_b_" + find_gene_2[i] + ".csv")
        gene_count_z = utils.count_z_score(gene_count)
        result_z_orign_.append(gene_count_z)
        gene_count_ = result_list[1].data_mat[:, x]
        # gene_count_a = gene_count_.reshape(data_array_a.shape[0], 1)
        # gene_for_box_a = np.concatenate((label_a, gene_count_a), axis=1)
        # a = pd.DataFrame(gene_for_box_a)
        # a.to_csv("./result_txt/gene_for_box_a_" + find_gene_2[i] + ".csv")
        gene_count_z_ = utils.count_z_score(gene_count_)
        result_z_new_.append(gene_count_z_)
    #     result_x.append(x)
    # for i in range(len(compare_list)):
    #     gene_count = result_list[i].data_mat[:,x]
    #     print(gene_count)
    #     gene_count_z = utils.count_z_score(gene_count)
    embeddings = umap_for_z_bar_lung_cancer(result_list,find_gene_1,find_gene_2, funtion_name, result_z_orign, result_z_new,result_z_orign_,result_z_new_, data_set_name)










if __name__=="__main__":
    data_array,data_label,gene_name,cell_name=data_Preprocess.read_cell("./test_csv/splatter_exprSet_test.csv", "./test_csv/splatter_exprSet_test_label.csv", 4)
    #-----------对h5数据集进行可视化处理-------------------------
    data_set_name = "Adam"
    compare_list = ["./compare_funtion/sagan/result/"+data_set_name+"/"+data_set_name+"_raw.csv"
        ,"./compare_funtion/sagan/result/"+data_set_name+"/"+data_set_name+"-Imputed-DCA.csv"
        ,"./compare_funtion/sagan/result/"+data_set_name+"/"+data_set_name+"-Imputed-deepImpute.csv"
        ,"./compare_funtion/sagan/result/"+data_set_name+"/"+data_set_name+"-Imputed-MAGIC.csv"
        ,"./compare_funtion/sagan/result/"+data_set_name+"/"+data_set_name+"-Imputed-scIGANs.csv"
        ,"./compare_funtion/sagan/result/"+data_set_name+"/"+data_set_name+"-Imputed-scScope.csv"
        ,"./compare_funtion/sagan/result/"+data_set_name+"/"+data_set_name+"-Imputed-autoImputed.csv"
        ,"./compare_funtion/sagan/result/"+data_set_name+"/"+data_set_name+"-Imputed-scImpute.csv"
        ,"./compare_funtion/sagan/result/"+data_set_name+"/"+"student.csv"]
    compare_function_list=["dropout","DCA","deepImputed","MAGIC",
                           "scIGANs","scScope","autoImpute","scImpute",
                           "scAFGRL"]
    show_result_h5(data_set_name,compare_list,compare_function_list,"./compare_funtion/sagan/test_csv/"+data_set_name+"/data.h5")

    # data_set_name = "Romanov"
    # compare_list = [
    #     "./compare_funtion/sagan/result/parmeter/AD_cluster-imputed-Romanov_0.3_w_0.5.csv"
    #     , "./compare_funtion/sagan/result/parmeter/AD_cluster-imputed-Romanov_0.3_w_0.6.csv"
    #     , "./compare_funtion/sagan/result/parmeter/AD_cluster-imputed-Romanov_0.3_w_0.7.csv"
    #     , "./compare_funtion/sagan/result/parmeter/AD_cluster-imputed-Romanov_0.3_w_0.8.csv"
    #     , "./compare_funtion/sagan/result/parmeter/AD_cluster-imputed-Romanov_0.3_w_0.9.csv"
    #     ]
    # compare_function_list = [
    #                          "sagan_0.3_w_0.5", "sagan_0.3_w_0.6","sagan_0.3_w_0.7","sagan_0.3_w_0.8","sagan_0.3_w_0.9"
    #                          ]
    # show_result_h5(data_set_name, compare_list, compare_function_list,
    #                "./compare_funtion/sagan/test_csv/" + data_set_name + "/data.h5")
    # data_set_name = "Muraro"
    # compare_list = ["./compare_funtion/sagan/result/" + data_set_name + "/" + data_set_name + "_raw.csv"
    #     , "./compare_funtion/sagan/result/" + data_set_name + "/" + data_set_name + "-imputed-scsagan.csv"]
    # compare_function_list=["Muraro_raw","Muraro_clustering"]
    # show_result_h5(data_set_name,compare_list,compare_function_list,"./compare_funtion/sagan/test_csv/"+data_set_name+"/data.h5")
    # -----------对HP数据集进行可解释性研究-------------------------
    # data_set_name = "AD_cluster"
    # compare_list = ["./compare_funtion/sagan/result/AD_interpretable/GSE138852_counts.csv","./compare_funtion/sagan/result/AD_interpretable/AD_cluster-imputed-scsagan.csv"]
    # compare_function_list = ["dropout","scSAGAN"]
    # cell_type = ['OPC', 'astro', 'doublet', 'endo', 'mg', 'neuron', 'oligo', 'unID']
    # label_file = "./compare_funtion/sagan/test_csv/AD_interpretable/GSE138852_covariates.csv"
    #show_result_interpretable(data_set_name,compare_list,compare_function_list,label_file,4900,"AD_cluster",cell_type)
    # # #前三个促进，后三个抑制
    # # #COVID-19的5个基因都是促进
    # # #小鼠前两个促进，后两个抑制
    # ais_to_mia = ["ENSG00000109320","ENSG00000124813","ENSG00000162924","ENSG00000122008"]
    # mia_to_iac = ["ENSG00000122691","ENSG00000126767","ENSG00000147065","ENSG00000160883"]
    # find_gene_in_set_lung(compare_list,label_file,compare_function_list,ais_to_mia,mia_to_iac,"lung_cancer",data_set_name)
    # gene = ["LINC00499","ZC3H6","PSAP","RP11-665G4.1","SFXN5","TKT"]
    # find_gene_in_set(compare_list,label_file,compare_function_list,gene,"AD_cluster",data_set_name)
    #-------------查看生成假细胞对分化轨迹的影响----------------------
    # data_set_name = ["Klein_old","Klein_new"]
    # compare_list = ["./compare_funtion/sagan/result/Klein/Klein_raw.csv",
    #                 "./compare_funtion/sagan/result/Klein/Klein_fakeimages.csv"]
    # label_list = ["./compare_funtion/sagan/result/Klein/label.csv",
    #             "./compare_funtion/sagan/result/Klein/Klein_fakelabels.csv"]
    # compare_function_list = ["original_cells","virtual_cells"]
    # show_result_Klein(compare_list,label_list,compare_function_list,data_set_name)
    #-------------消除批次效应-----------------------------------------
    # data_set_name = "Young"
    # compare_list = ["./compare_funtion/sagan/result/"+data_set_name+"/"+data_set_name+"_raw.csv","./compare_funtion/sagan/result/"+data_set_name+"/"+data_set_name+"_0.3-imputed-scsagan(filted).csv"]
    # compare_function_list = ["dropout","sagan_0.3"]
    # show_result_h5_batch(data_set_name,compare_list,compare_function_list,"./compare_funtion/sagan/test_csv/"+data_set_name+"/data.h5")