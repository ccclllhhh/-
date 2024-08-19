import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patches as patches

# 设置中文字体
def set_font():
    font_path = 'C:/Windows/Fonts/simsun.ttc'  # 替换为合适的字体路径
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()

# 绘制节点
def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.axl.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction',
                             va="center", ha="center", bbox=node_type, arrowprops=arrow_args)

# 获取叶节点数量
def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs

# 获取树深度
def get_tree_depth(my_tree):
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        max_depth = max(max_depth, this_depth)
    return max_depth

# 绘制中间文本
def plot_mid_text(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.axl.text(x_mid, y_mid, txt_string, va="center", ha="center",
                         bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

# 绘制树
def plot_tree(my_tree, parent_pt, node_txt):
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = list(my_tree.keys())[0]
    cntr_pt = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w, plot_tree.y_off)
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), cntr_pt, str(key))
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d

# 绘制决策树
def create_plot(in_tree):
    set_font()  # 设置字体
    fig = plt.figure(figsize=(10, 8), dpi=100)  # 调整图像大小
    fig.clf()
    axprops = dict(xticks=[], yticks=[], title='决策树', xlabel='', ylabel='')
    create_plot.axl = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_w = float(get_num_leafs(in_tree))
    plot_tree.total_d = float(get_tree_depth(in_tree))
    plot_tree.x_off = -0.5 / plot_tree.total_w
    plot_tree.y_off = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.title('决策树', fontsize=16)
    plt.show()

# 节点样式设置
decision_node = dict(boxstyle="sawtooth", fc='lightblue', edgecolor='black')
leaf_node = dict(boxstyle="round4", fc='lightgreen', edgecolor='black')
arrow_args = dict(arrowstyle="<-", color='black')

# 示例：使用决策树数据绘图
# 示例数据，实际中你可以用自己创建的树
sample_tree = {
    '脐部': {
        '凹陷': {
            '色泽': {
                '青绿': '是',
                '乌黑': '是',
                '浅白': '否'
            }
        },
        '稍凹': '否'
    }
}

# 绘制决策树
create_plot(sample_tree)
