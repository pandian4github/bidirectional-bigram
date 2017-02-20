import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_graph(x_axis, y_axis1, y_axis2, x_label, y_label, title, axis, file_name):

    plt.plot(x_axis, y_axis1, 'r-o', label='training')
    plt.plot(x_axis, y_axis2, 'b-o', label='test')

    red_training_patch = mpatches.Patch(color='red', label='Training set', linestyle='solid', linewidth=0.1)
    blue_test_patch = mpatches.Patch(color='blue', label='Test set', linestyle='solid', linewidth=0.1)

    lgd = plt.legend(handles=[red_training_patch, blue_test_patch], loc='upper right', bbox_to_anchor=(1.4, 1.0))

    plt.grid(True)
    plt.axis(axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    fig = plt.figure(1)
    fig.savefig(file_name, dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.close()


def main(target_corpus):
    lambda3s = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    if target_corpus == 'atis':
        training_perplexities = [9.438037739398142, 8.708723041938676, 8.257280773392038, 7.942886690200488, 7.71294362892478, 7.541848703931222, 7.415389383061672, 7.325134678745447, 7.26601520057024, 7.235173934082009, 7.2314209752273175, 7.255039123296924, 7.307863155306111, 7.393671404380485, 7.519077748817468, 7.6954397448902885, 7.9432643870480835, 8.304303388624515, 8.888034980651815]
        test_perplexities = [18.437481499314085, 16.425586552119235, 15.25996322241703, 14.473949867317595, 13.908040678356457, 13.488738814714045, 13.17674213852131, 12.949317973874642, 12.792998282007789, 12.700210156018738, 12.667706052956225, 12.696005051120713, 12.789608130707185, 12.958105857620245, 13.218764870403565, 13.602251728136917, 14.166478851522845, 15.037163180148722, 16.579982703909163]
        axis = [0, 1, 0, 25]
        corpus = 'atis'
    elif target_corpus == 'wsj':
        training_perplexities = [62.96085561829097, 56.88377096390624, 53.34896025157767, 50.99519973381952, 49.34396255061554, 48.171440122996934, 47.357435906384424, 46.83304816334652, 46.558773871482046, 46.514445090631405, 46.69472851118176, 47.10790714145462, 47.777370420863555, 48.74644697298152, 50.08894672537915, 51.93183374713426, 54.508918810983154, 58.31449953667728, 64.73468201202915]
        test_perplexities = [177.08946258375565, 157.53862505431553, 146.58734764028338, 139.4474228803418, 134.504559815185, 131.02427658662242, 128.61976032547457, 127.07234618406198, 126.25784195756114, 126.1131573880389, 126.62152989276237, 127.80855365063704, 129.74707455049523, 132.57312887103024, 136.5210838584855, 142.00041921756323, 149.7814466703079, 161.54180534599658, 182.20253028784037]
        axis = [0, 1, 30, 200]
        corpus = 'wsj'
    elif target_corpus == 'brown':
        training_perplexities = [81.68720150807204, 74.27118804405376, 69.9266556544562, 67.01944032599995, 64.97324735203797, 63.51749561136968, 62.50636798324809, 61.856050631456476, 61.51840915742327, 61.46886647115352, 61.70097525896954, 62.224950694181885, 63.069475502031366, 64.2875387543983, 65.9691458868263, 68.26858691379772, 71.46884025280694, 76.16541675595285, 84.02100884796712]
        test_perplexities = [223.7702150993273, 202.53943505230586, 190.31265546609373, 182.2429688771805, 176.63224064404895, 172.69089178312655, 169.9974394664157, 168.3118703042146, 167.49784289904488, 167.48711091425574, 168.26375206248065, 169.86008221231313, 172.3622496703593, 175.92784214536516, 180.82407791842974, 187.50974128070146, 196.82868177408045, 210.56030597717665, 233.64630618359485]
        axis = [0, 1, 50, 250]
        corpus = 'brown'
    else:
        print 'Invalid corpus name!'
        return

    x_label = 'Interpolation weight for forward'
    y_label = 'Word Perplexity'
    title = 'Effect of forward-backward interpolation weight on word perplexity (' + corpus + ')'
    file_name = 'bidirectional_' + corpus + '.png'
    plot_graph(lambda3s, training_perplexities, test_perplexities, x_label, y_label, title, axis, file_name)


main('atis')
main('wsj')
main('brown')