import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

def readTensor(path, name):
    g = glob.glob(path+ '/**/' + name, recursive=True)
    # print(g)
    d = {}
    for f in g:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        f= f.replace(path, '')
        f= f.replace(name, '')
        f= f.replace('//', '')
        f = f.replace('\\', '')
        d[f] = img
    return d

def getHist(maskname, d):
    temp = np.copy(d['original'])
    temp = temp.astype(np.int)
    if not np.any(d[maskname] > 0):
        return np.zeros(256), np.arange(257)
    temp[d[maskname] == 0] = -1
    return np.histogram(temp.ravel(), bins=256, range=(0.0, 255.0), density=True)

# d = readTensor(r"F://Dissertation//Synthetic//Synthetics//Synthetics python//dataset//new","0_2022_12_15.png")


def calcAreas(path, name):
    print('calc', path, name)
    d = readTensor(path, name)
    print(d.keys())
    background = 255 - (d['vesicles'] + d['axon'] + d['PSD'] + d['mitochondria'] + d['mitochondrial_boundaries'] + d['boundaries'])
    d['background'] = background
    num = background.shape[0] * background.shape[1]
    # masks have white color 255
    vesicles = np.sum(d['vesicles']) * 100 / (255 * num)
    axon = np.sum(d['axon']) * 100 / (255 * num)
    PSD = np.sum(d['PSD']) * 100 / (255 * num)
    mitochondria = np.sum(d['mitochondria'] + d['mitochondrial_boundaries']) * 100 / (255 * num)
    boundaries = np.sum(d['boundaries']) * 100 / (255 * num)
    ground = np.sum(d['background']) * 100 / (255 * num)
    
    print(num, vesicles, axon, PSD, mitochondria, boundaries, ground)
    
    return vesicles, axon, PSD, mitochondria, boundaries, ground

def calcSlice(path, name):
    print('calc', path, name)
    d = readTensor(path, name)
    print(d.keys())
    background = 255 - (d['vesicles'] + d['axon'] + d['PSD'] + d['mitochondria'] + d['mitochondrial_boundaries'] + d['boundaries'])
    d['background'] = background

    vesicles, bin_edges = getHist('vesicles', d)
    axon, bin_edges = getHist('axon', d)
    PSD, bin_edges = getHist('PSD', d)
    mitochondria, bin_edges = getHist('mitochondria', d)
    mitochondrial_boundaries, bin_edges = getHist('mitochondrial_boundaries', d)
    boundaries, bin_edges = getHist('boundaries', d)
    ground, bin_edges = getHist('background', d)
    
    return bin_edges, vesicles, axon, PSD, mitochondria, mitochondrial_boundaries, boundaries, ground
    
def calcSliceO(path, name):
    print('calc', path, name)
    d = readTensor(path, name)
    print(d.keys())
    background = 255 - (d['vesicles'] + d['axon'] + d['PSD'] + d['mitochondria'] + d['mitochondrial boundaries'] + d['boundaries'])
    d['background'] = background

    vesicles, bin_edges = getHist('vesicles', d)
    axon, bin_edges = getHist('axon', d)
    PSD, bin_edges = getHist('PSD', d)
    mitochondria, bin_edges = getHist('mitochondria', d)
    mitochondrial_boundaries, bin_edges = getHist('mitochondrial boundaries', d)
    boundaries, bin_edges = getHist('boundaries', d)
    ground, bin_edges = getHist('background', d)

    return bin_edges, vesicles, axon, PSD, mitochondria, mitochondrial_boundaries, boundaries, ground


def printPlot(title, bin_edges, vesicles, axon, PSD, mitochondria, mitochondrial_boundaries, boundaries, ground):
   
    plt.rcParams['figure.figsize'] = (12, 3)
    plt.rcParams.update({'font.size': 15})
    plt.subplots_adjust(left=0.16, bottom=0.19, top=0.82)
    
    plt.text(6, 0.03, title, bbox = {'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'round'}, fontsize=15)
    #plt.title(title)
    #plt.xlabel("grayscale value")
    plt.ylabel("density")
    plt.xlim([0.0, 255.0])
    plt.ylim([0.0, 0.04])

    separate = False

    plt.plot(bin_edges[0:-1], ground, 'g', label = 'background')
    plt.plot(bin_edges[0:-1], vesicles, 'r', label = 'vesicles')
    plt.plot(bin_edges[0:-1], axon, 'b', label = 'axon')
    plt.plot(bin_edges[0:-1], PSD, 'c', label = 'PSD')
    plt.plot(bin_edges[0:-1], mitochondria, 'm', label = 'mitochondria')
    plt.plot(bin_edges[0:-1], mitochondrial_boundaries, 'y', label = 'mitochondrial boundaries')
    plt.plot(bin_edges[0:-1], boundaries, 'k', label = 'boundaries')
    plt.legend()
    
    

    plt.show()

    if separate:
        plt.title(title)
        plt.ylim([0.0, 0.05])
        plt.plot(bin_edges[0:-1], ground, 'g', label = 'background')
        plt.legend()
        plt.show()

        plt.title(title)
        plt.ylim([0.0, 0.05])
        plt.plot(bin_edges[0:-1], vesicles, 'r', label = 'vesicles')
        plt.legend()
        plt.show()

        plt.title(title)
        plt.ylim([0.0, 0.05])
        plt.plot(bin_edges[0:-1], axon, 'b', label = 'axon')
        plt.legend()
        plt.show()

        plt.title(title)
        plt.ylim([0.0, 0.05])
        plt.plot(bin_edges[0:-1], PSD, 'c', label = 'PSD')
        plt.legend()
        plt.show()

        plt.title(title)
        plt.ylim([0.0, 0.05])
        plt.plot(bin_edges[0:-1], mitochondria, 'm', label = 'mitochondria')
        plt.legend()
        plt.show()

        plt.title(title)
        plt.ylim([0.0, 0.05])
        plt.plot(bin_edges[0:-1], mitochondrial_boundaries, 'y', label = 'mitochondrial boundaries')
        plt.legend()
        plt.show()

        plt.title(title)
        plt.ylim([0.0, 0.05])
        plt.plot(bin_edges[0:-1], boundaries, 'k', label = 'boundaries')
        plt.legend()

        plt.show()


def printTwoPlot(title, bin_edges, original, synthetic):
    plt.title(title)
    plt.xlabel("grayscale value")
    plt.ylabel("density")
    plt.xlim([0.0, 255.0])
    plt.ylim([0.0, 0.05])

    plt.plot(bin_edges[0:-1], original, 'g', label = 'original')
    plt.plot(bin_edges[0:-1], synthetic, 'r', label = 'synthetic')
    plt.legend()
    plt.show()
    
def printTreePlot(title, bin_edges, original, original2, synthetic):
    plt.rcParams['figure.figsize'] = (12, 3)
    plt.rcParams.update({'font.size': 15})
    plt.subplots_adjust(left=0.16, bottom=0.19, top=0.82)

    
    #plt.title(title)
    plt.text(6, 0.03, title, bbox = {'facecolor': 'white', 'edgecolor': 'black', 'boxstyle': 'round'}, fontsize=15)
    #plt.xlabel("grayscale value")
    plt.ylabel("density")
    plt.xlim([0.0, 255.0])
    plt.ylim([0.0, 0.04])

    plt.plot(bin_edges[0:-1], original, 'g', label = 'originals all', linewidth = 3, alpha=0.5)
    plt.plot(bin_edges[0:-1], original2, 'r', label = 'original 0', linewidth = 2, alpha=0.5)
    plt.plot(bin_edges[0:-1], synthetic, 'b', label = 'synthetic', linewidth = 2, alpha=0.5)
    plt.legend()
    
    plt.savefig("3 print color/"+title+".png", dpi=600)
    
    plt.show()

path = r"G:\Data\Unet_multiclass\data\original data"
go = glob.glob(path +"//original//*.png")

sumvesicles = np.zeros(256)
sumaxon = np.zeros(256)
sumPSD = np.zeros(256)
summitochondria = np.zeros(256)
summitochondrial_boundaries = np.zeros(256)
sumboundaries = np.zeros(256)
sumground = np.zeros(256)

for f in go:
    f = f.split('\\')[-1]
    bin_edges, vesicles, axon, PSD, mitochondria, mitochondrial_boundaries, boundaries, ground = calcSliceO(path, f)
    sumvesicles = sumvesicles + vesicles
    sumaxon = sumaxon + axon
    sumPSD = sumPSD + PSD
    summitochondria = summitochondria + mitochondria
    summitochondrial_boundaries = summitochondrial_boundaries + mitochondrial_boundaries
    sumboundaries = sumboundaries + boundaries
    sumground = sumground + ground
    
o_vesicles = sumvesicles / len(go)
o_axon = sumaxon / len(go)
o_PSD = sumPSD / len(go)
o_mitochondria = summitochondria / len(go)
o_mitochondrial_boundaries = summitochondrial_boundaries / len(go)
o_boundaries = sumboundaries / len(go)
o_ground = sumground / len(go)


printPlot('Original all layers',bin_edges, o_vesicles, o_axon, o_PSD, o_mitochondria, o_mitochondrial_boundaries, o_boundaries, o_ground)

bin_edges2, vesicles2, axon2, PSD2, mitochondria2, mitochondrial_boundaries2, boundaries2, ground2 = calcSliceO(path, "training0000.png")
printPlot('Original 0 layer',bin_edges2, vesicles2, axon2, PSD2, mitochondria2, mitochondrial_boundaries2, boundaries2, ground2)


path = r"C:\Users\Sokol-PC\Synthetics\Synthetics python\dataset\synthetic_dataset3"
g = glob.glob(path +"//original//*.png")

sumvesicles = np.zeros(256)
sumaxon = np.zeros(256)
sumPSD = np.zeros(256)
summitochondria = np.zeros(256)
summitochondrial_boundaries = np.zeros(256)
sumboundaries = np.zeros(256)
sumground = np.zeros(256)


for f in g:
    f = f.split('\\')[-1]
    bin_edges, vesicles, axon, PSD, mitochondria, mitochondrial_boundaries, boundaries, ground = calcSliceO(path, f)
    sumvesicles = sumvesicles + vesicles
    sumaxon = sumaxon + axon
    sumPSD = sumPSD + PSD
    summitochondria = summitochondria + mitochondria
    summitochondrial_boundaries = summitochondrial_boundaries + mitochondrial_boundaries
    sumboundaries = sumboundaries + boundaries
    sumground = sumground + ground

sumvesicles = sumvesicles / len(g)
sumaxon = sumaxon / len(g)
sumPSD = sumPSD / len(g)
summitochondria = summitochondria / len(g)
summitochondrial_boundaries = summitochondrial_boundaries / len(g)
sumboundaries = sumboundaries / len(g)
sumground = sumground / len(g)

printPlot('Synthetic', bin_edges, sumvesicles, sumaxon, sumPSD, summitochondria, summitochondrial_boundaries, sumboundaries, sumground)

printTwoPlot('vesicles', bin_edges, o_vesicles, sumvesicles)
printTwoPlot('axon', bin_edges, o_axon, sumaxon)
printTwoPlot('PSD', bin_edges, o_PSD, sumPSD)
printTwoPlot('mitochondria', bin_edges, o_mitochondria, summitochondria)
printTwoPlot('mitochondrial_boundaries', bin_edges, o_mitochondrial_boundaries, summitochondrial_boundaries)
printTwoPlot('membranes', bin_edges, o_boundaries, sumboundaries)
printTwoPlot('ground', bin_edges, o_ground, sumground)

printTreePlot('Vesicles', bin_edges, o_vesicles, vesicles2, sumvesicles)
printTreePlot('Axon', bin_edges, o_axon, axon2,sumaxon)
printTreePlot('PSD', bin_edges, o_PSD, PSD2, sumPSD)
printTreePlot('Mitochondria', bin_edges, o_mitochondria, mitochondria2, summitochondria)
printTreePlot('Mitochondrial boundaries', bin_edges, o_mitochondrial_boundaries, mitochondrial_boundaries2, summitochondrial_boundaries)
printTreePlot('Membranes', bin_edges, o_boundaries, boundaries2, sumboundaries)
printTreePlot('Ground', bin_edges, o_ground, ground2, sumground)
######################################################################################################################################
sumvesicles = 0
sumaxon = 0
sumPSD = 0
summitochondria = 0
sumboundaries = 0
sumground = 0

path = r"C:\Users\Sokol-PC\Synthetics\Synthetics python\dataset\synthetic_dataset3"
g = glob.glob(path +"//original//*.png")

for f in g:
    f = f.split('\\')[-1]
    vesicles, axon, PSD, mitochondria, boundaries, ground = calcAreas(path, f)
    sumvesicles = sumvesicles + vesicles
    sumaxon = sumaxon + axon
    sumPSD = sumPSD + PSD
    summitochondria = summitochondria + mitochondria
    sumboundaries = sumboundaries + boundaries
    sumground = sumground + ground

sumvesicles = sumvesicles / len(g)
sumaxon = sumaxon / len(g)
sumPSD = sumPSD / len(g)
summitochondria = summitochondria / len(g)
sumboundaries = sumboundaries / len(g)
sumground = sumground / len(g)


labels = ['vesicles', 'axon', 'PSD', 'mitochondria', 'boundaries']
synthetics = [sumvesicles, sumaxon, sumPSD, summitochondria, sumboundaries]

vesicles, axon, PSD, mitochondria, boundaries, ground = calcAreas(r"F://Dissertation//EPFL//new","training0000.png")
original = [vesicles, axon, PSD, mitochondria, boundaries]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, synthetics, width, label='synthetics')
rects2 = ax.bar(x + width/2, original, width, label='original')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Persent')
ax.set_title('Persent of slice area')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()
