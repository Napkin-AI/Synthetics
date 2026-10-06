import cv2
import numpy as np
import shutil


img_path = "C:\\Users\\matve\\PycharmProjects\\Synthetics\\dataset\\test_dataset_no_noise\\" #path dataset
img_new_path = "C:\\Users\\matve\\PycharmProjects\\Synthetics\\dataset\\sort_dataset\\" #path sort_dataset
#need to create folders:axon, boundaries, mitochondria, mitochondrial_boundaries, original, PSD, vesicles in the img_new_path directory
img_date = "_2024_07_05" #date dataset


def cnt_white(img):
    cnt=0
    for i in range(256):
        for j in range(256):
            if(img[i][j]==0):
                cnt+=1
    return 256*256-cnt

def copy_image(img_name, img_type):
    shutil.copy2(img_path + "axon\\" + img_name + ".png", img_new_path + img_type + "axon\\" + img_name + ".png")
    shutil.copy2(img_path + "boundaries\\" + img_name + ".png", img_new_path + img_type + "boundaries\\" + img_name + ".png")
    shutil.copy2(img_path + "mitochondria\\" + img_name + ".png", img_new_path + img_type + "mitochondria\\" + img_name + ".png")
    shutil.copy2(img_path + "mitochondrial_boundaries\\" + img_name + ".png", img_new_path + img_type + "mitochondrial_boundaries\\" + img_name + ".png")
    shutil.copy2(img_path + "original\\" + img_name + ".png", img_new_path + img_type + "original\\" + img_name + ".png")
    shutil.copy2(img_path + "PSD\\" + img_name + ".png", img_new_path + img_type + "PSD\\" + img_name + ".png")
    shutil.copy2(img_path + "vesicles\\" + img_name + ".png", img_new_path + img_type + "vesicles\\" + img_name + ".png")

def main():
    file = open(img_new_path + "logs.txt", "w")

    for j in range(0, 1000):
        img_name = str(j) + img_date
        label_map = cv2.imread(img_path + "axon\\" + img_name + ".png", cv2.IMREAD_GRAYSCALE)

        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(label_map[:, :], connectivity, cv2.CV_32S)

        cnt_axon = 0
        if (num_labels > 1):
            cnt_axon = 1
        file.write(img_name + ';Axon;' + str(cnt_axon) + '\n')

        img=label_map
        kernel = np.ones((10,10), np.uint8)
        erosion = cv2.erode(img, kernel, iterations=1)

        img_type = ""
        if(cnt_white(erosion)>100):
            if(cnt_white(img)>6000):
                img_type = "full_axon\\"
            else:
                img_type = "part_of_the_axon\\"
        else:
            if(cnt_white(img)>1300):
                img_type = "full_axon\\"
            elif(cnt_white(img)>0):
                img_type = "part_of_the_axon\\"
            else:
                img_type = "is_no_axon\\"
        copy_image(img_name,img_type)

    file.close()

if __name__ == "__main__":
    main()
