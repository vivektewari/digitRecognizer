import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc, random, cv2
from itertools import permutations, combinations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from commonFuncs import *
import torch
from prettytable import PrettyTable

def toImage(savePath,dataPath):
    train = pd.read_csv(dataPath / 'train.csv', nrows=100)
    plt.gray()
    for row in range(train.shape[0]):
        all = list(train.iloc[row])
        label = all[0]
        data = np.array(all[1:])
        data = np.where(data > 100, 255, 0)  # add step to preprocess the data
        data = data.reshape((28, 28))
        fig = plt.figure()
        plt.imshow(data)
        fig.savefig(savePath / str(str(row) + "_" + str(label) + '.png'))
        plt.close(fig)
        if row % 1000 == 0: gc.collect()
    # print("time taken ".format(time.time()-startTime))


def getMetrics(actual, predicted):
    """
    :param actual: actual series
    :param predicted: predicted series
    :return: list of accuracy ,precision ,recall and f1
    """
    met = []
    metrics = [accuracy_score, precision_score, recall_score, f1_score]
    for m in metrics:
        if m == accuracy_score:
            met.append(m(actual, predicted))
        else:
            met.append(m(actual, predicted, average="macro"))
    return met



def updateMetricsSheet(dev_actual, dev_pred, hold_actual, hold_pred, loc="", modelName="", extraInfo="",
                       force=False):
    model = 'model'
    f = pd.read_csv(loc, index_col='index')
    if modelName in list(f[model]):
        if not force: raise Exception("model exist. try with Force as True or different model name")
        # else:f.drop(f[f[model]==modelName].index,axis=0)
    metricsDev, metricsHold = getMetrics(dev_actual, dev_pred), getMetrics(hold_actual, hold_pred)
    entryVal = modelName, *metricsDev, *metricsHold, extraInfo
    dict = {}

    for i in range(f.shape[1]):
        dict[f.columns[i]] = entryVal[i]

    pd.DataFrame(dict, index=[f.shape[0]]).to_csv(loc, mode='a', header=False)


def get_dict_from_class(class1):
    return {k: v for k, v in class1.__dict__.items() if not (k.startswith('__') and k.endswith('__'))}

class vison_utils:
    @staticmethod
    def create_boxes(scores,n_classes,bounding_locs):
        """
        :param scores: tensor with dim as batch,claases scores
        :param n_classes: num classes should be num classes +1 for background
        :param bounding_locs: tensor with dim as batch,locs dim
        :return: boxes with dim batch,boxes,6([actual,prob,4 locs)
        """
        box_count=int(scores.shape[1]/(n_classes))
        batch_len=int(scores.shape[0])

        scores = scores.reshape(batch_len,box_count,n_classes)
        argmax_score,argmax_class=torch.max(scores,dim=2,keepdim=True)

        bounding_locs = bounding_locs.reshape(batch_len,box_count,int(bounding_locs.shape[1]/box_count))
        return_boxes=torch.cat([argmax_class,argmax_score,bounding_locs],dim=2)


        return return_boxes

    @classmethod
    def non_max_suppression(cls, boxes, classes, background_class, pred_thresh=0.3, overlap_thresh=0.7):
        """
        Algo:(for a single image input)
        1. Remove background boxes and have less than pred_thresh
        1.iterate over all the class
            1: Subset boxes with this class. Sorts in decreasign order of probs.
            2. create a 2 dim array with u triangle all 0, find max of each column and if max is greater
            overlap threshold , drop the the box
            3. remaining boxes are the predicted boxes

        :param boxes: tensor with label,max_prob,4 location parmas
        :param pred_thresh: threshold below which box will not be considered
        :param overlap_thresh: suppression jaccard
        :param classes : classes list of n dim(excludes background class)
        :param background_class: background class  identifier. single string /int.
        :return: subset of above boxes
        """
        boxes_wt_bg = boxes[boxes[:, 0] != background_class]
        boxes_wt_bg=boxes_wt_bg[boxes_wt_bg[:, 1] > pred_thresh]
        if len(boxes_wt_bg)==0 : return []
        final_boxes=None
        for i in classes:
            boxes_i=boxes_wt_bg[boxes_wt_bg[:,0]==i]
            if boxes_i.shape[0]==0: continue
            _,sort_order =boxes_i[:,1].sort(dim=0, descending=True)
            boxes_i=boxes_i[sort_order]
            temp1=cls.find_jaccard_overlap(boxes_i[:,2:],boxes_i[:,2:])

            temp = torch.where(temp1 <= overlap_thresh,  torch.tensor(0.0),temp1)
            temp = torch.tril(temp, diagonal=-1)
            temp=torch.sum(temp,dim=1)
            indices = (temp==0).nonzero(as_tuple=True)
            temp=boxes_i[indices]
            if final_boxes is None:
                final_boxes=temp
            else:
                final_boxes=torch.cat([final_boxes,temp],dim=0)
            #final_boxes.extend([temp[i] for i in range(temp.shape[0])])
        return final_boxes
    @classmethod
    def tp_fp_fn(cls, pred_boxes, true_boxes, classes, iou_threshold=0.5):
        """
        Algortihm:
        1. iterate over all the classes
            1.find jaccard over pre vs true boxes for that class. take the upper triangle. pred rows and true columns
            2.in a row then column aprt from max, make all other element 0.
            3.max row, to get 1 d array. Count number of them greater than IOU threshold. This is TP
            4. count rows - TP =FP nad count coulmns - TP is False Negatives
            5. save tp, fp,fn in dictionary
        2. combine dictionary to get precision and recall for each class as well as overall(averaged)

        :param pred_boxes: tensor with label,max_prob,4 location parmas
        :param true_boxes: tensor with label,max_prob=1,4 location parmas
        :return:Dataframe with class +all as row indexes and precision and recall as column index
        """
        final_dict= torch.zeros((len(classes),3))
        for i in classes:
            true_boxes_i = true_boxes[true_boxes[:, 0] == i][:, 2:]
            if len(pred_boxes)==0:
                tp=0
                pred_boxes_i=[]
            else:
                pred_boxes_i=pred_boxes[pred_boxes[:,0]==i][:,2:]

                if len(pred_boxes_i)==0 or len(true_boxes_i)==0 :
                    tp=0
                else:
                    jaccard = cls.find_jaccard_overlap(pred_boxes_i,true_boxes_i)
                    jaccard = torch.where(jaccard>=iou_threshold,jaccard,torch.tensor(0.0))
                    row_max = torch.argmax(jaccard,dim=0).tolist()
                    col_max = torch.argmax(jaccard, dim=1).tolist()
                    row_max_indices=[(i,row_max[i]) for i in range(jaccard.shape[1])]
                    col_max_indices = [(col_max[i],i) for i in range(jaccard.shape[0])]

                    final_list = list(set( row_max_indices).intersection(set( col_max_indices)))
                    tp = len(final_list)


            fp = len(pred_boxes_i) - tp
            fn = len(true_boxes_i) - tp
            final_dict[i,:] = final_dict[i,:]+torch.tensor([tp,fp,fn])
            i+=1
        #print(final_dict)
        return final_dict






    def find_intersection(set_1, set_2):
        """
        Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
        :param set_1: set 1, a tensor of dimensions (n1, 4)
        :param set_2: set 2, a tensor of dimensions (n2, 4)
        :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
        """

        # PyTorch auto-broadcasts singleton dimensions
        lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
        upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
        intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
        return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

    @staticmethod
    def find_jaccard_overlap(set_1, set_2):
        """
        Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
        :param set_1: set 1, a tensor of dimensions (n1, 4)
        :param set_2: set 2, a tensor of dimensions (n2, 4)
        :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
        """

        # Find intersections
        intersection = vison_utils.find_intersection(set_1, set_2)  # (n1, n2)

        # Find areas of each box in both sets
        areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
        areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

        # Find the union
        # PyTorch auto-broadcasts singleton dimensions
        union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

        return intersection / union  # (n1, n2)

    def xy_to_cxcy(xy):
        """
        Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).
        :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
        :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
        """
        return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                          xy[:, 2:] - xy[:, :2]], 1)  # w, h

    def cxcy_to_xy(cxcy):
        """
        Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
        :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
        :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
        """
        return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                          cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max

    def cxcy_to_gcxgcy(cxcy, priors_cxcy):
        """
        Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).
        For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
        For the size coordinates, scale by the size of the prior box, and convert to the log-space.
        In the model, we are predicting bounding box coordinates in this encoded form.
        :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
        :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
        :return: encoded bounding boxes, a tensor of size (n_priors, 4)
        """

        # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
        # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
        # See https://github.com/weiliu89/caffe/issues/155
        return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                          torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h

    def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
        """
        Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.
        They are decoded into center-size coordinates.
        This is the inverse of the function above.
        :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
        :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
        :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
        """

        return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                          torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h

    @classmethod
    def get_scale_aspect(cls,w=None,h=None,xy=None):
        """

        :param w: width of image . tensor
        :param h: height of image . tensor
        :param xy: coords of bounding box. tensor [:,4]
        :return: scale and aspect in tensor of shape 2.
        """
        if w is not None and h is not None:
            pass
        elif xy is not None :
            point=cls.xy_to_cxcy(xy)
            w = point[:,2]
            h = point[:, 3]
        else:
           raise TypeError("Input not correct")

        aspect = w / h
        scale = torch.sqrt(w * h)
        return aspect,scale




class clusterring():
    @classmethod
    def kmeans(cls,x,standardized=False,max_clusters=20,n_cluster=None):
        from sklearn.cluster import KMeans
        import seaborn as sns

        if standardized :
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
        if n_cluster is not None:
            kmeans = KMeans( n_clusters=n_cluster, init='k-means++')
            kmeans.fit(x)

            if len(x.shape)<3:
                if standardized :x = scaler.inverse_transform(x)
                frame = pd.DataFrame({'x':x[:,0], 'y':x[:,1],'cluster': kmeans.labels_})
                sns.scatterplot(data=frame, x="x", y="y", hue="cluster")
                plt.show()
                sns.countplot(x="cluster", data=frame)
                plt.show()
            return scaler.inverse_transform(kmeans.cluster_centers_)
        else:
            SSE = []
            for cluster in range(1, max_clusters):
                kmeans = KMeans( n_clusters=cluster, init='k-means++')
                kmeans.fit(x)
                SSE.append(kmeans.inertia_)

            # converting the results into a dataframe and plotting them
            frame = pd.DataFrame({'Cluster': range(1, 20), 'SSE': SSE})
            plt.figure(figsize=(12, 6))
            plt.plot(frame['Cluster'], frame['SSE'], marker='o')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            plt.show()



class DataCreation:
    def __init__(self, data_path=None, image_path_=None,max_image=100):
        self.data_path = data_path
        self.image_path = image_path_
        self.to_csv = True
        self.start_time=time.time()
        self.max_image=max_image

    def darker(self, data):

        pixel = ['pixel' + str(i) for i in range(784)]
        for i in range(data.shape[0]):
            labe = data.iloc[i].label
            pix_start = labe * 28 * 2
            start = random.randint(0, 54)
            finish = random.randint(1, 55 - start) + start

            data.iloc[i][['pixel' + str(j) for j in range(start + pix_start, finish + pix_start)]] = 255
            ar = np.array(data.iloc[i][pixel]).reshape((28, 28))
            fig = plt.figure()
            plt.imshow(ar)
            fig.savefig(str(self.image_path) + '/' + str(i) + '_' + str(labe) + '.png')
            plt.close(fig)

        data.to_csv(str(self.data_Path) + '/newData.csv')

    def circles_and_rectngles(self, data):
        length = 8
        pixel = ['pixel' + str(i) for i in range(784)]
        for i in range(1000):
            startRow = random.randint(0, 28 - length - 1)
            startCol = random.randint(0, 28 - length - 1)
            shp = random.choice([0, 1])
            data.iloc[i]['label'] = shp
            if shp:
                pixs = ['pixel' + str(startRow * 28 + startCol + k[0] * 28 + k[1]) for k in
                        combinations([length - 1 - i for i in range(length)], 2)]
            else:
                pixs = ['pixel' + str(startRow * 28 + startCol + k[0] * 28 + k[1]) for k in
                        combinations([i for i in range(length)], 2)]
            data.iloc[i][pixs] = 255

            ar = np.array(data.iloc[i][pixel]).reshape((28, 28))
            fig = plt.figure()
            plt.imshow(ar)
            fig.savefig(str(image_path) + '/' + str(i) + "_" + str(shp) + '.png')
            plt.close(fig)

        data.to_csv(str(dataPath) + '/newData.csv')

    def get_fig(self, data, img_no, label):
        if self.image_path is not None:
            fig = plt.figure()
            plt.imshow(data.astype(int))
            fig.savefig(str(self.image_path) + '/' + str(img_no) + "_" + str(label) + '.png')
            plt.close(fig)

    def black_image(self, data_count=100, size=(112, 112), ret=True):
        pixel = ['pixel' + str(i) for i in range(size[0] * size[1])]
        data = pd.DataFrame(columns=['label'] + pixel, index=[i for i in range(data_count)])
        for i in range(data_count):
            ar = np.zeros(shape=size, dtype=int)
            data.loc[i] = [0] + list(ar.flatten())
            if not ret:
                self.get_fig(ar, i, 0)
        if ret:
            return data
        else:
            data.to_csv(str(self.data_path) + '/newData.csv')

    def shifter(self, data, data_count=100, size=(112, 112),size2=(28,28)):
        data_big = self.black_image(data_count, size)
        pixel = ['pixel' + str(i) for i in range(size[0] * size[1])]
        pixel2 = ['pixel' + str(i) for i in range(size2[0] * size2[1])]
        for i in range(data_count):
            temp = np.array(data_big.iloc[i][pixel]).reshape(size)

            start_row, start_col = random.choice(range(0, size[0] - size2[0] + 1)),  \
                                        random.choice(range(0, size[1] - size2[1] + 1))
            temp[start_row:start_row + size2[0], start_col:start_col + size2[1]] = \
                np.array(data.iloc[i][pixel2]).reshape(size2)
            data_big.iloc[i][pixel] = temp.flatten()
            data_big.iloc[i]['label'] = data.iloc[i]['label']
            self.get_fig(temp, i, data.iloc[i]['label'])
            if i % 500 == 0:
                print(str(i) + " completed")
                print("time elapsed: " +str(time.time()-self.start_time))
        if self.to_csv:
            data_big.to_csv(str(self.data_path) + '/newData.csv')
        return data_big

    def scaler(self, data, data_count=100, size=(112, 112),size2=(28,28),scales=4):
        data_big = self.black_image(data_count, size)
        pixel = ['pixel' + str(i) for i in range(size[0] * size[1])]
        pixel2 = ['pixel' + str(i) for i in range(size2[0] * size2[1])]
        for i in range(data_count):
            temp = np.array(data_big.iloc[i][pixel]).reshape(size)
            scale=random.choice(range(1,scales+1))
            res = cv2.resize(np.array(data.iloc[i][pixel2]).reshape(size2).astype('float32')
                             , dsize=(size2[0]*scale, size2[1]*scale)
                             , interpolation=cv2.INTER_CUBIC)
            temp[0: size2[0]*scale, 0:size2[1]*scale] = np.array(res)
            data_big.iloc[i][pixel] = temp.flatten()
            data_big.iloc[i]['label'] = data.iloc[i]['label']
            self.get_fig(temp, i, data.iloc[i]['label'])
            if i%500==0:
                print(str(i)+" completed")
                print("time elapsed: " + str(time.time() - self.start_time))
        if self.to_csv:
            data_big.to_csv(str(self.data_path) + '/newData.csv')
        return data_big

    def coords(self,data):

            x,y = np.argmax(data,axis=0),np.argmax(data,axis=1)

            x2, y2 = np.argmax( np.flip(data, axis=0), axis=0), np.argmax( np.flip(data, axis=1), axis=1)
            x1,y1=min((np.trim_zeros(x))),min((np.trim_zeros(y)))
            x2, y2 = min((np.trim_zeros(x2))),min((np.trim_zeros(y2)))
            x1,y1,x2,y2 = x1, y1, data.shape[0] - x2-1, data.shape[1] - y2-1
            return [x1,y1,x2,y2]

    def draw_box(self,x1=0,y1=0,x2=0,y2=0,data=None,dim=1,color_intensity=(0,200,0),save_loc=None,msg= None):
        torch.set_printoptions(precision=2)
        x1,x2,y1,y2=int(x1),int(x2),int(y1),int(y2)
        if data is not None :
            cv2.imwrite(save_loc, data)
        img = cv2.imread(save_loc)

        cv2.rectangle(img, (y1, x1), (y2, x2), color_intensity,0)
        if msg is not None:
            w=9
            h=3
            #cv2.rectangle(img, (y1, x1), (y1 + w, x1 + h), color_intensity, 0)
            text = "{:.2f}: {:.2f}".format(float(msg[0]), float(msg[1]))
            #cv2.putText(img, text, (y1, x1),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=(0,100,0),thickness= 0)
            save_loc=save_loc.replace('.png','_pred_'+text+'.png')
        cv2.imwrite(save_loc,img)
    def draw_box(self,locs,data=None,color_intensity=[(0,200,0)],scale=None,save_loc=None,msg= None,imgscale=2):
        """

        :param locs: [[prediction,prob,x1,y1,x2,y2],...] |each terminal boxes will have  prediction,prob,x1,y1,x2,y2
        :param data: tesnor |if imgae loc is not provide than image data need to be provided
        :param color_intensity:list of (x,y,z)| color for each boxes
        :param save_loc: str|where to save the image after making boxes
        :param scale: int| multipler by which each loc multiplied
        :param msg: str | new name for image saving
        :return: image Nond| as it saves the image to save_locs
        """
        torch.set_printoptions(precision=2)
        img=None
        imgscale=1
        scale=imgscale*scale
        for i in range(len(locs)):
         for j in range(len(locs[i])):
            actual=locs[i][j][0]
            pred=locs[i][j][1]
            x1,x2,y1,y2=locs[i][j][2],locs[i][j][4],locs[i][j][3],locs[i][j][5]
            if save_loc is not None:x1, x2, y1, y2=int(x1*scale),int(x2*scale),int(y1*scale),int(y2*scale)
            if data is not None :
                    cv2.imwrite(save_loc, data)
            if img is  None:
                img = cv2.imread(save_loc)
                img = cv2.resize(img,
                                 dsize=(28*4*imgscale, 28*4*imgscale)
                                 , interpolation=cv2.INTER_CUBIC)

            img=cv2.rectangle(img, (y1, x1), (y2, x2), color_intensity[i],0)
            text = "{:.2f} : {:.2f}".format(float(actual), float(pred))
            img =  cv2.putText(img, text,org= (y1, x1),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.2, color=(255,255,255),thickness=1,lineType= cv2.LINE_AA,bottomLeftOrigin = False)
        save_loc=save_loc.replace('.png','_pred_'+'.png')
        cv2.imwrite(save_loc,img)

    def rub_box(self,data,dim=1,color_intensity=0,save_loc=None):
        data[:,:, dim]= color_intensity
        cv2.imwrite(save_loc, data)



    def create_localization(self,data, data_count=10, size=(28, 28)):
        ret_data = data[0:data_count]
        ret_data['localisation'] = ""
        pixel = ['pixel' + str(i) for i in range(size[0] * size[1])]
        for i in range(data_count):
            data_temp=data.iloc[i]
            temp=np.array(data.iloc[i][pixel]).reshape(size)
            temp=np.where(temp>0,225,0)
            x1,y1,x2,y2=self.coords(temp)
            ret_data.loc[i,['localisation']] = packing.pack([x1,y1,x2,y2])
            if len(temp.shape)<3 :
                temp=np.expand_dims(temp,axis=2)
                temp=np.concatenate((temp,np.zeros(temp.shape),np.zeros(temp.shape)),axis=2)
            if self.image_path is not None and i<self.max_image: self.draw_box(x1,y1,x2,y2,data=temp,save_loc=self.image_path +'/'+str(i)+'_'+str(data_temp['label'])+".png")
            if i % 500 == 0:
                print(str(i) + " completed")
                print("time elapsed: " + str(time.time() - self.start_time))
        if self.data_path is not None:ret_data.to_csv(str(self.data_path) + '/newData.csv')


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params



if __name__ == "__main__":
    import unittest
    class test_DataCreation():#unittest.TestCase
        def __init__(self):
            self.obj=DataCreation(image_path_='/home/pooja/PycharmProjects/digitRecognizer/test_rough/images')
        def test_create_localization(self):
            file=pd.read_csv('/home/pooja/PycharmProjects/digitRecognizer/data/dataCreated/holdout.csv')
            fin=self.obj.create_localization(data=file,size=(28,28))

    # c = test_DataCreation()
    # c.test_create_localization()
    class test_vison_utils():  # unittest.TestCase
        def __init__(self):
            self.obj = vison_utils

        def test_get_detection_pipeline(self):
            n_classes=11
            batch=32
            boxes=50
            classes=[i for i in range(n_classes) ]
            scores=torch.rand((batch,n_classes*boxes))
            bounding_locs = torch.rand((batch,boxes,4))
            bounding_locs,_=bounding_locs.sort(dim=2) #doing this so x2,y2>=x1,y1
            bounding_locs= bounding_locs.reshape(batch,boxes*4)



            t1a = self.obj.create_boxes(scores,n_classes,bounding_locs)
            assert t1a.shape==(batch,boxes,6)
            t2=self.obj.non_max_suppression( t1a[0], classes, background_class=11, pred_thresh=0.3, overlap_thresh=1)
            assert t2.shape[1]==6
            bounding_locs = torch.rand((batch, boxes, 4))
            bounding_locs, _ = bounding_locs.sort(dim=2)
            t1b = torch.rand((batch,boxes,2))
            t1b[:,:,0]=torch.floor(t1b[:,:,0]*n_classes)
            t1b[:,:, 1]=1
            t1b=torch.cat([t1b,bounding_locs],dim=2)


            t3 =self.obj.tp_fp_fn(t2, t1b[0], classes[:-1], iou_threshold=0.5)


    c = test_vison_utils()
    c.test_get_detection_pipeline()






