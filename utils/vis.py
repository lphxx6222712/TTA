import numpy as np
import os
import os.path as osp
import cv2


def vis_multi_class(imt, ant, pred, save_dir,n_class=3,is_boundary=True):

    for idx in range(pred.shape[0]):
        img = gray2rgbimage(imt[idx])
        pred_img = draw_img(imt[idx], pred[idx],is_boundary=is_boundary,n_class=n_class)
        if(ant is None):
            cv2.imwrite(osp.join(save_dir, '%02d.png'%idx), np.hstack((img, pred_img)).astype('uint8'))
        else:
            # Draw GT
            ant_img = draw_img(imt[idx], ant[idx],is_boundary=is_boundary, n_class=n_class)            
            cv2.imwrite(osp.join(save_dir, '%02d.png'%idx), np.hstack((img, pred_img, ant_img)).astype('uint8'))

def vis_multi_class_w_entropy(imt, ant, pred, entropy, save_dir,n_class=3,is_boundary=True):

    for idx in range(pred.shape[0]):
        img = gray2rgbimage(imt[idx])
        pred_img = draw_img(imt[idx], pred[idx],is_boundary=is_boundary,n_class=n_class)

        entropy_img = draw_entropy(imt[idx], entropy[idx])
        if(ant is None):
            cv2.imwrite(osp.join(save_dir, '%02d.png'%idx), np.hstack((img, pred_img)).astype('uint8'))
        else:
            # Draw GT
            ant_img = draw_img(imt[idx], ant[idx],is_boundary=is_boundary, n_class=n_class)
            cv2.imwrite(osp.join(save_dir, '%02d.png'%idx), np.hstack((img, pred_img, entropy_img[0], ant_img)).astype('uint8'))



def draw_img(img, seg, title = None, is_boundary = False,n_class=3):
    label_set = [i+1 for i in range(n_class)]
    color_set = {
                    1:(255,0,0),
                    2:(0,255,0),
                    3:(0,0,255),
                    4:(0,255,255),
                }

    img = gray2rgbimage(img)
    #img = np.zeros_like(img)
    if(title is not None):
        img = cv2.putText(img, title, (16, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # white title
    for draw_label in label_set:
        if(is_boundary):
            img[:, :, 0][seg == draw_label] = color_set[draw_label][0]
            img[:, :, 1][seg == draw_label] = color_set[draw_label][1]
            img[:, :, 2][seg == draw_label] = color_set[draw_label][2]
        else:
            seg_label = (seg == draw_label).astype('uint8')
            image, contours, hierarchy = cv2.findContours(seg_label,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            color= color_set[draw_label]
            cv2.drawContours(img, contours, -1, color, 1)     #粗细    
    return img

def draw_entropy(img, entropy, title = None, is_boundary = False,n_class=3):
    img = gray2rgbimage(img).astype(np.uint8)
    entropy_maps = np.zeros([4,img.shape[0],img.shape[1],3],np.uint8)
    for i in range(entropy.shape[0]):
        entropy_img = entropy[i]
        #norm_img = entropy_img.astype(np.uint8)*255
        norm_img = (entropy_img-np.min(entropy_img))/(np.max(entropy_img)-np.min(entropy_img))
        norm_img = (norm_img*255).astype(np.uint8)
        heat_img = cv2.applyColorMap(norm_img,cv2.COLORMAP_JET)
        entropy_maps[i,:,:,:]=heat_img
        #heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    img0 = cv2.addWeighted(img,0.8,entropy_maps[0,:,:,:],0.2,0)
    img1 = cv2.addWeighted(img,0.8,entropy_maps[1,:,:,:],0.2,0)
    img2 = cv2.addWeighted(img,0.8,entropy_maps[2,:,:,:],0.2,0)
    img3 = cv2.addWeighted(img,0.8,entropy_maps[3,:,:,:],0.2,0)
    return entropy_maps

def gray2rgbimage(image):
    a,b = image.shape
    new_img = np.ones((a,b,3))
    new_img[:,:,0] = image.reshape((a,b)).astype('uint8')
    new_img[:,:,1] = image.reshape((a,b)).astype('uint8')
    new_img[:,:,2] = image.reshape((a,b)).astype('uint8')
    return new_img








