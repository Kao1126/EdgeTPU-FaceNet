import os 
import numpy as np
from PIL import Image
from edgetpu.classification.engine import ClassificationEngine
import time
from config import*


def main(txt_path, image_path):
    pair = read_pairs(txt_path)
    path_list, issame_list = get_paths(image_path, pair)     
    
    min_THREAD = 0.5
    max_THREAD = 2
    step = 0.5
    accuracy_list = [] 
    THREAD_list = []

    print("-------------------------------------------------")

    for THREAD in np.arange(min_THREAD, max_THREAD, step):
        THREAD_list.append(THREAD)
        tick1 = time.time()
        accuracy = classify(path_list, issame_list, THREAD)
        accuracy_list.append(accuracy)
        tick2 = time.time()
        print("THREAD:{:.2f} finish!".format(THREAD))
        print("Accuracy:{:.2f}".format(accuracy))
        print("Time: {:.2f}".format(tick2 - tick1))
        print("-------------------------------------------------")
        save_txt(accuracy, THREAD)

    
    print("max_accuracy:", max(accuracy_list))

def save_txt(accuracy_list, THREAD_list):

    with open('lfw_score.txt', 'a+') as f:

        f.write(' Thread:')
        f.write('{:2f}'.format(THREAD_list))
        f.write(' Accuracy:')
        f.write('{:2f}\n'.format(accuracy_list))

def classify(path_list_, same_list_, THREAD):
    engine  = ClassificationEngine(FaceNet_weight)
    pred = bool()
    correct = 0
    for same_index, pair in enumerate(path_list_):
        picture1_embs = []
        picture2_embs = []
    
        for k, img in enumerate(pair):
            img = Image.open(img)
            img = np.asarray(img).flatten()
            result = engine.ClassifyWithInputTensor(img, top_k=200, threshold=-0.5)
            result.sort(key=takeSecond)
            
            if k == 1:
                for i in range(0, len(result)):
                    picture1_embs.append(result[i][1])
            else:
                for i in range(0, len(result)):
                    picture2_embs.append(result[i][1])
        
        picture1_embs = np.array(picture1_embs)
        picture2_embs = np.array(picture2_embs)
        
        diff = np.mean(np.square(picture1_embs-picture2_embs))
        
        if diff < THREAD:
            pred = True
        else:
            pred = False

        if pred == same_list_[same_index]:
            correct += 1


    accuracy = correct/len(path_list_)

    return accuracy

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    path0 = ''
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]))+'.jpg'
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]))+'.jpg'
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]))+'.jpg'
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]))+'.jpg'
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list.append([path0, path1])
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def takeSecond(elem):
  return elem[0]


if __name__ == "__main__":
    main('lfw_pairs.txt',  'lfw')