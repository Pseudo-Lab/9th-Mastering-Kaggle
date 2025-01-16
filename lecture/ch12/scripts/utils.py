import cv2
import random

import numpy as np

import torch
from torch.utils.data import Dataset

import seaborn as sns
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    # 초기화 메서드(생성자)
    def __init__(self, df, img_dir='./', transform=None, is_test=False):
        super().__init__() # 상속받은 Dataset의 __init__() 메서드 호출
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    # 데이터셋 크기 반환 메서드
    def __len__(self):
        return len(self.df)

    # 인덱스(idx)에 해당하는 데이터 반환 메서드
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]             # 이미지 ID
        img_path = self.img_dir + img_id + '.jpg' # 이미지 파일 경로
        image = cv2.imread(img_path)         # 이미지 파일 읽기
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 이미지 색상 보정
        # 이미지 변환
        if self.transform is not None:
            image = self.transform(image=image)['image']
        # 테스트 데이터면 이미지 데이터만 반환, 그렇지 않으면 타깃값도 반환
        if self.is_test:
            return image # 테스트용일 때
        else:
            # 타깃값 4개 중 가장 큰 값의 인덱스
            label = np.argmax(self.df.iloc[idx, 1:5])
            return image, label # 훈련/검증용일 때

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def apply_label_smoothing(df, target, alpha, threshold):
    # 타깃값 복사
    df_target = df[target].copy()
    k = len(target) # 타깃값 개수

    for idx, row in df_target.iterrows():
        if (row > threshold).any():         # 임계값을 넘는 타깃값인지 여부 판단
            row = (1 - alpha)*row + alpha/k # 레이블 스무딩 적용
            df_target.iloc[idx] = row       # 레이블 스무딩을 적용한 값으로 변환
    return df_target # 레이블 스무딩을 적용한 타깃값 반환

def save_plot(x, y_values, labels, xlabel, ylabel, title, filename):
    plt.figure()
    for y, label in zip(y_values, labels):
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrix(cm, title, filename, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig(filename)
    plt.close()