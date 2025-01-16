import os
import random
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

from transformers import get_cosine_schedule_with_warmup

import wandb
import mlflow
from torch.utils.tensorboard import SummaryWriter

from scripts.models import get_model
from scripts.utils import ImageDataset
from scripts.utils import seed_worker
from scripts.train_test import train_model, test_model

def main(args, device=None):
    # 시드값 고정
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # == [ 데이터 로드 및 경로 설정 ] =============================
    data_path = './data'
    results_path = f'./results'

    train = pd.read_csv(data_path + '/train.csv')
    test = pd.read_csv(data_path + '/test.csv')
    submission = pd.read_csv(data_path + '/sample_submission.csv')

    train, valid = train_test_split(
        train,
        test_size=0.1,
        stratify=train[['healthy', 'multiple_diseases', 'rust', 'scab']],
        random_state=50
    )

    # == [ 데이터 변환 설정 ] =====================================
    transform_train = A.Compose([
        A.Resize(args.img_size[0], args.img_size[1]), # 이미지 크기 조절
        A.RandomBrightnessContrast(brightness_limit=0.2, # 밝기 대비 조절
                                contrast_limit=0.2, p=0.3),
        A.VerticalFlip(p=0.2),    # 상하 대칭 변환
        A.HorizontalFlip(p=0.5),  # 좌우 대칭 변환
        A.ShiftScaleRotate(       # 이동, 스케일링, 회전 변환
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=30, p=0.3),
        A.OneOf([A.Emboss(p=1),   # 양각화, 날카로움, 블러 효과
                A.Sharpen(p=1),
                A.Blur(p=1)], p=0.3),
        A.PiecewiseAffine(p=0.3), # 어파인 변환
        A.Normalize(),            # 정규화 변환
        ToTensorV2()              # 텐서로 변환
    ])


    transform_test = A.Compose([
        A.Resize(args.img_size[0], args.img_size[1]), # 이미지 크기 조절
        A.Normalize(),      # 정규화 변환
        ToTensorV2()        # 텐서로 변환
    ])

    # == [ 데이터셋 및 데이터 로더 생성 ] ==========================
    dataset_train = ImageDataset(train, img_dir=data_path + "/images/", transform=transform_train)
    dataset_valid = ImageDataset(valid, img_dir=data_path + "/images/", transform=transform_test)
    dataset_test = ImageDataset(test, img_dir=data_path + "/images/", transform=transform_test, is_test=True)

    g = torch.Generator()
    g.manual_seed(0)

    loader_train = DataLoader(dataset_train, batch_size=args.batch, shuffle=True, worker_init_fn=seed_worker, generator=g, num_workers=2)
    loader_valid = DataLoader(dataset_valid, batch_size=args.batch, shuffle=False, worker_init_fn=seed_worker, generator=g, num_workers=2)
    loader_test = DataLoader(dataset_test, batch_size=args.batch, shuffle=False, worker_init_fn=seed_worker, generator=g, num_workers=2)

    # model = get_model(args.model, num_classes=4, pretrained=True, device=device)
    model = get_model(args.model, args.img_size, num_classes=4, pretrained=True, device=device)
    # model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=4)
    # model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006, weight_decay=0.0001)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(loader_train) * 3, num_training_steps=len(loader_train) * args.epochs)

    # WandB 또는 MLflow 초기화
    if args.save_log == "wandb":
        wandb.init(project="Kaggle_test", config=args, name=args.model)
        wandb.watch(model, log="all", log_freq=100)
    elif args.save_log == "mlflow":
        mlflow.start_run(run_name=args.model)
    elif args.save_log == "tensorboard":
        tb_writer = SummaryWriter(log_dir=f"{results_path}/{args.model}/tensorboard_logs")
    else:
        print("Logging is disabled.")
    
    model=train_model(
        model=model,
        loader_train=loader_train,
        loader_valid=loader_valid,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        device=device
        )

    # WandB 또는 MLflow 종료
    if args.save_log == "wandb":
        wandb.finish()
    elif args.save_log == "mlflow":
        mlflow.end_run()
    elif args.save_log == "tensorboard":
        tb_writer.close()

    # == [ 테스트 및 TTA 예측 ] ===================================
    dataset_test = ImageDataset(test, img_dir=data_path+"/images/", transform=transform_test, is_test=True)
    dataset_TTA = ImageDataset(test, img_dir=data_path+"/images/", transform=transform_train, is_test=True)
    
    loader_test = DataLoader(dataset_test, batch_size=args.batch, shuffle=False, worker_init_fn=seed_worker, generator=g, num_workers=2)
    loader_TTA = DataLoader(dataset_TTA, batch_size=args.batch, shuffle=False, worker_init_fn=seed_worker, generator=g, num_workers=2)
    
    test_model(
        model=model,
        loader_test=loader_test,
        loader_TTA=loader_TTA,
        args=args,
        device=device,
        submission=submission,
        test=test,
        save_path=results_path,
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, default=8, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=7, help='number of epochs')
    parser.add_argument('-m', '--model', type=str, default='tf_efficientnet_b7.ns_jft_in1k', help='timm model name')
    parser.add_argument('-s', '--seed', type=int, default=50, help='seed value')
    parser.add_argument('-i', '--img_size',type=int, nargs=2, default=(450, 640), help='image size')
    #vit, swin -> (224, 224), efficientnet -> (224, 224) or (450, 640)
    parser.add_argument('-l', '--save_log', type=str, default='none', help='save log to [none, wandb, tensorboard, mlflow]')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args, device=device)