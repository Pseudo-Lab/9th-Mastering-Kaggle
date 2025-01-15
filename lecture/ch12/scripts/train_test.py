import os
import json
from io import BytesIO

import numpy as np
from tqdm import tqdm

import torch

from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

import mlflow
import wandb
from torch.utils.tensorboard import SummaryWriter

from .utils import apply_label_smoothing
from .utils import plot_confusion_matrix

def train_model(model, loader_train, loader_valid, criterion, optimizer, scheduler, args, device, save_path='./results'):

    os.makedirs(f"{save_path}/{args.model}/plots", exist_ok=True)
    os.makedirs(f"{save_path}/{args.model}/weights", exist_ok=True)

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []
    best_valid_loss = float('inf')
    roc_auc_scores = []

    metrics = []

    # 모델의 모든 파라미터 크기 가져오기
    model_name = f"{model.__class__.__name__} ({args.model})"
    total_params = sum(p.numel() for p in model.parameters())

    for epoch in tqdm(range(args.epochs), desc='epoch', leave=True):
        # Training loop
        model.train()
        epoch_train_loss, correct_train = 0, 0
        total_train = 0

        for images, labels in tqdm(loader_train, desc='train', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_losses.append(epoch_train_loss / len(loader_train))
        train_accuracies.append(correct_train / total_train)

        # print(f'Epoch [{epoch+1}/{args.epochs}] - Train Loss: {epoch_train_loss/len(loader_train):.4f}')

        # Validation loop
        model.eval()
        epoch_valid_loss, correct_valid = 0, 0
        total_valid = 0
        preds_list, true_onehot_list = [], []

        with torch.no_grad():
            for images, labels in tqdm(loader_valid, desc='valid', leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_valid_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct_valid += (preds == labels).sum().item()
                total_valid += labels.size(0)

                preds_list.extend(torch.softmax(outputs.cpu(), dim=1).numpy())
                true_onehot_list.extend(torch.eye(4)[labels.cpu()].numpy())

        valid_losses.append(epoch_valid_loss / len(loader_valid))
        valid_accuracies.append(correct_valid / total_valid)

        roc_auc = roc_auc_score(true_onehot_list, preds_list)
        roc_auc_scores.append(roc_auc)
        
        # Confusion Matrix
        cm = confusion_matrix(
            np.argmax(true_onehot_list, axis=1),
            np.argmax(preds_list, axis=1)
        )

        cm_save_path = f"{save_path}/{args.model}/plots/confusion_matrix_epoch_{epoch + 1}.png"

        plot_confusion_matrix(cm, title=f"Confusion Matrix (Epoch {epoch + 1})", filename=cm_save_path)


        metrics.append({
            "epoch": epoch + 1,
            "loss": epoch_valid_loss / len(loader_valid),
            "roc_auc": roc_auc
        })

        if args.save_log == "wandb":
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_losses[-1],
                "valid_loss": valid_losses[-1],
                "valid_accuracy": valid_accuracies[-1],
                "roc_auc": roc_auc,
                "confusion_matrix": wandb.Image(cm_save_path)  # Log confusion matrix as image
            })

        elif args.save_log == "mlflow":
            mlflow.log_metric("train_loss", train_losses[-1], step=epoch + 1)
            mlflow.log_metric("valid_loss", valid_losses[-1], step=epoch + 1)
            mlflow.log_metric("valid_accuracy", valid_accuracies[-1], step=epoch + 1)
            mlflow.log_metric("roc_auc", roc_auc, step=epoch + 1)

            mlflow.log_artifact(cm_save_path, artifact_path=f"confusion_matrices/epoch_{epoch + 1}")


        elif args.save_log == "tensorboard":
            tensorboard_save_path = f"{save_path}/{args.model}/logs"
            os.makedirs(tensorboard_save_path, exist_ok=True)
            writer = SummaryWriter(tensorboard_save_path)

            writer.add_scalar('Loss/train', train_losses[-1], epoch + 1)
            writer.add_scalar('Loss/valid', valid_losses[-1], epoch + 1)
            writer.add_scalar('Accuracy/train', train_accuracies[-1], epoch + 1)
            writer.add_scalar('Accuracy/valid', valid_accuracies[-1], epoch + 1)
            writer.add_scalar('ROC_AUC', roc_auc, epoch + 1)

            # Log confusion matrix as image to TensorBoard
            buffer = BytesIO()
            buffer.seek(0)
            writer.add_image(f"Confusion_Matrix/Epoch_{epoch + 1}", np.array(plt.imread(buffer)), epoch + 1)

            writer.close()
        # Save the best model
        if epoch_valid_loss / len(loader_valid) < best_valid_loss:
            best_valid_loss = epoch_valid_loss / len(loader_valid)
            torch.save(model.state_dict(), f"{save_path}/{args.model}/weights/{args.model}_best.pth")

        print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_losses[-1]:.4f}, Valid Loss: {valid_losses[-1]:.4f}, Valid Accuracy: {valid_accuracies[-1]:.4f}, ROC AUC: {roc_auc:.4f}")

    # Save the final model
    torch.save(model.state_dict(), f"{save_path}/{args.model}/weights/{args.model}_final.pth")
    
    # Save plots
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.savefig(f"{save_path}/{args.model}/plots/train_valid_loss.png")

    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, args.epochs + 1), valid_accuracies, label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()
    plt.savefig(f"{save_path}/{args.model}/plots/train_valid_accuracy.png")
    
    plt.figure()
    plt.plot(range(1, args.epochs + 1), roc_auc_scores, label='ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.title('Validation ROC AUC Across Epochs')
    plt.legend()
    plt.savefig(f"{save_path}/{args.model}/plots/roc_auc.png")
    plt.close()

    # Confusion Matrices
    cm_before_ls = confusion_matrix(
        np.argmax(true_onehot_list, axis=1),
        np.argmax(preds_list, axis=1)
    )
    plot_confusion_matrix(cm_before_ls, "Confusion Matrix (Before Label Smoothing)", f"{save_path}/{args.model}/plots/confusion_matrix_before_ls.png")

    try:
        with open('model_parameters_sizes.json', 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}

    data[model_name] = {
        "total_params": f"{total_params:,}",
        "metrics": metrics
    }

    with open('model_parameters_sizes.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return model

def test_model(model, loader_test, loader_TTA, args, device, submission, test, save_path='./results', num_TTA=7):
    model.eval()
    preds_test = np.zeros((len(test), 4))
    with torch.no_grad():
        for i, images in tqdm(enumerate(loader_test), desc='test', leave=True):
            images = images.to(device)
            outputs = model(images)
            preds_test[i*args.batch:(i+1)*args.batch] += torch.softmax(outputs.cpu(), dim=1).numpy()

    num_TTA = num_TTA
    preds_tta = np.zeros((len(test), 4))
    for _ in tqdm(range(num_TTA), desc='TTA', leave=True):
        with torch.no_grad():
            for i, images in tqdm(enumerate(loader_TTA), desc='tta_test', leave=False):
                images = images.to(device)
                outputs = model(images)
                preds_tta[i*args.batch:(i+1)*args.batch] += torch.softmax(outputs.cpu(), dim=1).numpy()
    preds_tta /= num_TTA

    # Save predictions
    submission_test = submission.copy()
    submission_tta = submission.copy()

    submission_test[['healthy', 'multiple_diseases', 'rust', 'scab']] = preds_test
    submission_test.to_csv(f'{save_path}/{args.model}/{args.model}_submission_test.csv', index=False)

    submission_tta[['healthy', 'multiple_diseases', 'rust', 'scab']] = preds_tta
    submission_tta.to_csv(f'{save_path}/{args.model}/{args.model}_submission_tta.csv', index=False)

    alpha, threshold = 0.001, 0.999
    
    submission_test_ls = submission_test.copy()
    submission_tta_ls = submission_tta.copy()

    target = ['healthy', 'multiple_diseases', 'rust', 'scab'] # 타깃값 열 이름
    
    submission_test_ls[target] = apply_label_smoothing(submission_test_ls, target, alpha, threshold)
    submission_tta_ls[target] = apply_label_smoothing(submission_tta_ls, target, alpha, threshold)
    submission_test_ls.to_csv(f'{save_path}/{args.model}/{args.model}_submission_test_ls.csv', index=False)
    submission_tta_ls.to_csv(f'{save_path}/{args.model}/{args.model}_submission_tta_ls.csv', index=False)