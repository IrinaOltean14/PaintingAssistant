import os
import time
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import wandb


def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = setup_device()

def setup_wandb():
    wandb.login()
    wandb.init(
        project="ResNet50-FT",
        name="Attempt7",
        config={
            "model": "ResNet50",
            "experiment_type": "FineTuning",
            "optimizer": "SGD",
            "learning_rate": 1e-3,
            "momentum": 0.9,
            "weight_decay": 1e-5,
            "scheduler": "StepLR",
            "step_size": 15,
            "gamma": 0.1,
            "batch_size": 128,
            "epochs_phase": 25,
            "patience": 7
        }
    )
    return wandb.config


def load_csv(csv_path):
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        df['image_path'] = df['IMAGE_FILE'].apply(lambda x: os.path.join(images_path, x))
        return df[['image_path', 'SCHOOL', 'TYPE']]


def load_and_encode_data(train_csv, val_csv, test_csv, images_path):
    train_df = load_csv(train_csv)
    val_df = load_csv(val_csv)
    test_df = load_csv(test_csv)

    school_encoder = LabelEncoder()
    type_encoder = LabelEncoder()

    train_df['SCHOOL'] = school_encoder.fit_transform(train_df['SCHOOL'])
    train_df['TYPE'] = type_encoder.fit_transform(train_df['TYPE'])

    val_df['SCHOOL'] = school_encoder.transform(val_df['SCHOOL'])
    val_df['TYPE'] = type_encoder.transform(val_df['TYPE'])

    test_df['SCHOOL'] = school_encoder.transform(test_df['SCHOOL'])
    test_df['TYPE'] = type_encoder.transform(test_df['TYPE'])

    return train_df, val_df, test_df, school_encoder, type_encoder

class BALANCEDDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        school_label = self.dataframe.iloc[idx]['SCHOOL']
        type_label = self.dataframe.iloc[idx]['TYPE']

        return image, torch.tensor(school_label, dtype=torch.long), torch.tensor(type_label, dtype=torch.long)

def get_dataloaders(train_df, val_df, test_df, batch_size):
    train_t = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = BALANCEDDataset(train_df, train_t)
    val_dataset = BALANCEDDataset(val_df, test_val_transforms)
    test_dataset = BALANCEDDataset(test_df, test_val_transforms)

    return {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8),
    }, {
        'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)
    }



def train_model(model, criterion_school, criterion_type, optimizer, scheduler, dataloaders, sizes, save_path, patience=7, epochs=25, num_classes_school = 8, num_classes_type = 8):
    best_acc, no_improve = 0.0, 0
    torch.save(model.state_dict(), save_path)

    for epoch in range(epochs):
        print(f'\nEpoch {epoch}/{epochs - 1}\n{"-" * 10}')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects_school, running_corrects_type = 0.0, 0, 0

            for inputs, school_labels, type_labels in tqdm(dataloaders[phase], desc=f"{phase} Phase"):
                inputs, school_labels, type_labels = inputs.to(device), school_labels.to(device), type_labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs_school = outputs[:, :num_classes_school]  
                    outputs_type = outputs[:, num_classes_school:]
                    loss_school = criterion_school(outputs_school, school_labels)
                    loss_type = criterion_type(outputs_type, type_labels)
                    loss = 0.6 * loss_school + 0.4 * loss_type

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    _, preds_school = torch.max(outputs_school, 1)
                    _, preds_type = torch.max(outputs_type, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects_school += torch.sum(preds_school == school_labels.data)
                running_corrects_type += torch.sum(preds_type == type_labels.data)

            epoch_loss = running_loss / sizes[phase]
            epoch_acc_school = running_corrects_school / sizes[phase]
            epoch_acc_type = running_corrects_type / sizes[phase]
            epoch_avg_acc = (epoch_acc_school + epoch_acc_type) / 2

            if phase == 'val':
                scheduler.step()

            wandb.log({
                f"{phase}_loss": epoch_loss,
                f"{phase}_school_acc": epoch_acc_school,
                f"{phase}_type_acc": epoch_acc_type,
                f"{phase}_avg_acc": epoch_avg_acc,
                "epoch": epoch
            })

            print(f'{phase} Loss: {epoch_loss:.4f} '
                  f'SCHOOL Acc: {epoch_acc_school:.4f} '
                  f'TYPE Acc: {epoch_acc_type:.4f} '
                  f'AVG Acc: {epoch_avg_acc:.4f}')

            if phase == 'val':
                if epoch_avg_acc > best_acc:
                    best_acc = epoch_avg_acc
                    no_improve = 0
                    print(f"New best model found! Saving to {save_path}")
                    torch.save(model.state_dict(), save_path)
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch} due to no improvement.")
                        model.load_state_dict(torch.load(save_path))
                        return model

    print(f'Best val Avg Acc: {best_acc:.4f}')
    model.load_state_dict(torch.load(save_path))
    return model

if __name__ == "__main__":
    config = setup_wandb()

    images_path = "../dataset/Images"
    train_df, val_df, test_df, school_encoder, type_encoder = load_and_encode_data(
        '../dataset/my_train.csv',
        '../dataset/my_val.csv',
        '../dataset/my_test.csv',
        images_path
    )

    num_classes_school = len(school_encoder.classes_)
    num_classes_type = len(type_encoder.classes_)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes_school + num_classes_type)
    )
    model.to(device)

    dataloaders, sizes = get_dataloaders(train_df, val_df, test_df, config.batch_size)

    criterion_school = nn.CrossEntropyLoss()
    criterion_type = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=config.step_size, gamma=config.gamma)

    model = train_model(model, criterion_school, criterion_type, optimizer_ft, scheduler, dataloaders, sizes,
                        save_path="../models/attempt7.pth",
                        patience=config.patience,
                        epochs=25, num_classes_school=num_classes_school, num_classes_type=num_classes_type)
    wandb.finish()

