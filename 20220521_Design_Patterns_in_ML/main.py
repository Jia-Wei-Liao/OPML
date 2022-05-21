from torch.nn import CrossEntropyLoss, Linear
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
import lib

data_list = lib.data.read_data_csv(
    "data/label.csv",
    dtype={"category": int},
    rename_map={"filename": "input", "category": "label"}
)
train_list, val_list = train_test_split(
    data_list,
    test_size=0.3,
    stratify=[d["label"] for d in data_list]
)
transforms = [
    lib.transforms.ReadImage("input", "data/image"),
    lib.transforms.Resize("input", (128, 128)),
    lib.transforms.ToTensor(["input", "label"])
]
train_loader = DataLoader(
    lib.data.Dataset(train_list, transforms),
    batch_size=32,
    collate_fn=lib.data.CollationFunction(["input", "label"]),
    shuffle=True
)
val_loader   = DataLoader(
    lib.data.Dataset(val_list, transforms),
    batch_size=32,
    collate_fn=lib.data.CollationFunction(["input", "label"])
)
network    = resnet18()
network.fc = Linear(network.fc.in_features, 219)
optimizer  = SGD(network.parameters(), lr=0.01, momentum=0.9)
criterion  = CrossEntropyLoss()
trainer    = lib.trainer.SupervisedTrainer(
    network,
    optimizer,
    criterion,
    {"acc": lib.metrics.accuracy}
)
trainer.train(train_loader, epochs=100, val_loader=val_loader)
