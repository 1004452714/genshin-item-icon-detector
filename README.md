# ItemDetect

用于训练和测试原神物品图标识别模型，覆盖材料、食物、圣遗物和武器。模型导出为 `ONNX + prototypes.csv`，推理时按 `item_class_id` 聚合 TopK。

## 数据目录

仓库只提交代码、配置和 JSON 元数据，不提交训练 PNG、checkpoint、ONNX、prototypes 和测试截图。

本地资源放置：

```text
assets/
  raw/
    weapons/    原始武器导出图预存目录
  processed/
    weapons/    武器图标处理工作区，含 icon/side/review/invalid/process.log
  icons/
    items/      材料、食物图标（UI_ItemIcon_.*?）
    relics/     圣遗物图标（UI_RelicIcon_.*?）
    weapons/    武器图标（UI_EquipIcon_.*?）
  backgrounds/  背景图（UI_QUALITY_.*?）
  overlays/     美味食物叠层等合成素材

data/
  metadata/      Material.json、Reliquary.json、Weapon.json
  generated/     生成的 labels.csv
```

PNG 文件名必须保持 JSON 中的原始 `Icon` / `AwakenIcon` 名称，例如 `UI_ItemIcon_101.png`。

武器可以先放到 `assets/raw/weapons/` 预处理。

## 环境安装

推荐使用 Python 3.11。

如果需要生成标签、训练、生成 prototypes 或导出 ONNX，安装完整训练环境。`requirements.txt` 使用固定版本，但不绑定 CUDA，默认适合不同机器先完成基础安装：

```powershell
python -m pip install -r requirements.txt
```

如果需要 GPU 训练，在基础安装后按自己的显卡驱动和 CUDA 版本，从 PyTorch 官网选择对应命令重装 `torch`。例如 CUDA 12.1：

```powershell
python -m pip install --force-reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

如果已经有 `outputs/item.onnx` 和 `outputs/prototypes.csv`，只是运行单图测试、目录测试或实机测试，可以安装轻量测试环境：

```powershell
python -m pip install -r requirements-runtime.txt
```

轻量测试环境不包含 `torch` 和 `onnx`，不能执行 `prepare`、`train`、`preview`、`prototypes` 或 `export`。

## 首次使用流程

本仓库不包含训练 PNG。使用前需要自行获取图标素材，并按目录放好：

1. 把材料和食物 PNG 放入 `assets/icons/items/`（`UI_ItemIcon_.*?`）。
2. 把圣遗物 PNG 放入 `assets/icons/relics/`（`UI_RelicIcon_.*?`）。
3. 把背景 PNG 放入 `assets/backgrounds/`（`UI_QUALITY_.*?`）。
4. 把美味食物叠层 `UI_CookIcon_Delicious.png` 放入 `assets/overlays/`。
5. 武器图标先把原始导出 PNG 放入 `assets/raw/weapons/`（`UI_EquipIcon_.*?`）。
6. 在菜单选择 `8. 处理武器图标`，或运行 `.\itemdetect.ps1 process-weapons`，脚本会把结果分到 `assets/processed/weapons/icon|side|review|invalid/`。
7. 人工复查 `review/`，把确认可训练的图标重命名为 `_ICON.png` 后缀；终端输入 `APPLY` 后才会复制到 `assets/icons/weapons/`，复制时自动去掉 `_ICON` 后缀。
8. 素材准备完成后，菜单选择 `1. 准备数据` 校验标签和图片。
9. 校验通过后，菜单选择 `2. 开始完整训练`。

## 常用命令

不传参数会进入菜单：

```powershell
.\itemdetect.ps1
```

准备数据：

```powershell
.\itemdetect.ps1 prepare
```

从头训练并导出：

```powershell
.\itemdetect.ps1 train
```

单图测试：

```powershell
.\itemdetect.ps1 infer --image test\a.png
```

目录测试：

```powershell
.\itemdetect.ps1 test --image-dir test
```

实机窗口测试：

```powershell
.\itemdetect.ps1 live
```

处理武器图标：

```powershell
.\itemdetect.ps1 process-weapons
```

如果没有激活 conda 环境，可以指定 Python：

```powershell
.\itemdetect.ps1 -Python C:\ProgramData\anaconda3\envs\avatardetect\python.exe train
```

## 步骤说明

菜单中的每个步骤会执行以下操作：

- `1. 准备数据`：生成 `data/generated/labels.csv`，检查 Python 语法，并按 `configs/train.yaml` 校验图片、背景、标签字段是否可用。对应命令：`.\itemdetect.ps1 prepare`。
- `2. 开始完整训练`：先准备数据，再清理旧的全量训练产物，然后从头训练，最后生成 `outputs/prototypes.csv` 并导出 `outputs/item.onnx`。对应命令：`.\itemdetect.ps1 train`。
- `3. 单图测试`：提示输入一张图片路径，使用 `outputs/item.onnx` 和 `outputs/prototypes.csv` 输出 TopK 识别结果。对应命令：`.\itemdetect.ps1 infer --image test\a.png`。
- `4. 目录测试`：提示输入图片目录，默认 `test/`；批量测试目录下的 PNG，并输出每张图的推理耗时、总图片数和平均耗时。对应命令：`.\itemdetect.ps1 test --image-dir test`。
- `5. 实机测试`：查找原神窗口，实时截图、定位物品格子、裁剪图标并识别；可在控制窗口调整 HSV、面积和裁剪参数。对应命令：`.\itemdetect.ps1 live`。
- `6. 生成预览图`：根据当前 labels 和配置生成合成预览图到 `outputs/previews/`，用于检查图标、背景、叠层和增强效果。对应命令：`.\itemdetect.ps1 preview`。
- `7. 清理生成物`：删除生成物和缓存，包括 `outputs/`、`data/generated/labels.csv`、`temp/` 和 `__pycache__/`；不会删除 `assets/` 下的训练 PNG。对应命令：`.\itemdetect.ps1 clean`。
- `8. 处理武器图标`：读取 `assets/raw/weapons/` 的原始武器 PNG，区分 `ICON`、`SIDE`、`review` 和 `invalid`，结果写到 `assets/processed/weapons/`，只生成 `process.log`，不生成 CSV 或 HTML 报告。人工复查后，把正确文件改名为 `_ICON.png`，在终端输入 `APPLY` 才会复制到 `assets/icons/weapons/`，复制时会去掉 `_ICON` 后缀。对应命令：`.\itemdetect.ps1 process-weapons`。

高级命令：

- `prototypes`：用指定 checkpoint 重新生成 `prototypes.csv`，通常在训练完成后使用。
- `export`：用指定 checkpoint 重新导出 ONNX，通常在训练完成后使用。
- `train --skip-cleanup`：执行完整训练流程，但训练前不清理旧的 `outputs/checkpoints/`、`outputs/prototypes.csv`、`outputs/item.onnx`。

## 识别规则

- 主类别使用 `item_class_id`。
- 食物普通、奇怪、美味作为不同类别训练。
- 武器普通图标和等级达到上限后的满级图标合并为同一类别，推理时通过最接近的 prototype 输出 `normal/awaken`。
- 圣遗物按图标对应的物品类别区分，不使用 `EquipType`。
- 当前仅圣遗物参与品质辅助训练；其他类别主要通过随机背景和限定遮挡增强提升实机鲁棒性。
