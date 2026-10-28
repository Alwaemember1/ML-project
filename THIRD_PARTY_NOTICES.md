# 第三方内容与许可范围

根目录LICENSE的MIT许可仅适用于本项目维护者有权授权的原创代码和原创说明，不授予第三方数据、预训练权重、机构商标或网页资料的新权利。

| 内容 | 本仓库处理方式 | 授权说明 |
| --- | --- | --- |
| PyTorch、torchvision、Streamlit、Pillow及其依赖 | 使用依赖清单，不复制其安装目录 | 依照各上游项目许可，使用者应核对实际分发版本的许可证 |
| ImageNet预训练ResNet18及项目微调权重 | 按维护者要求附原第3轮冻结检查点，非原创代码MIT许可范围 | 需分别核查预训练来源、数据和分发权限，不宣称项目MIT覆盖它们 |
| Plant_leave_diseases_dataset_with_augmentation | 原图不进入Git | 本地数据集上游来源与许可未在本项目中完整确认；文件夹名称不是再分发授权 |
| 大学推广和农业机构参考资料 | 保存链接及简短整理，来源在plant_care.py | 属于各自权利人，不表示相关机构为本模型诊断背书；链接可能变化 |
| 原创实验曲线与指标记录 | 保留在docs/供复核 | 实验局限见MODEL_CARD和DATA_CARD，不保证真实生产场景表现 |

发布GitHub Release前，请单独确认模型和任何图片的再分发权限。老师包是本地课程交付副本，不应不经检查直接作为公开模型Release上传。

上游参考：[PyTorch](https://github.com/pytorch/pytorch)、[torchvision](https://github.com/pytorch/vision)、[Streamlit](https://github.com/streamlit/streamlit)、[Pillow](https://github.com/python-pillow/Pillow)。
