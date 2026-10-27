# 参与开发

本项目是固定39类植物叶片分类与参考资料展示项目。提交前请阅读根目录README、模型说明和数据说明。原创代码采用MIT；第三方数据、权重和资料不自动继承该许可。

## 本地检查

安装README中的CPU或GPU依赖后，从项目根目录执行：

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.\.venv\Scripts\python.exe' -m scripts.prepare_formal_split --metadata-only
```

默认测试只使用合成图片及元数据，不需要权重或原始数据。真实模型上传集成检查需要自行取得冻结权重并显式设置 `LEAF_RUN_MODEL_TESTS=1`；不得在CI下载整套数据或重新跑最终测试。

## 提交约定

- 功能改动附上触发条件、实际行为、验证方法及限制。文档命令改动应检查对应CLI的 `--help`。
- 不提交.venv、权重、原图、上传文件、缓存、私钥、Token、IDE和代理状态。
- `splits/`中的原始字节受SHA256锁保护；不要为换行或格式整理重写它们，不静默重建划分。
- 原FINAL_MODEL.json和最终测试记录是历史实验身份。新部署用FINAL_MODEL.local.json，新实验用独立目录，不把旧测试成绩转给新模型。
- 模型预测不准确不等于实现错误。不得以提高分数为由重复查看测试集调参，也不得把颜色阈值或分类概率宣称成经过验证的严重程度。
- 引用防治资料须保留来源和核对日期；不要捏造登记药剂、剂量或田间验证结果。

## 目录约定

根目录放Streamlit入口和共用逻辑，CLI放 `scripts/`，测试放 `tests/`，结果证据和资料放 `docs/`。新增脚本使用 `python -m scripts.<名称>` 运行，保持main入口保护；禁止在导入模块时启动训练、下载或评价数据集。
