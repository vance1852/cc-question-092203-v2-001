# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、场地边界和间距约束、遗传算法与粒子群优化、经济性分析以及无界面图表输出。

## 安装

建议使用 Python 3.10 或更新版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 可以使用 `.venv\\Scripts\\Activate.ps1` 激活环境。

## 快速验证

```bash
python quick_test.py
```

快速验证会覆盖模型、约束、年发电量、优化、经济性和图表生成，并在 `test_output/` 写入临时图片。该目录不会纳入版本控制。

## 完整分析

```bash
python -m wind_farm_opt --help
python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50 --output-dir output
```

也可以先生成配置文件，再通过 `--config` 运行：

```bash
python -m wind_farm_opt --generate-config my_config.json
python -m wind_farm_opt --config my_config.json
```

所有运行结果默认写入 `output/`，可以用 `--no-plots` 跳过图表生成。命令行使用无界面绘图后端，适合容器和服务器环境。

## 布局可行性保障

规则网格、交错网格与 GA/PSO 初始化共用同一套可行性语义（`wind_farm_opt/constraints/feasibility.py`）：

- **搜索前判定**：根据真实多边形、转子直径和最小间距计算容量上界，明显无解的“台数 × 场地”组合在开始搜索前直接报错；
- **统一尝试预算**：随机补点与间距修复共享同一尝试预算，任何场地都不会整夜不返回；
- **终态校验**：所有成功返回的布局都再次通过边界与间距校验，绝不交出越界、过密或部分布局；
- **失败报告**：不可行时报告请求容量、可用面积、容量上界与首要违规原因（CLI 以退出码 2 结束）。

运行可行性回归测试（可行 / 临界 / 确定无解三类场地）：

```bash
python -m unittest discover -s tests -v
```
