# 智能排课系统 - 基于遗传算法的完整解决方案

## 系统概述

这是一个功能完备的智能排课系统，采用高级遗传算法解决复杂的课程排课优化问题。系统严格遵循教学规律，支持多种约束条件，并实现了智能的课程时段分配策略。

### 🎯 核心特性

- **智能优先级排序**：课时长度优先 + 课程性质优先，避免时间块冲突
- **课程时段智能分配**：必修课优先安排在白天黄金时段
- **严格约束检查**：零冲突保证 + 完整性验证
- **高效算法**：基于遗传算法的全局优化
- **生产就绪**：完整的日志、错误处理和结果验证

## 文件结构

```
d:\suanf\
├── suan2.py                    # 主程序入口
├── data_models.py              # 数据模型定义 (包含完整字段映射)
├── db_connector.py             # 数据库连接器 (优化后查询)
├── genetic_algorithm.py        # 遗传算法核心 (智能排序+时段偏好)
├── test_data_generator.py      # 测试数据生成器 (新字段支持)
├── verify_data.py              # 数据验证脚本
├── verify_schedule_results.py  # 排课结果验证脚本
├── check_data_scale.py         # 数据规模检查和参数推荐
├── test_optimization.py        # 算法优化测试
├── scheduling.log              # 系统运行日志
└── README.md                   # 本说明文件
```

## 环境要求

- Python 3.9+
- PyMySQL 1.0+
- MySQL 5.7+ 或 MariaDB 10.3+

## 快速安装

### 1. 克隆或下载项目文件

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install pymysql
```

## 数据库配置

### 1. 创建数据库

```sql
CREATE DATABASE `paike-c` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 2. 执行表结构 SQL

使用提供的 `表3.txt` 中的 SQL 脚本创建完整的表结构。

### 3. 数据库连接配置

默认连接参数：

- Host: localhost
- User: root
- Password: 123456
- Database: paike-c

可通过环境变量自定义：

```bash
# Windows PowerShell
$env:DB_HOST="localhost"
$env:DB_USER="root"
$env:DB_PASSWORD="123456"
$env:DB_NAME="paike-c"

# Linux/Mac
export DB_HOST="localhost"
export DB_USER="root"
export DB_PASSWORD="123456"
export DB_NAME="paike-c"
```

## 快速开始

### 场景一：使用现有数据（推荐）

如果您已经有了完整的排课数据，可以直接开始排课：

```bash
# 1. 激活虚拟环境
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. 检查数据规模并获得推荐参数
python check_data_scale.py

# 3. 验证数据完整性
python verify_data.py

# 4. 运行排课算法（使用推荐参数）
python suan2.py --version 1 --population 30 --generations 50

# 5. 验证排课结果
python verify_schedule_results.py
```

### 场景二：从零开始（测试数据）

如果您需要生成测试数据来体验系统：

```bash
# 1. 激活虚拟环境
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. 生成完整测试数据集
python test_data_generator.py

# 3. 验证数据完整性
python verify_data.py

# 4. 运行排课算法
python suan2.py --version 1 --population 30 --generations 50

# 5. 验证排课结果
python verify_schedule_results.py
```

## 命令行参数

| 参数                | 说明                | 默认值 |
| ------------------- | ------------------- | ------ |
| `--version`         | 排课版本 ID（必需） | -      |
| `--population`      | 种群大小            | 100    |
| `--generations`     | 进化代数            | 200    |
| `--crossover-rate`  | 交叉率              | 0.8    |
| `--mutation-rate`   | 变异率              | 0.1    |
| `--tournament-size` | 锦标赛大小          | 5      |
| `--elitism-size`    | 精英个体数量        | 10     |
| `--max-stagnation`  | 最大停滞代数        | 50     |

## 算法核心特性

### 🔥 智能排课优先级 (解决时间块冲突问题)

```
1. 课时长度优先：4节课 > 3节课 > 2节课
   ✓ 避免2节课占用3节课专用时间块
   ✓ 确保长课程优先获得合适的时间段

2. 课程性质优先：通识 > 必修 > 选修
   ✓ 重要课程优先安排

3. 学生人数优先：大班 > 小班
   ✓ 合班课程优先安排

4. 总学时优先：高学时 > 低学时
   ✓ 学分高的课程优先安排
```

### 🌟 课程时段智能分配

```
必修课 + 通识课：
- 优先安排在周一至周五的 1-10 节（黄金时段）
- 惩罚安排在晚上 11-13 节（惩罚 500 分）
- 惩罚安排在周末（惩罚 300 分）

选修课：
- 可以安排在任何时段（包括晚上、周末）
- 占用黄金时段会有轻微惩罚（惩罚 50 分）
```

### 🎯 严格时间块设计

```
2节连堂课：可选时间块
- 1-2节、3-4节、6-7节、9-10节、11-12节

3节连堂课：专用时间块
- 3-5节、6-8节、11-13节

4节连堂课：专用时间块
- 1-4节、6-9节

⚠️ 重要：2节课不能占用3节课的完整时间块！
```

### 🚫 硬约束（违反直接淘汰）

- ✅ **时空唯一性**：同一教师/班级/教室在同一时间只能有一门课
- ✅ **教室容量**：教室容量必须 ≥ 实际上课人数
- ✅ **设施要求**：必须满足课程的强制设施要求
- ✅ **教师黑名单**：不能在教师禁止的时间安排课程
- ✅ **周四下午锁定**：周四下午 6-13 节不安排任何课程
- ✅ **时间块完整性**：课程必须安排在合法的连续时间块内

### 🎨 软约束优化（通过惩罚分数引导）

- 🕐 **课程时段偏好**：必修课白天优先（新增功能）
- 👨‍🏫 **教师时间偏好**：尊重教师的偏好和避免时段
- 🚌 **校区通勤优化**：同一教师同一天尽量在同一校区
- 🏫 **连堂课同教室**：连续课程尽量安排在同一教室
- 📊 **教室利用率**：减少教室容量浪费
- 📚 **学生负荷均衡**：避免学生一天课程过多或过散
- 🔗 **任务关系约束**：处理课程间的时间关系要求

## 运行示例与结果

### 成功运行输出示例

```
2025-09-20 21:55:18,112 - genetic_algorithm - INFO - 开始遗传算法进化
2025-09-20 21:55:18,256 - genetic_algorithm - INFO - 第 0 代，新的最佳适应度: -10047.00
2025-09-20 21:55:18,622 - genetic_algorithm - INFO - 算法停滞 50 代，提前结束
2025-09-20 21:55:18,637 - genetic_algorithm - INFO - 进化完成，最终最佳适应度: -10047.00
2025-09-20 21:55:18,656 - __main__ - INFO - 排课覆盖率: 100.0% (9/9)
2025-09-20 21:55:18,657 - __main__ - INFO - ✓ 没有发现硬冲突
2025-09-20 21:55:18,658 - __main__ - INFO - 教室平均利用率: 136.1%
2025-09-20 21:55:18,659 - __main__ - INFO - 排课完成，总耗时: 0.56 秒
```

### 典型排课结果

```
任务1: C001(必修) - 教室CR006
  时间: 周二 3-5节 (共3节) ✓ 课时匹配

任务2: C001(必修) - 教室CR006
  时间: 周五 1-2节 (共2节) ✓ 课时匹配

任务7: C004(选修) - 教室CR005
  时间: 周二 9-10节 (共2节) ✓ 课时匹配

检查时间冲突:
  ✓ 没有教师时间冲突
  ✓ 没有教室时间冲突
```

## 性能优化建议

### 小规模数据（<50 个任务）

```bash
python suan2.py --version 1 --population 50 --generations 100
```

### 中等规模数据（50-200 个任务）

```bash
python suan2.py --version 1 --population 100 --generations 200
```

### 大规模数据（>200 个任务）

```bash
python suan2.py --version 1 --population 150 --generations 300 --max-stagnation 80
```

## 🛠️ 故障排除

### 常见问题及解决方案

#### ❌ `Field 'start_week' doesn't have a default value`

**原因**：数据库表结构版本不匹配  
**解决**：

1. 确保使用最新的 `表3.txt` SQL 脚本
2. 重新创建表结构或添加缺失字段
3. 重新运行 `test_data_generator.py`

#### ❌ `CourseOffering.__init__() got an unexpected keyword argument 'created_at'`

**原因**：数据模型与数据库字段不匹配  
**解决**：系统已修复，数据库查询只选择必要字段

#### ❌ 提示"没有找到教学任务"

**解决**：

1. 检查 `course_offerings` 表中学期数据
2. 验证 `teaching_tasks` 表有对应记录
3. 运行 `verify_data.py` 检查数据完整性

#### ❌ 算法收敛很慢或适应度很低

**解决**：

1. 增大种群大小：`--population 200`
2. 检查约束是否过于严格
3. 验证教师数量是否充足

#### ❌ 教室利用率很低

**说明**：这是正常现象，系统优先保证约束满足

## ⚙️ 系统扩展

### 添加新的软约束

1. 在 `genetic_algorithm.py` 的 `fitness()` 方法中添加检查逻辑
2. 在 `_default_config()` 中添加相应的惩罚分数
3. 可参考 `_check_course_time_preference()` 方法

### 添加新的数据字段

1. 更新 `data_models.py` 中的数据类
2. 更新 `db_connector.py` 中的 SQL 查询
3. 重新生成测试数据

### 自定义时间块规则

修改 `data_models.py` 中的 `TIME_WINDOWS` 和相关函数

### 调整课程时段偏好

修改 `genetic_algorithm.py` 中的：

- `_get_preferred_time_slots()` 方法
- `_check_course_time_preference()` 方法
- 配置中的惩罚分数

## 📊 性能基准

| 数据规模           | 推荐配置         | 预期运行时间 | 内存使用 |
| ------------------ | ---------------- | ------------ | -------- |
| 小型 (<50 任务)    | pop=50, gen=100  | <30 秒       | <100MB   |
| 中型 (50-200 任务) | pop=100, gen=200 | 1-5 分钟     | <500MB   |
| 大型 (>200 任务)   | pop=200, gen=300 | 5-30 分钟    | <1GB     |

## 🔧 开发模式

### 启用调试日志

```python
# 在 suan2.py 中修改日志级别
logging.basicConfig(level=logging.DEBUG)
```

### 快速测试小数据集

```bash
python suan2.py --version 1 --population 10 --generations 5
```

## 📜 更新日志

### v2.0 (2025-09-20)

- ✅ 解决了数据库字段不匹配问题
- ✅ 新增智能课程时段分配（必修课白天优先）
- ✅ 新增课时长度优先排序，解决时间块冲突
- ✅ 完善的结果验证和冲突检查
- ✅ 优化数据库查询性能
- ✅ 新增详细的故障排除指南

### v1.0 (初始版本)

- 基础遗传算法实现
- 硬约束和软约束处理
- 数据库集成

## 🤝 技术支持

遇到问题时的检查顺序：

1. **查看日志**：检查 `scheduling.log` 文件
2. **验证数据**：运行 `verify_data.py`
3. **检查结果**：运行 `verify_schedule_results.py`
4. **数据库连接**：确认 MySQL 服务和连接参数
5. **环境依赖**：确认 Python 版本和 PyMySQL 安装

系统采用模块化设计，便于维护和功能扩展。所有核心算法都有详细注释，支持二次开发。
