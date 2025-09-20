# 智能排课系统运行说明

## 系统概述

这是一个基于遗传算法的智能排课系统，能够处理复杂的排课约束和优化目标。

## 文件结构

```
d:\suanf\
├── suan2.py                 # 主程序入口
├── data_models.py           # 数据模型定义
├── db_connector.py          # 数据库连接器
├── genetic_algorithm.py     # 遗传算法核心
├── test_data_generator.py   # 测试数据生成器
└── README.md               # 本说明文件
```

## 环境要求

- Python 3.9+
- PyMySQL
- MySQL 5.7+

## 安装依赖

```bash
pip install pymysql
```

## 数据库配置

1. 确保 MySQL 服务正在运行
2. 创建数据库 `paike2`
3. 执行提供的 SQL 脚本创建表结构
4. 设置环境变量（可选）：

```bash
# Windows PowerShell
$env:DB_HOST="localhost"
$env:DB_USER="root"
$env:DB_PASSWORD="123456"
$env:DB_NAME="paike2"

# Linux/Mac
export DB_HOST="localhost"
export DB_USER="root"
export DB_PASSWORD="123456"
export DB_NAME="paike2"
```

## 快速开始

### 1. 生成测试数据

```bash
python test_data_generator.py
```

### 2. 运行排课算法

```bash
# 基本运行
python suan2.py --version 1

# 自定义参数运行
python suan2.py --version 1 --population 80 --generations 150 --mutation-rate 0.15
```

### 3. 查看结果

运行完成后，结果会保存在 `schedules` 表中。可以使用以下 SQL 查询：

```sql
-- 查看排课结果
SELECT s.*, t.offering_id, t.task_sequence, c.course_name
FROM schedules s
JOIN teaching_tasks t ON s.task_id = t.task_id
JOIN course_offerings o ON t.offering_id = o.offering_id
JOIN courses c ON o.course_id = c.course_id
WHERE s.version_id = 1
ORDER BY s.week_day, s.start_slot;

-- 检查冲突
SELECT
    s.week_day, s.start_slot, s.classroom_id,
    COUNT(*) as conflict_count
FROM schedules s
WHERE s.version_id = 1
GROUP BY s.week_day, s.start_slot, s.classroom_id
HAVING COUNT(*) > 1;
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

## 算法特点

### 硬约束

- 教师、班级、教室时间冲突检测
- 教室容量限制
- 特殊设施要求匹配
- 教师禁止时间
- 周四下午锁定

### 软约束优化

- **课程时段偏好**：必修课优先安排在白天，选修课可安排在晚上
- 教师偏好时段
- 校区通勤最小化
- 连堂课同教室偏好
- 教室利用率优化
- 学生课程负荷均衡

### 智能排课优先级

- **课时长度优先**：4 节课 > 3 节课 > 2 节课，避免短课程占用长课程的时间块
- **课程性质优先**：通识 > 必修 > 选修
- **学生人数优先**：大班优先安排
- **总学时优先**：学时多的课程优先安排

### 时间块设计

- 2 节课：1-2, 3-4, 6-7, 9-10, 11-12
- 3 节课：3-5, 6-8, 11-13
- 4 节课：1-4, 6-9

## 日志输出

系统会将运行日志输出到：

- 控制台
- `scheduling.log` 文件

日志包含：

- 数据加载进度
- 算法进化过程
- 最终结果统计
- 错误信息和调试信息

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

## 常见问题

### Q: 提示"没有找到教学任务"

A: 检查 `course_offerings` 表中是否有对应学期的数据，以及 `teaching_tasks` 表是否有关联的任务。

### Q: 算法收敛很慢

A: 尝试增大种群大小，或者降低变异率。

### Q: 出现大量硬冲突

A: 检查数据完整性，特别是教师数量是否充足，教室容量是否合理。

### Q: 教室利用率很低

A: 这是正常现象，系统会优先保证约束满足，再优化利用率。

## 系统扩展

### 添加新的约束

1. 在 `genetic_algorithm.py` 的适应度函数中添加检查逻辑
2. 在配置中添加相应的惩罚分数

### 添加新的数据字段

1. 更新 `data_models.py` 中的数据类
2. 更新 `db_connector.py` 中的加载逻辑

### 调整算法参数

系统支持运行时参数调整，无需修改代码。

## 技术支持

如果遇到问题，请检查：

1. 数据库连接是否正常
2. 数据完整性是否通过验证
3. 日志文件中的错误信息

系统设计遵循模块化原则，便于维护和扩展。
